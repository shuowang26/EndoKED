import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import LCA_blcok, ESA_blcok
from models.res2net import res2net50_v1b_26w_4s


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DecoderBlock, self).__init__()

        self.conv1 = ConvBlock(in_channels, in_channels // 4, kernel_size=kernel_size, stride=stride, padding=padding)

        self.conv2 = ConvBlock(in_channels // 4, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.upsample(x)
        return x


class HeadUpdator(nn.Module):
    def __init__(self, in_channels=64, feat_channels=64, out_channels=None, conv_kernel_size=1):
        super(HeadUpdator, self).__init__()
        
        self.conv_kernel_size = conv_kernel_size

        # C == feat
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.out_channels = out_channels if out_channels else in_channels
        # feat == in == out
        self.num_in = self.feat_channels
        self.num_out = self.feat_channels

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.pred_transform_layer = nn.Linear(self.in_channels, self.num_in + self.num_out)
        self.head_transform_layer = nn.Linear(self.in_channels, self.num_in + self.num_out, 1)

        self.pred_gate = nn.Linear(self.num_in, self.feat_channels, 1)
        self.head_gate = nn.Linear(self.num_in, self.feat_channels, 1)

        self.pred_norm_in = nn.LayerNorm(self.feat_channels)
        self.head_norm_in = nn.LayerNorm(self.feat_channels)
        self.pred_norm_out = nn.LayerNorm(self.feat_channels)
        self.head_norm_out = nn.LayerNorm(self.feat_channels)

        self.fc_layer = nn.Linear(self.feat_channels, self.out_channels, 1)
        self.fc_norm = nn.LayerNorm(self.feat_channels)
        self.activation = nn.ReLU(inplace=True)


    def forward(self, feat, head, pred):

        bs, num_classes = head.shape[:2]
        # C, H, W = feat.shape[-3:]

        pred = self.upsample(pred)
        pred = torch.sigmoid(pred)

        """
        Head feature assemble 
        - use prediction to assemble head-aware feature
        """

        # [B, N, C]
        assemble_feat = torch.einsum('bnhw,bchw->bnc', pred, feat)

        # [B, N, C, K, K] -> [B, N, C, K*K] -> [B, N, K*K, C]
        head = head.reshape(bs, num_classes, self.in_channels, -1).permute(0, 1, 3, 2)
        
        """
        Update head
        - assemble_feat, head -> linear transform -> pred_feat, head_feat
        - both split into two parts: xxx_in & xxx_out
        - gate_feat = head_feat_in * pred_feat_in
        - gate_feat -> linear transform -> pred_gate, head_gate
        - update_head = pred_gate * pred_feat_out + head_gate * head_feat_out
        """
        # [B, N, C] -> [B*N, C]
        assemble_feat = assemble_feat.reshape(-1, self.in_channels)
        bs_num = assemble_feat.size(0)

        # [B*N, C] -> [B*N, in+out]
        pred_feat = self.pred_transform_layer(assemble_feat)
        
        # [B*N, in]
        pred_feat_in = pred_feat[:, :self.num_in].view(-1, self.feat_channels)
        # [B*N, out]
        pred_feat_out = pred_feat[:, -self.num_out:].view(-1, self.feat_channels)

        # [B, N, K*K, C] -> [B*N, K*K, C] -> [B*N, K*K, in+out]
        head_feat = self.head_transform_layer(
            head.reshape(bs_num, -1, self.in_channels))

        # [B*N, K*K, in]
        head_feat_in = head_feat[..., :self.num_in]
        # [B*N, K*K, out]
        head_feat_out = head_feat[..., -self.num_out:]

        # [B*N, K*K, in] * [B*N, 1, in] -> [B*N, K*K, in]
        gate_feat = head_feat_in * pred_feat_in.unsqueeze(-2)

        # [B*N, K*K, feat]
        head_gate = self.head_norm_in(self.head_gate(gate_feat))
        pred_gate = self.pred_norm_in(self.pred_gate(gate_feat))

        head_gate = torch.sigmoid(head_gate)
        pred_gate = torch.sigmoid(pred_gate)

        # [B*N, K*K, out]
        head_feat_out = self.head_norm_out(head_feat_out)
        # [B*N, out]
        pred_feat_out = self.pred_norm_out(pred_feat_out)

        # [B*N, K*K, feat] or [B*N, K*K, C]
        update_head = pred_gate * pred_feat_out.unsqueeze(-2) + head_gate * head_feat_out

        update_head = self.fc_layer(update_head)
        update_head = self.fc_norm(update_head)
        update_head = self.activation(update_head)

        # [B*N, K*K, C] -> [B, N, K*K, C]
        update_head = update_head.reshape(bs, num_classes, -1, self.feat_channels)
        # [B, N, K*K, C] -> [B, N, C, K*K] -> [B, N, C, K, K]
        update_head = update_head.permute(0, 1, 3, 2).reshape(bs, num_classes, self.feat_channels, self.conv_kernel_size, self.conv_kernel_size)

        return update_head


class LDNet(nn.Module):
    def __init__(self, num_classes=1, unified_channels=64, conv_kernel_size=1):
        super(LDNet, self).__init__()
        self.num_classes = num_classes
        self.conv_kernel_size = conv_kernel_size
        self.unified_channels = unified_channels

        res2net = res2net50_v1b_26w_4s(pretrained=True)
        
        # Encoder
        self.encoder1_conv = res2net.conv1
        self.encoder1_bn = res2net.bn1
        self.encoder1_relu = res2net.relu
        self.maxpool = res2net.maxpool
        self.encoder2 = res2net.layer1
        self.encoder3 = res2net.layer2
        self.encoder4 = res2net.layer3
        self.encoder5 = res2net.layer4

        self.reduce2 = nn.Conv2d(256, 64, 1)
        self.reduce3 = nn.Conv2d(512, 128, 1)
        self.reduce4 = nn.Conv2d(1024, 256, 1)
        self.reduce5 = nn.Conv2d(2048, 512, 1)
        # Decoder
        self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=512+256, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=256+128, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=128+64, out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=64+64, out_channels=64)

        # self.outconv = nn.Sequential(
        #     ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
        #     nn.Dropout2d(0.1),
        #     nn.Conv2d(32, num_classes, 1)
        # )

        self.gobal_average_pool = nn.Sequential(
            nn.GroupNorm(16, 512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        #self.gobal_average_pool = nn.AdaptiveAvgPool2d(1)
        self.generate_head = nn.Linear(512, self.num_classes*self.unified_channels*self.conv_kernel_size*self.conv_kernel_size)

        # self.pred_head = nn.Conv2d(64, self.num_classes, self.conv_kernel_size)

        self.headUpdators = nn.ModuleList()
        for i in range(4):
            self.headUpdators.append(HeadUpdator())

        # Unified channel
        self.unify1 = nn.Conv2d(64, 64, 1)
        self.unify2 = nn.Conv2d(64, 64, 1)
        self.unify3 = nn.Conv2d(128, 64, 1)
        self.unify4 = nn.Conv2d(256, 64, 1)
        self.unify5 = nn.Conv2d(512, 64, 1)

        # Efficient self-attention block
        self.esa1 = ESA_blcok(dim=64)
        self.esa2 = ESA_blcok(dim=64)
        self.esa3 = ESA_blcok(dim=128)
        self.esa4 = ESA_blcok(dim=256)
        #self.esa5 = ESA_blcok(dim=512)
        # Lesion-aware cross-attention block
        self.lca1 = LCA_blcok(dim=64)
        self.lca2 = LCA_blcok(dim=128)
        self.lca3 = LCA_blcok(dim=256)
        self.lca4 = LCA_blcok(dim=512)

        self.decoderList = nn.ModuleList([self.decoder4, self.decoder3, self.decoder2, self.decoder1])
        self.unifyList = nn.ModuleList([self.unify4, self.unify3, self.unify2, self.unify1])
        self.esaList = nn.ModuleList([self.esa4, self.esa3, self.esa2, self.esa1])
        self.lcaList = nn.ModuleList([self.lca4, self.lca3, self.lca2, self.lca1])


    def forward(self, x):
        # x = H*W*3
        bs = x.shape[0]
        e1_ = self.encoder1_conv(x)  # H/2*W/2*64
        e1_ = self.encoder1_bn(e1_)
        e1_ = self.encoder1_relu(e1_)
        e1_pool_ = self.maxpool(e1_)  # H/4*W/4*64
        e2_ = self.encoder2(e1_pool_) # H/4*W/4*64
        e3_ = self.encoder3(e2_)      # H/8*W/8*128
        e4_ = self.encoder4(e3_)      # H/16*W/16*256
        e5_ = self.encoder5(e4_)      # H/32*W/32*512
        
        e1 = e1_
        e2 = self.reduce2(e2_)
        e3 = self.reduce3(e3_)
        e4 = self.reduce4(e4_)
        e5 = self.reduce5(e5_)
        
        #e5 = self.esa5(e5)
        d5 = self.decoder5(e5)      # H/16*W/16*512
        
        feat5 = self.unify5(d5)

        decoder_out = [d5]
        encoder_out = [e4, e3, e2, e1]

        """
        B = batch size (bs)
        N = number of classes (num_classes)
        C = feature channels
        K = conv kernel size
        """
        # [B, 512, 1, 1] -> [B, 512]
        gobal_context = self.gobal_average_pool(e5)
        gobal_context = gobal_context.reshape(bs, -1)
        
        # [B, N*C*K*K] -> [B, N, C, K, K]
        head = self.generate_head(gobal_context)
        head = head.reshape(bs, self.num_classes, self.unified_channels, self.conv_kernel_size, self.conv_kernel_size)
        
        pred = []
        for t in range(bs):
            pred.append(F.conv2d(
                feat5[t:t+1],
                head[t],
                padding=int(self.conv_kernel_size // 2)))
        pred = torch.cat(pred, dim=0)
        H, W = feat5.shape[-2:]
        # [B, N, H, W]
        pred = pred.reshape(bs, self.num_classes, H, W)
        stage_out = [pred]

        # feat size: [B, C, H, W]
        # feats = [feat4, feat3, feat2, feat1]
        feats = []

        for i in range(4):
            esa_out = self.esaList[i](encoder_out[i])
            lca_out = self.lcaList[i](decoder_out[-1], stage_out[-1])
            comb = torch.cat([lca_out, esa_out], dim=1)
            
            d = self.decoderList[i](comb)
            decoder_out.append(d)
            
            feat = self.unifyList[i](d)
            feats.append(feat)

            head = self.headUpdators[i](feats[i], head, pred)
            pred = []

            for j in range(bs):
                pred.append(F.conv2d(
                    feats[i][j:j+1],
                    head[j],
                    padding=int(self.conv_kernel_size // 2)))
            pred = torch.cat(pred, dim=0)
            H, W = feats[i].shape[-2:]
            pred = pred.reshape(bs, self.num_classes, H, W)
            stage_out.append(pred)
            
        stage_out.reverse()
        #return stage_out[0], stage_out[1], stage_out[2], stage_out[3], stage_out[4]
        return torch.sigmoid(stage_out[0]), torch.sigmoid(stage_out[1]), torch.sigmoid(stage_out[2]), \
               torch.sigmoid(stage_out[3]), torch.sigmoid(stage_out[4])
