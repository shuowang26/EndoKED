<img width="1206" alt="image" src="https://github.com/shuowang26/EndoKED/assets/51802231/87ab3ddb-5fea-421a-93e6-4666974a2112">


# EndoKED
The official codes for **Knowledge Extraction and Distillation from Large-Scale Image-Text Colonoscopy Reports Leveraging Large Language and Vision Models**.

## Dependencies

To clone all files:

```
git clone -i https://github.com/zwyang6/ENDOKED.git
```

To install Python dependencies:

```
pip install -r requirements.txt

```

## Datasets

#### **Evaluation Dataset**   

EndoKED is evaluated on six public out-of-domain datasets, i.e., [CVC-ClinicDB](https://www.sciencedirect.com/science/article/abs/pii/S0895611115000567), [Kvasir-SEG](https://link.springer.com/chapter/10.1007/978-3-030-37734-2_37), [ETIS](https://link.springer.com/article/10.1007/s11548-013-0926-3), [CVC-ColonDB](https://ieeexplore.ieee.org/abstract/document/7294676/), [CVC-300](https://www.hindawi.com/journals/jhe/2017/4037190/), and the multicentre dataset [PolyGen](https://www.nature.com/articles/s41597-023-01981-y) (including both sequence and frame images). EndoKED is also evaluated on the small polyp subset in PolypGen (i.e., PolypGen-Small), which is more challenging to locate and segment. 

To download these six datasets, you may refer to the corresponding papers or directly download them [HERE]()

| Dataset       | Year |     Resolution    | Training | Testing | Total |
|---------------|:----:|:-----------------:|:--------:|:-------:|:-----:|
| CVC-ClinincDB | 2015 |      384x384      |    550   |    62   |  612  |
| Kvasir-SEG    | 2020 | 332x487~1920x1072 |    900   |   100   |  1000 |
| ETIS          | 2014 |      1225x966     |    N/A   |   196   |  196  |
| CVC-ColonDB   | 2016 |      574x500      |    N/A   |   380   |  380  |
| CVC-300       | 2017 |      574x500      |    N/A   |    60   |   60  |
| PolypGen-Frame| 2023 | 228x384~1080x1920 |    N/A   |    1537 |   1537|
| PolypGen-Video| 2023 | 576x720~1080x1920 |    N/A   |    1710 |   1710|
| PolypGen-Small| 2023 | 228x384~1080x1920 |    N/A   |    93   |   93  |

## Pretrained models

#### EndoKED Full Checkpoints
Due to hospital confidentiality agreements, we are currently unable to release the training dataset. However, based on our EndoKED dataset, we release the checkpoint of our EndoKED-SEG model, which shows exceptional generalisation ability across the six public datasets.

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
</style>
<table class="tg"><thead>
  <tr>
    <th class="tg-c3ow" rowspan="2"><br>Model</th>
    <th class="tg-c3ow" colspan="2">Download</th>
    <th class="tg-c3ow" colspan="8">Performance</th>
  </tr>
  <tr>
    <th class="tg-c3ow">Checkpoints</th>
    <th class="tg-c3ow">Logs</th>
    <th class="tg-c3ow">Kvasir-SEG</th>
    <th class="tg-c3ow">CVC-ClinicDB</th>
    <th class="tg-c3ow">CVC-ColonDB</th>
    <th class="tg-c3ow">CVC-300</th>
    <th class="tg-c3ow">ETIS</th>
    <th class="tg-c3ow">PolypGen-Frame</th>
    <th class="tg-c3ow">PolyGen-Video</th>
    <th class="tg-c3ow">PolyGen-Small</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-c3ow">EndoKED</td>
    <td class="tg-c3ow"><a href="https://drive.google.com/drive/folders/1Ywg4exzgZTGg_2rRn8Qao-CSCSMiaiTG?usp=sharing" target="_blank" rel="noopener noreferrer">full_ckpt</a></td>
    <td class="tg-c3ow"><a href="https://drive.google.com/file/d/1iNpzCAN_MUhpaVA0VRO49McheA3cZQhi/view?usp=sharing" target="_blank" rel="noopener noreferrer">log</a></td>
    <td class="tg-c3ow">0.817 </td>
    <td class="tg-c3ow">0.794 </td>
    <td class="tg-c3ow">0.622 </td>
    <td class="tg-c3ow">0.849 </td>
    <td class="tg-c3ow">0.537 </td>
    <td class="tg-c3ow">0.656 </td>
    <td class="tg-c3ow">0.453 </td>
    <td class="tg-c3ow">0.202 </td>
  </tr>

</tbody></table>

#### Other Baseline Models
Moreover, we have pre-trained 9 powerful baseline models (including CNN- and ViT-based architectures), i.e., [Unet](https://github.com/milesial/Pytorch-UNet), [Unet++](https://github.com/4uiiurz1/pytorch-nested-unet), [C2FNet](https://github.com/thograce/C2FNet.git), [DCRNet](https://github.com/PRIS-CV/DCRNet.git), [LDNet](https://github.com/ReaFly/LDNet.git), [Polyp-PVT](https://github.com/DengPingFan/Polyp-PVT), [FCBFormer](https://github.com/ESandML/FCBFormer.git), [Polyp-CASCADE](https://github.com/SLDGroup/CASCADE.git), and [PIDNet (Lightweigh)](https://github.com/XuJiacong/PIDNet.git). We have publicly released the pretrained checkpoints for further research and reproducibility. 

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-9wq8{border-color:inherit;text-align:center;vertical-align:middle}
</style>
<table class="tg"><thead>
  <tr>
    <th class="tg-9wq8">Model</th>
    <th class="tg-9wq8">Unet</th>
    <th class="tg-9wq8">Unet++</th>
    <th class="tg-9wq8">C2FNet</th>
    <th class="tg-9wq8">DCRNet</th>
    <th class="tg-9wq8">LDNet</th>
    <th class="tg-9wq8">Polyp-PVT</th>
    <th class="tg-9wq8">FCBFormer</th>
    <th class="tg-9wq8">Polyp-CASCADE</th>
    <th class="tg-9wq8">PIDNet(lightweight)</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-9wq8">Checkpoints</td>
    <td class="tg-9wq8"><a href="https://drive.google.com/file/d/1aeogerGkW4132xZ9oBbPMdif9XEWYSkt/view?usp=sharing" target="_blank" rel="noopener noreferrer">full_ckpt</a></td>
    <td class="tg-9wq8"><a href="https://drive.google.com/file/d/1D2bB_2Aq7uMO5g5FEvcaEq8Zava39J46/view?usp=sharing" target="_blank" rel="noopener noreferrer">full_ckpt</a></td>
    <td class="tg-9wq8"><a href="https://drive.google.com/file/d/1KWjUcNbQTLd7qm8WAP88jp0OmGX3s1D9/view?usp=sharing" target="_blank" rel="noopener noreferrer">full_ckpt</a></td>
    <td class="tg-9wq8"><a href="https://drive.google.com/file/d/13Kv1GZUfq8_qmY1xsuFDcqy3fEUMwRL2/view?usp=sharing" target="_blank" rel="noopener noreferrer">full_ckpt</a></td>
    <td class="tg-9wq8"><a href="https://drive.google.com/file/d/13Kv1GZUfq8_qmY1xsuFDcqy3fEUMwRL2/view?usp=sharing" target="_blank" rel="noopener noreferrer">full_ckpt</a></td>
    <td class="tg-9wq8"><a href="https://drive.google.com/file/d/1UMytqnz3g1S0_yyrSJGFIYVvUrVHjoWq/view?usp=sharing" target="_blank" rel="noopener noreferrer">full_ckpt</a></td>
    <td class="tg-9wq8"><a href="https://drive.google.com/file/d/1MA0pj4dAu19uCanVnNHWw9QxwbnX0Gi0/view?usp=sharing" target="_blank" rel="noopener noreferrer">full_ckpt</a></td>
    <td class="tg-9wq8"><a href="https://drive.google.com/file/d/1dcCsobeNvM1IIL99q40poWkUZ1_G4Wqp/view?usp=sharing" target="_blank" rel="noopener noreferrer">full_ckpt</a></td>
    <td class="tg-9wq8"><a href="https://drive.google.com/file/d/18ETF_b4Jz4rwrL4Kawv57rtiPcICVTTp/view?usp=sharing" target="_blank" rel="noopener noreferrer">full_ckpt</a></td>
  </tr>
</tbody></table>

## Semantic Results

With the pre-trained segmentation models using EndoKED annotation, we combine the training set from CVC-ClinicDB and Kvasir-SEG as the final training set and evaluate its effectiveness in the testing set of the six public datasets. It demonstrate that new SOTA performance and better generalisation ability of supervised models can be achieved, with a significant gain compared with pre-training on ImageNet.

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
</style>
<table class="tg"><thead>
  <tr>
    <th class="tg-c3ow" rowspan="2"><br>Model</th>
    <th class="tg-c3ow" colspan="2">Download</th>
    <th class="tg-c3ow" colspan="8">Performance</th>
  </tr>
  <tr>
    <th class="tg-c3ow">Checkpoints</th>
    <th class="tg-c3ow">Logs</th>
    <th class="tg-c3ow">Kvasir-SEG</th>
    <th class="tg-c3ow">CVC-ClinicDB</th>
    <th class="tg-c3ow">CVC-ColonDB</th>
    <th class="tg-c3ow">CVC-300</th>
    <th class="tg-c3ow">ETIS</th>
    <th class="tg-c3ow">PolypGen-Frame</th>
    <th class="tg-c3ow">PolyGen-Video</th>
    <th class="tg-c3ow">PolyGen-Small</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-c3ow">Unet</td>
    <td class="tg-c3ow"><a href="https://drive.google.com/file/d/1vcqitDIlu76b4evmsLoY1RiOg83YSj-q/view?usp=sharing" target="_blank" rel="noopener noreferrer">full_ckpt</a></td>
    <td class="tg-c3ow"><a href="https://drive.google.com/file/d/1iNpzCAN_MUhpaVA0VRO49McheA3cZQhi/view?usp=sharing" target="_blank" rel="noopener noreferrer">log</a></td>
    <td class="tg-c3ow">0.817 </td>
    <td class="tg-c3ow">0.794 </td>
    <td class="tg-c3ow">0.622 </td>
    <td class="tg-c3ow">0.849 </td>
    <td class="tg-c3ow">0.537 </td>
    <td class="tg-c3ow">0.656 </td>
    <td class="tg-c3ow">0.453 </td>
    <td class="tg-c3ow">0.202 </td>
  </tr>
  <tr>
    <td class="tg-c3ow">Unet++</td>
    <td class="tg-c3ow"><a href="https://drive.google.com/file/d/1FrnTps6E6gyF0xOxZUzJXEbD46i42lEl/view?usp=sharing" target="_blank" rel="noopener noreferrer">full_ckpt</a></td>
    <td class="tg-c3ow"><a href="https://drive.google.com/file/d/1LuwIJndEjYkBj1mY-WJAalNage1ddtDP/view?usp=sharing" target="_blank" rel="noopener noreferrer">log</a></td>
    <td class="tg-c3ow">0.833 </td>
    <td class="tg-c3ow">0.792 </td>
    <td class="tg-c3ow">0.616 </td>
    <td class="tg-c3ow">0.814 </td>
    <td class="tg-c3ow">0.524 </td>
    <td class="tg-c3ow">0.657 </td>
    <td class="tg-c3ow">0.454 </td>
    <td class="tg-c3ow">0.196 </td>
  </tr>
  <tr>
    <td class="tg-c3ow">C2FNet</td>
    <td class="tg-c3ow"><a href="https://drive.google.com/file/d/1emQcF4nkNOFi1f4ffa1HPicl6Q27ccGM/view?usp=sharing" target="_blank" rel="noopener noreferrer">full_ckpt</a></td>
    <td class="tg-c3ow"><a href="https://drive.google.com/file/d/1WcBgtL_qTqCLpEPU8cN5SVc66txMixe2/view?usp=sharing" target="_blank" rel="noopener noreferrer">log</a></td>
    <td class="tg-c3ow">0.913 </td>
    <td class="tg-c3ow">0.920 </td>
    <td class="tg-c3ow">0.809 </td>
    <td class="tg-c3ow">0.886 </td>
    <td class="tg-c3ow">0.811 </td>
    <td class="tg-c3ow">0.762 </td>
    <td class="tg-c3ow">0.703 </td>
    <td class="tg-c3ow">0.504 </td>
  </tr>
  <tr>
    <td class="tg-c3ow">DCRNet</td>
    <td class="tg-c3ow"><a href="https://drive.google.com/file/d/1xGe1F7b3nDpRv9yYrnrN_v51_Bj9GKis/view?usp=sharing" target="_blank" rel="noopener noreferrer">full_ckpt</a></td>
    <td class="tg-c3ow"><a href="https://drive.google.com/file/d/1V4NXchcemFzFQ3Ru5vSKFeMpDvjcdiF6/view?usp=sharing" target="_blank" rel="noopener noreferrer">log</a></td>
    <td class="tg-c3ow">0.888 </td>
    <td class="tg-c3ow">0.901 </td>
    <td class="tg-c3ow">0.749 </td>
    <td class="tg-c3ow">0.870 </td>
    <td class="tg-c3ow">0.799 </td>
    <td class="tg-c3ow">0.742 </td>
    <td class="tg-c3ow">0.684 </td>
    <td class="tg-c3ow">0.593 </td>
  </tr>
  <tr>
    <td class="tg-c3ow">LDNet</td>
    <td class="tg-c3ow"><a href="https://drive.google.com/file/d/1ydpNB8KjltgsbbZQlMPa04ztVVWRuRsu/view?usp=sharing" target="_blank" rel="noopener noreferrer">full_ckpt</a></td>
    <td class="tg-c3ow"><a href="https://drive.google.com/file/d/1Hjl3G5Wqz-7MK9bweYn9d2fADK_kfgot/view?usp=sharing" target="_blank" rel="noopener noreferrer">log</a></td>
    <td class="tg-c3ow">0.908 </td>
    <td class="tg-c3ow">0.905 </td>
    <td class="tg-c3ow">0.798 </td>
    <td class="tg-c3ow">0.902 </td>
    <td class="tg-c3ow">0.793 </td>
    <td class="tg-c3ow">0.747 </td>
    <td class="tg-c3ow">0.655 </td>
    <td class="tg-c3ow">0.420 </td>
  </tr>
  <tr>
    <td class="tg-c3ow">Polyp-PVT</td>
    <td class="tg-c3ow"><a href="https://drive.google.com/file/d/1SMrP9X9PsV6vO6m9yvGs7ZHIr4IKfMaK/view?usp=sharing" target="_blank" rel="noopener noreferrer">full_ckpt</a></td>
    <td class="tg-c3ow"><a href="https://drive.google.com/file/d/1GTcDQQJZRDPqhlccn_82BSQZBN0NtluI/view?usp=sharing" target="_blank" rel="noopener noreferrer">log</a></td>
    <td class="tg-c3ow">0.923 </td>
    <td class="tg-c3ow">0.937 </td>
    <td class="tg-c3ow">0.808 </td>
    <td class="tg-c3ow">0.900 </td>
    <td class="tg-c3ow">0.835 </td>
    <td class="tg-c3ow">0.777 </td>
    <td class="tg-c3ow">0.693 </td>
    <td class="tg-c3ow">0.585 </td>
  </tr>
  <tr>
    <td class="tg-c3ow">FCBFormer</td>
    <td class="tg-c3ow"><a href="https://drive.google.com/file/d/1ttNr5IKI50cyvE9ayeyTYtxkza6_qpK4/view?usp=sharing" target="_blank" rel="noopener noreferrer">full_ckpt</a></td>
    <td class="tg-c3ow"><a href="https://drive.google.com/file/d/1JfZkGSwsWAlp3VschK3r-_bUqgDG0Lag/view?usp=sharing" target="_blank" rel="noopener noreferrer">log</a></td>
    <td class="tg-c3ow">0.914 </td>
    <td class="tg-c3ow">0.912 </td>
    <td class="tg-c3ow">0.812 </td>
    <td class="tg-c3ow">0.897 </td>
    <td class="tg-c3ow">0.821 </td>
    <td class="tg-c3ow">0.761 </td>
    <td class="tg-c3ow">0.646 </td>
    <td class="tg-c3ow">0.512 </td>
  </tr>
  <tr>
    <td class="tg-c3ow">Polyp-CASCADE</td>
    <td class="tg-c3ow"><a href="https://drive.google.com/file/d/1C46TSJuiwVQmBAW6cGaST-YEge1P22yq/view?usp=sharing" target="_blank" rel="noopener noreferrer">full_ckpt</a></td>
    <td class="tg-c3ow"><a href="https://drive.google.com/file/d/19-P7b4XcOcuIdypXTo8Bo9B3CuVbTIE9/view?usp=sharing" target="_blank" rel="noopener noreferrer">log</a></td>
    <td class="tg-c3ow">0.925 </td>
    <td class="tg-c3ow">0.936 </td>
    <td class="tg-c3ow">0.817 </td>
    <td class="tg-c3ow">0.905 </td>
    <td class="tg-c3ow">0.813 </td>
    <td class="tg-c3ow">0.769 </td>
    <td class="tg-c3ow">0.716 </td>
    <td class="tg-c3ow">0.561 </td>
  </tr>
  <tr>
    <td class="tg-c3ow">PIDNet (Lightweigh)</td>
    <td class="tg-c3ow"><a href="https://drive.google.com/file/d/1yGKgwkvHZXwnOig2XMNpzuC5QLRwgzjG/view?usp=sharing" target="_blank" rel="noopener noreferrer">full_ckpt</a></td>
    <td class="tg-c3ow"><a href="https://drive.google.com/file/d/1wIdHtJv7ThKcFnDKUQS6j3_CGX7xGy-b/view?usp=sharing" target="_blank" rel="noopener noreferrer">log</a></td>
    <td class="tg-c3ow">0.885 </td>
    <td class="tg-c3ow">0.878 </td>
    <td class="tg-c3ow">0.733 </td>
    <td class="tg-c3ow">0.881 </td>
    <td class="tg-c3ow">0.728 </td>
    <td class="tg-c3ow">0.719 </td>
    <td class="tg-c3ow">0.615 </td>
    <td class="tg-c3ow">0.389 </td>
  </tr>
</tbody></table>

## Application of EndoKED-SEG
With the released checkpoints, you can utilize them for the downstream tasks.
To get started, please refer to the official codebase or choose your preferred architecture from ```./EndoKED_SEG_Baselines```.

Before running your baseline, download the corresponding checkpoint pretrained on the EndoKED dataset and place it under the ```./logs``` directory.
You should also configure your dataset path in the script ```run_train.sh```.
Finally, you can transfer EndoKED to your downstream tasks using the following command:

```
bash run_train.sh
```

## Training of EndoKED

**1. EndoKED-MIL**

 ```
 pyhon ./EndoKED_MIL/train_Endo_BagDistillation_SharedEnc_Similarity_StuFilter.py
 ```

**2. EndoKED-WSSS**

- ###### 2.1 Data processing
  
  ```
  bash ./EndoKED_WSSS/launch/1_data_processing.sh
  ```

- ###### 2.2 Generating Class Activation Maps (CAMs)

  ```
  bash ./EndoKED_WSSS/launch/run_ALL.sh
  ```

- ###### 2.3 Refine CAMs to Pseudo Labels

  ```
  bash ./EndoKED_WSSS/launch/3_refine_CAM_2_Pseudo.sh
  ```

**3. EndoKED-SEG**

- ###### 3.1 Train EndoKED-SEG
  


  ```
  bash ./EndoKED_SEG/train.sh
  ```

- ###### 3.2 Refine Preds to Pseudo Labels

  ```
  bash ./EndoKED_WSSS/launch/5_refine_Preds_2_Pseudo.sh
  ```

- ###### Iterate Step 3.1-3.2 to optimize EndoKED-SEG

## Evaluation of EndoKED


```
python ./EndoKED_WSSS/eval_tools/a1_eval_pseuo_labels_from_SAM_byPreds_fromDecoder.py
```



## Acknowledgement

We borrowed [Polyp-PVT](https://github.com/DengPingFan/Polyp-PVT) as our segmentation model. 9 powerful baseline models, i.e., [Unet](https://github.com/milesial/Pytorch-UNet), [Unet++](https://github.com/4uiiurz1/pytorch-nested-unet), [C2FNet](https://github.com/thograce/C2FNet.git), [DCRNet](https://github.com/PRIS-CV/DCRNet.git), [LDNet](https://github.com/ReaFly/LDNet.git), [Polyp-PVT](https://github.com/DengPingFan/Polyp-PVT), [FCBFormer](https://github.com/ESandML/FCBFormer.git), [Polyp-CASCADE](https://github.com/SLDGroup/CASCADE.git), and [PIDNet (Lightweigh)](https://github.com/XuJiacong/PIDNet.git), are also adopted as our baselines. [Segment Anything](https://github.com/NVlabs/SegFormer) and their [pre-trained weights](https://drive.google.com/drive/folders/1b7bwrInTW4VLEm27YawHOAMSMikga2Ia?usp=sharing) are leveraged to refine the pseudo labels. [ToCo](https://github.com/rulixiang/ToCo) inspires us to conduct the generation of CAMs. Many thanks to their brilliant works!



## Citation
If you find this repository useful, please consider giving a star :star: and citation :t-rex::
```
@article{wang2023knowledge,
  title={Knowledge extraction and distillation from large-scale image-text colonoscopy records leveraging large language and vision models},
  author={Wang, Shuo and Zhu, Yan and Luo, Xiaoyuan and Yang, Zhiwei and Zhang, Yizhe and Fu, Peiyao and Wang, Manning and Song, Zhijian and Li, Quanlin and Zhou, Pinghong and others},
  journal={arXiv preprint arXiv:2310.11173},
  year={2023}
}

```

If you have any question, please feel free to contact.
