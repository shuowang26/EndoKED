#! bash/bin


file=/data/PROJECTS/Endo_GPT/EndoGPT_WSSS/scripts/train_segmentation_Large_zhongshan.py
nproc_per_node=1
master_port=29529

CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --nproc_per_node=$nproc_per_node --master_port=$master_port $file
