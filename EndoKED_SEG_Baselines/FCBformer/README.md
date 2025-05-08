

## Train the model

`bash run_train.sh [gpus] [nproc_per_node] [master_port]

bash run_train.sh 0,1,2 3 29573
`

### 注意：
 - 脚本中的dataset 有多个选项 [Train_on_ZhongshanandKvasirandDB] 为我们的数据加上公开的训练集，来得到我们的数据对此方法的性能提升。[Train_on_KvasirandDB] 为复现其方法，作为baseline.

 - root 为公开数据集的地址 ***/home/gpu_user/data/EndoGPT/database/息肉公开数据集/dataset/TrainDataset/*** val_root为测试集的地址: ***/home/gpu_user/data/EndoGPT/database/息肉公开数据集/dataset/TestDataset/test/***


 ## Eval the model

 `bash run_eval.sh [gpu] [dataset] [cpt_path]

bash run_eval.sh 0 Train_on_ZhongshanandKvasirandDB /home/gpu_user/data/yzw/endokd_rebuttal/FCBFormer/logs/Train_on_ZhongshanandKvasirandDB/2024-03-16-14-04-04/Train_on_ZhongshanandKvasirandDB.pt

`

