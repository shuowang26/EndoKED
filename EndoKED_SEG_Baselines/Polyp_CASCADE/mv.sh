#! bash/bin

cd /home/gpu_user/data/EndoGPT/database/息肉公开数据集/dataset/TrainDataset/polygen_C1-5_training

rm -rf images
rm -rf masks

mkdir images
mkdir masks

cp ../../TestDataset/Polygen_all_frames/data_C5/images/* ./images/
cp ../../TestDataset/Polygen_all_frames/data_C4/images/* ./images/
cp ../../TestDataset/Polygen_all_frames/data_C3/images/* ./images/
cp ../../TestDataset/Polygen_all_frames/data_C2/images/* ./images/
cp ../../TestDataset/Polygen_all_frames/data_C1/images/* ./images/


cp ../../TestDataset/Polygen_all_frames/data_C5/masks_renamed/* ./masks/
cp ../../TestDataset/Polygen_all_frames/data_C4/masks_renamed/* ./masks/
cp ../../TestDataset/Polygen_all_frames/data_C3/masks_renamed/* ./masks/
cp ../../TestDataset/Polygen_all_frames/data_C2/masks_renamed/* ./masks/
cp ../../TestDataset/Polygen_all_frames/data_C1/masks_renamed/* ./masks/
