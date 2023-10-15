#! bash/bin

file=EndoKD_WSSS/datasets/data_processing_refining/a5_generate_pseuo_labels_from_SAM_byPreds_fromDecoder.py
python $file --csv_save_path 'set a path to save csv' --data_root 'path of training data' --cam_dir 'pseudo labels save path' --
