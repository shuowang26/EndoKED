#! bash/bin

file=./EndoKD_WSSS/datasets/data_processing_refining/a3_generate_pseulabel_from_SAM_byCAM.py
python $file --csv_save_path 'set a path to save csv' --data_root 'path of training data' --cam_dir 'pseudo labels save path' --pred_dir 'save path for pseudo labels from preds'
