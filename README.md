<img width="1206" alt="image" src="https://github.com/shuowang26/EndoKED/assets/51802231/87ab3ddb-5fea-421a-93e6-4666974a2112">


# EndoKED
The official codes for **Knowledge Extraction and Distillation from Large-Scale Image-Text Colonoscopy Reports Leveraging Large Language and Vision Models**.

## Dependencies

To clone all files:

```
git clone 
```

To install Python dependencies:

```
pip install -r requirements.txt
```

## Datasets

#### **Training Dataset**   

1. Updating soon.

#### **Evaluation Dataset**   

EndoKED is evaluated on five public out-of-domain datasets, i.e., [CVC-ClinicDB](https://www.sciencedirect.com/science/article/abs/pii/S0895611115000567), [Kvasir-SEG](https://link.springer.com/chapter/10.1007/978-3-030-37734-2_37), [ETIS](https://link.springer.com/article/10.1007/s11548-013-0926-3), [CVC-ColonDB](https://ieeexplore.ieee.org/abstract/document/7294676/), and [CVC-300](https://www.hindawi.com/journals/jhe/2017/4037190/). Following the common experimental setups, the training set from CVC-ClinicDB and Kvasir-SEG are not used during the training and we evaluate our model only in the testing set for a fair comparison. The detailed description for the datasets are reported in Table below. 

The five public datasets are publicly available at https://pan.baidu.com/s/1A4e7kmvAShaz3BCitpunFA?pwd=s5t5.

| Dataset       | Year |     Resolution    | Training | Testing | Total |
|---------------|:----:|:-----------------:|:--------:|:-------:|:-----:|
| CVC-ClinincDB | 2015 |      384x384      |    550   |    62   |  612  |
| Kvasir-SEG    | 2020 | 332x487~1920x1072 |    900   |   100   |  1000 |
| ETIS          | 2014 |      1225x966     |    N/A   |   196   |  196  |
| CVC-ColonDB   | 2016 |      574x500      |    N/A   |   380   |  380  |
| CVC-300       | 2017 |      574x500      |    N/A   |    60   |   60  |


## Semantic Results

The results on five public datasets for EndoKED-SEG are reported in the following Table.

|      Models     |   Kvasir  |  ClinicDB |  ColonDB  |  CVC-300  |    ETIS   |
|:---------------:|:---------:|:---------:|:---------:|:---------:|:---------:|
|      U-Net      |   0.818   |   0.823   |   0.504   |   0.710   |   0.398   |
|      U-Net      |   0.821   |   0.794   |   0.482   |   0.707   |   0.401   |
|      C2FNet     |   0.886   |   0.919   |   0.724   |   0.874   |   0.699   |
|      DCRNet     |   0.886   |   0.896   |   0.704   |   0.856   |   0.556   |
|      LDNet      |   0.887   |   0.881   |   0.740   |   0.869   |   0.645   |
|    Polyp-PVT    |   0.917   |   0.948   |   0.808   |   0.900   |   0.787   |
| **EndoKED-SEG** | **0.908** | **0.920** | **0.809** | **0.893** | **0.818** |


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

**1. EndoKED-MIL**

Updating soon.

**2. EndoKED-SEG**

```
python ./EndoKED_WSSS/eval_tools/a1_eval_pseuo_labels_from_SAM_byPreds_fromDecoder.py
```


## Model logs and checkpoints

We provide the models' logs and checkpoints for EndoKED-SEG, which can be download from https://pan.baidu.com/s/1HaxIZf281lWFpk2USXs6OQ (a9d4) or from google drive with link: [https://drive.google.com/drive/folders/1QPGI7T9fa2ogC6_ZB9TChJg2DHIwCvub?usp=drive_link](https://drive.google.com/drive/folders/1QPGI7T9fa2ogC6_ZB9TChJg2DHIwCvub?usp=drive_link).


## Acknowledgement

We borrowed [Polyp-PVT](https://github.com/DengPingFan/Polyp-PVT) as our segmentation model.[Segment Anything](https://github.com/NVlabs/SegFormer) and their [pre-trained weights](https://drive.google.com/drive/folders/1b7bwrInTW4VLEm27YawHOAMSMikga2Ia?usp=sharing) are leveraged to refine the pseudo labels. [ToCo](https://github.com/rulixiang/ToCo) inspires us to conduct the generation of CAMs. Many thanks to their brilliant works!



## Citation
Updating soon.

If you have any question, please feel free to contact.

