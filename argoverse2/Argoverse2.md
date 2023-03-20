# Argoverse2 Data Processing

We follow [FSD](https://github.com/tusen-ai/SST) for data processing.

## Step 1: 
Download the **Argoverse 2 Sensor Dataset** from the [official website](https://www.argoverse.org/av2.html#download-link), and then extract them.

## Step 2: 
Denerate info files for `train` and `val`. 
```shell
python3 generate_trainval_infos.py
```
The generated file is `argo2_infos_train.pkl` and `argo2_infos_val.pkl`. You can also download them here ([train]() and [val]()), if you do not want to generate by yourself.

## Step 3
Generate info file for ground truth.
```shell
python3 gather_argo2_anno_feather.py
```
The generated file is `val_anno.feather`. You can also download them [here](), if you do not want to generate by yourself.

## Step 4 (optional) 
Generate info file for the gt-sampling augmentation.
```shell
python3 create_argo_gt_database.py
```
The generated file is `argo2_dbinfos_train.pkl`. You can also download them [here](), if you do not want to generate by yourself.

Note that this is an optional step. Because we disable gt-sampling augmentation in our Argoverse2 training. But it should be helpful if used with proper hyper-parameters. Please try this if you want.

## Step5
Install the official API of Argoverse 2
```shell
pip install av2==0.2.0
```
- Note that this [issue](https://github.com/argoverse/av2-api/issues/102) from the argo2 api might be noticed. 
- If the memory of your machine is limited, you can set `--workers=0` in the training script.
- The organized files are as follows:
```
VoxelNeXt
├── data
│   ├── argo2
│   │   │── ImageSets
│   │   │   ├──train.txt & val.txt
│   │   │── training
│   │   │   ├──velodyne
│   │   │── sensor
│   │   │   ├──val
│   │   │── argo2_infos_train.pkl
│   │   │── argo2_infos_val.pkl
│   │   │── (optional: argo2_dbinfos_train.pkl)
│   │   │── val_anno.feather
├── pcdet
├── tools
```