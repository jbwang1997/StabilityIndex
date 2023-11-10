# Stable Index

## Introduction

In autonomous driving, the temporal stability of 3D object detection will significantly impact downstream tasks (*e.g.*, tracking and planning), and is therefore crucial for safety driving.
The detection stability however cannot be accessed by existing metrics, like mAP and MOTA, and consequently is less explored by the community.
To bridge this gap, this work proposes the *Stable Index* (*SI*), a new metric that can comprehensively evaluate the stability of 3D detectors in terms of confidence, box localization, extent, and heading.
We further introduce a general and effective training strategy, called *Prediction Consistency Learning* (*PCL*).
PCL essentially encourages the prediction consistency of the same objects under different timestamps and augmentations, leading to enhanced detection stability. 
By benchmarking current object detectors on the Waymo Open Dataset, SI reveals several interesting properties of object stability that have been previously overlooked by other metrics.
Furthermore, we examine the effectiveness of PCL with the popular CenterPoint, and achieve a remarkable 85.13\% SI for class vehicle, surpassing the baseline by 5.36\%.
We hope our work could serve as a reliable baseline and draw the committee's attention to the crucial issue in 3D object detection.
Code will be made publicly available.

## Demos

## Results

## Installation

Our *Stable Index* is implemented on the open source codebase [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).
Please follow the guidence of the [INSTALL.md](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md) and [GETTING_STARTED.md](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md#waymo-open-dataset) in OpenPCDet for installation of this repo and data preparation of Waymo Open Dataset.

**Notification**: Here, we provide some important dependencies in [environment.txt](environments.txt) for the compatiblity of Tensorflow-2.5.
It's highly recommend using `pip install -r evironment.txt` before installation.

## Training and Evaluation

All configures using in our expeirments can be found in [waymo_stable](tools/cfgs/waymo_stable/).
Here, we introduce the steps to reproduce the results in our paper.

### Traning Baseline CenterPoint

The first step to re-implement PCL is training a baseline checkpoint of CenterPoint.
We provide the configure in [centerpoint_baselin.yaml](tools/cfgs/waymo_stable/centerpoint_baseline.yaml). You can train the baseline CenterPoint by:

```
cd tools
bash scripts/dist_train.sh 8 --cfg_file cfgs/waymo_stable/centerpoint_baseline.yaml
```

### Fine-tuning without or with PCL

We provide the confuigures of fine-tuning CenterPoint without PCL ([centerpoint_finetune.yaml](tools/cfgs/waymo_stable/centerpoint_finetune.yaml)) and with PCL ([centerpoint_PCL_n{X}.yaml](tools/cfgs/waymo_stable/centerpoint_PCL_n16.yaml)).
Here, `X` is the maximum interval between the neighborhood sampling, described in our paper.

Before traning, you need to first change the `PRETRAINED` arguemnt in configures to the path of the baseline checkpoint:
```
MODEL:
    NAME: CenterPoint
    # change this path to the baseline checkpoint.
    PRETRAINED: '../output/waymo_stable/centerpoint_baseline/default/ckpt/checkpoint_epoch_36.pth'
```

Then, you can finue-tune the baseline without or with PCL.

- Fine-tune without PCL:
```
cd tools
bash scripts/dist_train.sh 8 --cfg_file cfgs/waymo_stable/centerpoint_finetune.yaml
```

- Fine-tune with PCL
```
cd tools
bash scripts/dist_train.sh 8 --cfg_file cfgs/waymo_stable/centerpoint_PCL_n16.yaml
```


### Evaluation Stable Index

The stable index has been integrated in [waymo_dataset.py](pcdet/datasets/waymo/waymo_dataset.py).
You can evluate SI by adding 'stabe_index' to `POST_PROCESSING.EVAL_METRIC` in the configuration.

Taking [centerpoint_PCL_n16.yaml](tools/cfgs/waymo_stable/centerpoint_PCL_n16.yaml) for example, its configuration is written as:
```
POST_PROCESSING:
    RECALL_THRESH_LIST: [0.3, 0.5, 0.7]
    EVAL_METRIC: ['waymo', 'stable_index']
```

Then, you can evalute the mAPH and SI by runing:
```
cd tools
python test.py \
    --cfg_file cfgs/waymo_stable/centerpoint_PCL_n16.yaml \
    --ckpt ../output/waymo_stable/centerpoint_PCL_n16/default/ckpt/checkpoint_epoch_5.pth
```

If you has tested the model and get the `result.pkl` in `output/waymo_stable/centerpoint_PCL_n16/default/eval/eval_with_train/epoch_5/`, you can directly evaluate stable index by:
```
python pcdet/datasets/waymo/waymo_stable_index.py \
    --gt_infos data/waymo/waymo_processed_data_v0_5_0_infos_val.pkl \
    --pred_infos output/waymo_stable/centerpoint_PCL_n16/default/eval/eval_with_train/epoch_5/result.py
```

## Acknowledgement

Our code is built on top of the open source codebase [OpenPCDet](https://github.com/open-mmlab/OpenPCDet). Thanks very much for their amazing works.
