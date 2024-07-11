# Ego-Exo4D Hand Ego Pose ViTFormer Solution
Implementation of hand-ego-pose-potter, a 3D hand pose estimation model based on ViTFormer Solution.

## Data Preparation
Follow instructions [here](https://github.com/EGO4D/ego-exo4d-egopose/tree/main/handpose/data_preparation) to get:
- ground truth annotation files in `$gt_output_dir/annotation/manual` or `$gt_output_dir/annotation/auto` if using automatic annotations,
referred to as `gt_anno_dir` below
- corresponding undistorted Aria images in `$gt_output_dir/image/undistorted`, 
referred to as `aria_img_dir` below

## Setup

- Follow the instructions below to set up the environment for model training and inference.
```
conda create -n potter_hand_pose python=3.9.16 -y
conda activate potter_hand_pose
pip install -r requirement.txt
```
- Install [pytorch](https://pytorch.org/get-started/previous-versions/). The model is tested with `pytorch==2.1.0` and `torchvision==0.16.0`.
- Install `mmdet==3.1.0` and `mmpretrain==1.2.0`.


## Training

For the ViT backbone, download the backbone from [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) and convert the model format to MMPretrain format. 

Run the command below to perform training on manual data with pre-trained weight:
```
python3 train_vit.py \
    --gt_anno_dir <gt_anno_dir> \
    --aria_img_dir <aria_img_dir> \
    --yolox_hsv_random_aug \
    --albumentation_aug \
    --random_vertical_flip \
    --loss MPJPELoss \
    --cfg_file <config_file> \
    --pretrained_ckpt <pretrained_weight_file>
```

Check the script `train_vit.sh` for the command to train ViT models. 

If choose to finetune on manual data with pre-trained weight on automatic data, set `pretrained_ckpt` to be the path of pre-trained hand-ego-pose-potter model weight.


## Inference

Run the command below to perform inference of the pre-trained model on the test set, and save the inference output as a single JSON file. It will be stored at `output/inference_output` by default. 
```
python3 inference.py \
    --pretrained_ckpt <pretrained_ckpt> \
    --gt_anno_dir <gt_anno_dir> \
    --aria_img_dir <aria_img_dir>
```

The results can be submitted to ([EvalAI leaderboard](https://eval.ai/web/challenges/challenge-page/2249/overview)).

## Results and Models
Test results the model on validation set:

|      Model       |    resolution    |   MPJPE    |    PA-MPJPE  |  config    |   ckpt   |
| :--------------: | :--------------: | :--------: | :----------: | :--------: | :------: | 
|  ViTFormer-Base  |      256x192     |    24.67   |     9.31     | [config](/configs/vit_base_transformerhead_joint.yaml) | [ckpt](https://drive.google.com/file/d/1Wdye-g2KiC_0XcGy33tfgvWy8Y06ir-4/view?usp=sharing) | 
|  ViTFormer-Large |      256x192     |    23.38   |     9.05     | [config](/configs/vit_large_transformerhead_joint_12depth_1024dim.yaml) | [ckpt](https://drive.google.com/file/d/1wzHyKPU4Hym9GS48n42J5I-uoTin_QZv/view?usp=sharing) | 


## Citation
Please cite the following report if our code is helpful to your research.
```
@misc{chen2024pcieegohandposesolutionegoexo4dhand,
      title={PCIE_EgoHandPose Solution for EgoExo4D Hand Pose Challenge}, 
      author={Feng Chen and Ling Ding and Kanokphan Lertniphonphan and Jian Li and Kaer Huang and Zhepeng Wang},
      journal={arXiv},
      year={2024}
}
```

