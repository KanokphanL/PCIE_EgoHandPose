# Ego-Exo4D Hand Ego Pose ViTFormer Solution
Implementation of hand-ego-pose-potter, a 3D hand pose estimation model based on ViTFormer Solution.

## Data preparation
Follow instructions [here](https://github.com/EGO4D/ego-exo4d-egopose/tree/main/handpose/data_preparation) to get:
- ground truth annotation files in `$gt_output_dir/annotation/manual` or `$gt_output_dir/annotation/auto` if using automatic annotations,
referred as `gt_anno_dir` below
- corresponding undistorted Aria images in `$gt_output_dir/image/undistorted`, 
referred as `aria_img_dir` below

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

For ViT backbone, download the backbone from [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) and convert the model format to MMPretrain format. 

Run command below to perform training on manual data with pretrained weight:
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

If choose to finetuning on manual data with pretrained weight on automatic data, set `pretrained_ckpt` to be the path of pretrained hand-ego-pose-potter model weight.


## Inference

Download pretrained ([EvalAI baseline](https://eval.ai/web/challenges/challenge-page/2249/overview)) model weight of hand-ego-pose-potter from [here](https://drive.google.com/drive/folders/1WSvV7wvmYBvFhB5KwK6PRXwV5dpHd9Hf?usp=sharing).

Run command below to perform inference of pretrained model on test set, and save the inference output as a single JSON file. It will be stored at `output/inference_output` by default. 
```
python3 inference.py \
    --pretrained_ckpt <pretrained_ckpt> \
    --gt_anno_dir <gt_anno_dir> \
    --aria_img_dir <aria_img_dir>
```
