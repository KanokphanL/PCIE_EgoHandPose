# POTTER s12 - manual
CUDA_VISIBLE_DEVICES=1 python3 train.py \
    --gt_anno_dir data/gt_anno_manual_dir \
    --aria_img_dir data/aria_img_dir 


# ViT-base - Transformer head
CUDA_VISIBLE_DEVICES=0 python3 train_vit.py \
    --gt_anno_dir data/gt_anno_manual_dir \
    --aria_img_dir data/aria_img_dir \
    --yolox_hsv_random_aug \
    --albumentation_aug \
    --random_vertical_flip \
    --loss MPJPELoss \
    --cfg_file configs/vit_base_transformerhead_joint.yaml \
    --pretrained_ckpt ckpts/backbone/vitpose-b_mmpretrain-format.pth


# ViT-large - Transformer head
CUDA_VISIBLE_DEVICES=1 python3 train_vit.py \
    --gt_anno_dir data/gt_anno_manual_dir \
    --aria_img_dir data/aria_img_dir \
    --yolox_hsv_random_aug \
    --albumentation_aug \
    --random_vertical_flip \
    --loss MPJPELoss \
    --cfg_file configs/vit_large_transformerhead_joint_12depth_1024dim.yaml\
    --pretrained_ckpt ckpts/backbone/vitpose-l-simple_mmpretrain-format.pth

# ViT-huge - Transformer head
CUDA_VISIBLE_DEVICES=1 python3 train_vit.py \
    --gt_anno_dir data/gt_anno_manual_dir \
    --aria_img_dir data/aria_img_dir \
    --yolox_hsv_random_aug \
    --albumentation_aug \
    --random_vertical_flip \
    --loss MPJPELoss \
    --cfg_file configs/vit_huge_transfromerhead_joint_12depth_1280dim.yaml \
    --pretrained_ckpt ckpts/backbone/vitpose-h_mmpretrain-format.pth