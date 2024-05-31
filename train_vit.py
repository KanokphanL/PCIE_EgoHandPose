import json
import os
import random
import copy

import torch
import torchvision.transforms as transforms
from dataset.ego4d_dataset import ego4dDataset
from models.PoolAttnHR_Pose_3D import load_pretrained_weights, PoolAttnHR_Pose_3D
from models.ViT_Pose_3D import ViT_Pose_3D


from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils.functions import (
    AverageMeter,
    create_logger,
    parse_args_function,
    update_config,
)
from utils.loss import Pose3DLoss, MPJPELoss, RLELoss3D
from evaluation.evaluate import mpjpe, p_mpjpe
import numpy as np

import utils.layer_wise_lr_decay as lrd
import utils.lr_sched as lr_sched

from easydict import EasyDict as edict

def train(
    config,
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    device,
    logger,
    writer_dict,
):
    loss_3d = AverageMeter()

    # switch to train mode
    model.train()
    train_loader = tqdm(train_loader, dynamic_ncols=True)
    print_interval = len(train_loader) // config.TRAIN_PRINT_NUM

    for i, (input, pose_3d_gt, vis_flag, _) in enumerate(train_loader):

        # Adjust learning rate
        if config.MODEL.NAME == 'ViT_Pose_3D':
            # we use a per iteration (instead of per epoch) lr scheduler for VideoMAEv2
            lr_args = edict()
            lr_args.lr = config.TRAIN.LR
            lr_args.warmup_epochs = config.TRAIN.WARMUP_EPOCH # 5
            lr_args.min_lr = 5e-6
            lr_args.epochs = config.TRAIN.END_EPOCH
            lr = lr_sched.adjust_learning_rate(optimizer, i / len(train_loader) + epoch, lr_args)
            # print("learning rate =", lr)
        else:
            lr = optimizer.param_groups[0]['lr']

        # compute output
        input = input.to(device)
        pose_3d_pred = model(input)
        # Assign None kpts as zero
        pose_3d_gt[~vis_flag] = 0
        pose_3d_gt = pose_3d_gt.to(device)
        vis_flag = vis_flag.to(device)

        pose_3d_loss = criterion(pose_3d_pred, pose_3d_gt, vis_flag)
        loss = pose_3d_loss

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        loss_3d.update(pose_3d_loss.item())

        # Log info
        if (i + 1) % print_interval == 0:
            msg = (
                "Epoch: [{0}][{1}/{2}]\t"
                "Lr  {Lr:.6f}\t"
                "3D Loss {loss_3d.val:.5f} ({loss_3d.avg:.5f})".format(
                    epoch, i + 1, len(train_loader), loss_3d=loss_3d, 
                    Lr=lr
                )
            )
            logger.info(msg)

            if writer_dict:
                writer = writer_dict["writer"]
                global_steps = writer_dict["train_global_steps"]
                writer.add_scalar("LR/lr", lr, global_steps)
                writer.add_scalar("Loss/train", loss_3d.avg, global_steps)
                writer_dict["train_global_steps"] = global_steps + 1


def validate(
    val_loader, model, criterion, device, logger, writer_dict
):
    loss_3d = AverageMeter()

    # switch to evaluate mode
    model.eval()
    dataset =val_loader.dataset
    epoch_loss_3d_pos = []
    epoch_loss_3d_pos_procrustes = []
    with torch.no_grad():
        val_loader = tqdm(val_loader, dynamic_ncols=True)
        for i, (input, pose_3d_gt, vis_flag, _) in enumerate(val_loader):
            # compute output
            input = input.to(device)
            pose_3d_pred = model(input)
            pose_3d_gt[~vis_flag] = 0
            pose_3d_gt = pose_3d_gt.to(device)
            pose_3d_gt_2 = copy.deepcopy(pose_3d_gt)
            vis_flag = vis_flag.to(device)

            pose_3d_loss = criterion(pose_3d_pred, pose_3d_gt, vis_flag)

            pose_3d_pred_2 = copy.deepcopy(pose_3d_pred[:, :, :3])
            # measure accuracy and record loss
            loss_3d.update(pose_3d_loss.item())

            #caculate metric

            pred_3d_pts = pose_3d_pred_2.cpu().detach().numpy()
            pred_3d_pts = (pred_3d_pts * dataset.joint_std + dataset.joint_mean)
            
            gt_3d_pts =pose_3d_gt_2.cpu().detach().numpy()
            gt_3d_pts = (gt_3d_pts * dataset.joint_std + dataset.joint_mean)
            pose_3d_gt_2.cpu().detach().numpy()
            
            # add offset
            pred_3d_pts +=gt_3d_pts[:,0:1,:]
            
            #Compute MPJPE and PA-MAJPE
            batch_size = pred_3d_pts.shape[0]
            for j in range(batch_size):
                valid_mask = vis_flag[j].cpu().detach().numpy()  # Assuming vis_flag is a tensor
                valid_pred_3d_kpts = pred_3d_pts[j][valid_mask].reshape(1, -1, 3)
                valid_pose_3d_gt = gt_3d_pts[j][valid_mask].reshape(1, -1, 3)

                # Compute MPJPE
                mpjpe_val = mpjpe(valid_pred_3d_kpts, valid_pose_3d_gt).item()
                epoch_loss_3d_pos.append(mpjpe_val)

                # Compute PA-MPJPE
                pamjpe_val = p_mpjpe(valid_pred_3d_kpts, valid_pose_3d_gt)
                epoch_loss_3d_pos_procrustes.append(pamjpe_val)
        epoch_loss_3d_pos_avg = np.mean(epoch_loss_3d_pos)
        epoch_loss_3d_pos_procrustes_avg = np.mean(epoch_loss_3d_pos_procrustes)

        # Log info
        msg = (
            "Val: [{0}/{1}]\t"
            "3D Loss {loss_3d.avg:.5f}\t"
            "MPJPE {mpjpe:.5f} (mm)\t"
            "P-MPJPE {pmpjpe:.5f} (mm)".format(
            i + 1, len(val_loader), 
            loss_3d=loss_3d, 
            mpjpe=epoch_loss_3d_pos_avg, 
            pmpjpe=epoch_loss_3d_pos_procrustes_avg
            )
        )
        logger.info(msg)

        if writer_dict:
            writer = writer_dict["writer"]
            global_steps = writer_dict["valid_global_steps"]
            writer.add_scalar("Loss/val", loss_3d.avg, global_steps)
            writer.add_scalar("Metrics/MPJPE", epoch_loss_3d_pos_avg, global_steps)
            writer.add_scalar("Metrics/P-MPJPE", epoch_loss_3d_pos_procrustes_avg, global_steps)
            writer_dict["valid_global_steps"] = global_steps + 1

    return loss_3d.avg, epoch_loss_3d_pos_avg, epoch_loss_3d_pos_procrustes_avg


def main(args):
    torch.cuda.empty_cache()
    cfg = update_config(args.cfg_file)
    pretrained_hand_pose_CKPT = args.pretrained_ckpt

    device = torch.device(
        f"cuda:{args.gpu_number[0]}" if torch.cuda.is_available() else "cpu"
    )
    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg_file, "train")

    logger.info(f"****************cfg:*************** \n {cfg}")
    logger.info(f"*************** args:************** \n {args}")

    ############ MODEL ###########
    
    # model = PoolAttnHR_Pose_3D(**cfg.MODEL)
    logger.info(f"Model: {cfg.MODEL.NAME}")
    model = eval(cfg.MODEL.NAME)(**cfg.MODEL)

    # Load pretrained cls_weight or available hand pose weight
    if cfg.MODEL.NAME in ["PoolAttnHR_Pose_3D"]:
        cls_weight = torch.load(args.cls_ckpt)
        if pretrained_hand_pose_CKPT:
            load_pretrained_weights(
                model, torch.load(pretrained_hand_pose_CKPT, map_location=device)
            )
            logger.info(f"Loaded pretrained weight from {pretrained_hand_pose_CKPT}")
        else:
            load_pretrained_weights(model.poolattnformer_pose.poolattn_cls, cls_weight)
            logger.info(f"Loaded pretrained POTTER-cls weight from {args.cls_ckpt}")
    else:
        model.load_pretrained_weights(
            torch.load(pretrained_hand_pose_CKPT, map_location=device)
        )
        logger.info(f"Loaded pretrained weight from {pretrained_hand_pose_CKPT}")

    model_without_ddp = model

    model = model.to(device)
    writer_dict = {
        "writer": SummaryWriter(log_dir=tb_log_dir),
        "train_global_steps": 0,
        "valid_global_steps": 0,
    }

    ########### DATASET ###########
    # Load Ego4D dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_dataset = ego4dDataset(args, cfg, split="train", transform=transform)
    valid_dataset = ego4dDataset(args, cfg, split="val", transform=transform)
    # Dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True,
    )
    logger.info(f"Loaded ground truth annotation from {args.gt_anno_dir}")
    logger.info(
        f"Number of annotation(s): Train: {len(train_dataset)}\t Val: {len(valid_dataset)}"
    )
    logger.info(
        f"Learning rate: {cfg.TRAIN.LR} || Batch size: Train:{cfg.TRAIN.BATCH_SIZE}\t Val: {cfg.TEST.BATCH_SIZE}"
    )

    ############# CRITERION AND OPTIMIZER ###########
    # define loss function (criterion) and optimizer
    if args.loss == "Pose3DLoss":
        criterion = Pose3DLoss().cuda()
    elif args.loss == "MPJPELoss":
        criterion = MPJPELoss().cuda()
    elif args.loss == "RLELoss3D":
        criterion = RLELoss3D().cuda()

    if cfg.MODEL.NAME == 'ViT_Pose_3D':
        no_weight_decay_list = ['backbone.pos_embed', 'backbone.cls_token']
        param_groups = lrd.param_groups_lrd(
            model_without_ddp, cfg.TRAIN.WEIGHT_DECAY,
            no_weight_decay_list=no_weight_decay_list,
            layer_decay=cfg.TRAIN.LAYER_DECAY)
        optimizer = torch.optim.AdamW(
            param_groups, lr=cfg.TRAIN.LR, weight_decay=0.05)
    else:
        # optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.TRAIN.LR, weight_decay=0.05)
    
    # define learning rate scheduler
    num_steps = len(train_loader) * cfg.TRAIN.END_EPOCH
    # CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.TRAIN.END_EPOCH, eta_min=(5e-6))

    ############ Train model & validation ###########
    best_val_loss = 1e2
    best_val_p_mpjpe =1e2
    for epoch in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):
        # train for one epoch
        logger.info(f"############# Starting Epoch {epoch} #############")
        train(
            cfg,
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            device,
            logger,
            writer_dict,
        )

        # evaluate on validation set
        val_loss, val_mpjpe, val_p_mpjpe = validate(  
            valid_loader,
            model,
            criterion,
            device,
            logger,
            writer_dict,
        )

        # Save best model weight by val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save model weight
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_p_mpjpe": val_p_mpjpe,
                },
                os.path.join(
                    final_output_dir, f"{cfg.MODEL.NAME}-best-loss-{cfg.DATASET.DATASET}.pt"
                ),
            )
        
        # Save best model by val p_mpjpe
        if val_p_mpjpe < best_val_p_mpjpe:
            best_val_p_mpjpe = val_p_mpjpe
            # Save model weight
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_p_mpjpe": val_p_mpjpe,
                },
                os.path.join(
                    final_output_dir, f"{cfg.MODEL.NAME}-best-metric.pt"
                ),
            )
            


if __name__ == "__main__":
    args = parse_args_function()
    main(args)
