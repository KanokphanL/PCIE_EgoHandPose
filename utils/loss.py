import numpy as np
import torch
import torch.nn as nn

import math
        
    

class RLELoss3D(nn.Module):
    ''' RLE Regression Loss 3D
    '''

    def __init__(self, OUTPUT_3D=False, size_average=True):
        super(RLELoss3D, self).__init__()
        self.size_average = size_average
        self.amp = 1 / math.sqrt(2 * math.pi)

    def logQ(self, gt_uv, pred_jts, sigma):
        return torch.log(sigma / self.amp) + torch.abs(gt_uv - pred_jts) / (math.sqrt(2) * sigma + 1e-9)

    def nfLoss(self, pose_3d_pred, pose_3d_gt, vis_flag, sigma):
          
            
            

        if pose_3d_gt is not None:
            gt_uvd = pose_3d_gt
            
            assert pose_3d_pred.shape == sigma.shape, (pose_3d_pred.shape, sigma.shape)
            bar_mu = (pose_3d_pred - gt_uvd) / sigma
            bar_mu = bar_mu.reshape(-1, 3)
    
            # (B, K, 3)
            log_phi = torch.zeros_like(bar_mu[:, 0])
            log_phi = log_phi.reshape(pose_3d_pred.shape[0], pose_3d_pred.shape[1], 1)

            nf_loss = torch.log(sigma) - log_phi
        else:
            nf_loss = None
        return nf_loss
        
        
        
    def forward(self, pose_3d_pred, pose_3d_gt, vis_flag):
        pred_sigma = pose_3d_pred[:, :, 3:6]
        sigma = pred_sigma.sigmoid() + 1e-9
        pose_3d_pred = pose_3d_pred[:, :, :3]

        nf_loss = self.nfLoss(pose_3d_pred, pose_3d_gt, vis_flag, sigma)
        pred_jts = pose_3d_pred
        flag = torch.unsqueeze(vis_flag, 2)
        flag = torch.cat((flag, flag, flag), 2)
        gt_uv = pose_3d_gt.reshape(pose_3d_pred.shape)
        nf_loss = nf_loss * flag
        residual = True
        if residual:
            Q_logprob = self.logQ(gt_uv, pose_3d_pred, sigma) * flag
            loss = nf_loss + Q_logprob

        if self.size_average and flag.sum() > 0:
            return loss.sum() / len(loss)
        else:
            return loss.sum()
        
class Pose3DLoss(nn.Module):
    """
    MSE Loss
    """
    def __init__(self):
        super(Pose3DLoss, self).__init__()

    def forward(self, pose_3d_pred, pose_3d_gt, vis_flag):
        # Compute MSE loss between pred and gt 3D hand joints for only visible kpts
        assert (
            pose_3d_pred.shape == pose_3d_gt.shape and len(pose_3d_pred.shape) == 3
        )  # (N, K, dim)
        pose_3d_diff = pose_3d_pred - pose_3d_gt
        pose_3d_loss = torch.mean(pose_3d_diff**2, axis=2) * vis_flag
        pose_3d_loss = torch.sum(pose_3d_loss) / torch.sum(vis_flag)

        return pose_3d_loss

class MPJPELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pose_3d_pred, pose_3d_gt, vis_flag):
        # Compute MSE loss between pred and gt 3D hand joints for only visible kpts
        assert (
            pose_3d_pred.shape == pose_3d_gt.shape and len(pose_3d_pred.shape) == 3
        )  # (N, K, dim)
        pose_3d_diff = pose_3d_pred - pose_3d_gt
        # pose_3d_loss = torch.mean(pose_3d_diff**2, axis=2) * vis_flag
        # pose_3d_loss = torch.sum(pose_3d_loss) / torch.sum(vis_flag)
        
        pose_3d_loss = torch.norm(pose_3d_diff, dim=-1) * vis_flag
        pose_3d_loss = torch.mean(pose_3d_loss)

        return pose_3d_loss


def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance) from
    https://github.com/zhaoweixi/GraFormer/blob/main/common/loss.py.
    Modified s.t. it could compute MPJPE for only those valid keypoints (where
    # of visible keypoints = num)
    """
    assert predicted.shape == target.shape
    pjpe = torch.norm(predicted - target, dim=len(target.shape) - 1)
    mpjpe = torch.mean(pjpe)
    return mpjpe


def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape

    # Convert to Numpy because this metric needs numpy array
    predicted = predicted.cpu().detach().numpy()
    target = target.cpu().detach().numpy()

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t

    # Return MPJPE
    pjpe = np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1)
    p_mpjpe = np.mean(pjpe)
    return p_mpjpe
