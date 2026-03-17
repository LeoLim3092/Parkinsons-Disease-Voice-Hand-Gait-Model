# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import numpy as np

bone15 = [0, 1, 2, 3, 1, 5, 6, 0, 8, 9, 0, 11, 12, 1]
bone17 = [0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
bone_pair = [3, 4, 5, 0, 1, 2, 6, 7, 8, 9, 13, 14, 15, 10, 11, 12]

is_initTrans = False
bone_trans_L = torch.tensor([])
bone_trans_R = torch.tensor([])
bone_symmetry = torch.tensor([])

def initTrans(predicted):
    global bone_trans_L, bone_trans_R, bone_symmetry
    sz = predicted.shape[2]
    bone_trans_L = torch.zeros(sz-1, sz)
    bone_trans_R = torch.zeros(sz-1, sz)
    bone_symmetry = torch.zeros(sz-1, sz-1)
    
    for i in range(sz-1):
        bone_trans_L[i,i+1] = 1
        bone_trans_R[i, bone17[i]] = 1
        
        bone_symmetry[i, i] += 1
        bone_symmetry[bone_pair[i], i] -= 1
    
    if torch.cuda.is_available():
        bone_trans_L = bone_trans_L.cuda()
        bone_trans_R = bone_trans_R.cuda()
        bone_symmetry = bone_symmetry.cuda()

def boneLoss(predicted, target):
    # predicted.shape: (Batch size, feature size, joints, 3)
    global bone_trans_L, bone_trans_R, bone_symmetry, is_initTrans
    if not is_initTrans:
        initTrans(predicted)
        is_initTrans = True
    
    bone_L = torch.matmul(bone_trans_L, predicted)
    bone_R = torch.matmul(bone_trans_R, predicted)
    bone_L_gt = torch.matmul(bone_trans_L, target)
    bone_R_gt = torch.matmul(bone_trans_R, target)
    
    B = bone_L - bone_R
    B_gt = bone_L_gt - bone_R_gt
    
    # length consistency
    bone_length = torch.squeeze(torch.norm(B, dim=-1))
    bone_length_gt = torch.squeeze(torch.norm(B_gt, dim=-1))
    length_loss = torch.mean(torch.sum(abs(bone_length - bone_length_gt), dim=1)) 
    
    # direction
    direction_loss = torch.mean(torch.sum(torch.norm(B-B_gt, dim=-1), dim=-1))

    # symmetry
    symmetry_loss = 0 #torch.mean(torch.sum(torch.abs(torch.matmul(bone_length, bone_symmetry)), 1))
    return length_loss, direction_loss, symmetry_loss

def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))

def mpjpe_reprojection_multi_hypo(predicted, target, num_hypotheses):
    hypotheses = torch.zeros(num_hypotheses)
    if torch.cuda.is_available():
        hypotheses = hypotheses.cuda()
    
    for i in range(num_hypotheses):
        hypotheses[i] = torch.mean(torch.norm(predicted[:,i:i+1] - target, dim=len(target.shape)-1))
    return torch.mean(hypotheses)

def mpjpe_multi_hypo(predicted, target):
    num_hypotheses = predicted.shape[1]
    hypotheses = torch.zeros(num_hypotheses)
    if torch.cuda.is_available():
        hypotheses = hypotheses.cuda()
    
    for i in range(num_hypotheses):
        # print(predicted.shape, target.shape)
        # predicted.shape = (batch, hypo, joints, 2)
        hypotheses[i] = torch.mean(torch.norm(predicted[:,i:i+1] - target, dim=len(target.shape)-1))
    return torch.min(hypotheses)

class TrainableLoss(nn.Module):
    def __init__(self):
        super(TrainableLoss, self).__init__()
        self.reproject = nn.Parameter(torch.tensor(0.3, requires_grad=True))
        self.bone = nn.Parameter(torch.tensor(0.6, requires_grad=True))
        self.direction = nn.Parameter(torch.tensor(0.5, requires_grad=True))
    
    def forward(self, loss_reprojection, length_loss, direction_loss):
        return loss_reprojection*torch.exp(-self.reproject) + \
                length_loss*torch.exp(-self.bone) + \
                direction_loss*torch.exp(-self.direction) + \
                self.reproject*0.5 + \
                self.bone*0.5 + \
                self.direction*0.5
                

def weighted_mpjpe(predicted, target, w):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    assert w.shape[0] == predicted.shape[0]
    return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape)-1))

def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape
    
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
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    
    # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t
    
    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape)-1))
    
def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape
    
    norm_predicted = torch.mean(torch.sum(predicted**2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target*predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return mpjpe(scale * predicted, target)

def weighted_bonelen_loss(predict_3d_length, gt_3d_length):
    loss_length = 0.001 * torch.pow(predict_3d_length - gt_3d_length, 2).mean()
    return loss_length

def weighted_boneratio_loss(predict_3d_length, gt_3d_length):
    loss_length = 0.1 * torch.pow((predict_3d_length - gt_3d_length)/gt_3d_length, 2).mean()
    return loss_length

def mean_velocity_error(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    
    velocity_predicted = np.diff(predicted, axis=0)
    velocity_target = np.diff(target, axis=0)
    
    return np.mean(np.linalg.norm(velocity_predicted - velocity_target, axis=len(target.shape)-1))