import numpy as np
from common.camera import *
import math

# joint_list = [12, 13, 15, 16, 5, 6, 2, 3]
# parent_list = [11, 12, 14, 15, 4, 5, 1, 2]
# affect_list = [13, -1, 16, -1, 6, -1, 3, -1]
idx_list = [1, 2, 4, 8]
joint_list = [12, 13, 15, 16]
parent_list = [11, 12, 14, 15]
affect_list = [13, -1, 16, -1]

def line_sphere_intersection(o, r, p):
    e = p / np.linalg.norm(p)
    b = -2 * (o@e)
    c = o@o - r*r
    t = (-b + math.sqrt(b**2 - 4*a*c)) / a * 0.5
    return t * e

def map_projection(pose_3d, joint, parent, affect):
    ret = pose_3d.copy()
    v = np.array([0,0,1])
    a = ret[joint] - ret[parent]
    u = (v@a) * v
    ret[joint] = ret[joint] - 2 * u
    
    if affect != -1:
        ret[affect] = pose_3d[affect] - 2 * u
    return ret

v_torch = torch.tensor([0,0,1], dtype=torch.float32).cuda()
def map_projection_torch(pose_3d, joint, parent, affect):
    global v_torch
    ret = pose_3d.clone()
    a = ret[joint] - ret[parent]
    u = torch.inner(a, v_torch) * v_torch
    ret[joint] = ret[joint] - 2 * u
    
    if affect != -1:
        ret[affect] = pose_3d[affect] - 2 * u
    return ret

def map_with_camera_matrix(pose_3d, joint, parent, affect):
    ret = pose_3d.copy()
    v = ret[joint] / np.linalg.norm(ret[joint])
    a = ret[joint] - ret[parent]
    u = (v@a) * v
    ret[joint] = ret[joint] - 2 * u
    
    if affect != -1:
        ret[affect] = line_sphere_intersection(
            ret[joint],
            np.linalg.norm(pose_3d[affect] - pose_3d[joint]),
            pose_3d[affect]
        )
    return ret

def hypothesize(pose_3d, is_camera=False, remove_first=True):
    res = []
    tmp = pose_3d.cpu().numpy()
    # tmp = pose_3d.copy()
    tmp[1:] += tmp[0]
    for j, p, a in zip(joint_list, parent_list, affect_list):
        if len(res) == 0:
            res.append(tmp)
            if is_camera:
                res.append(map_with_camera_matrix(tmp, j, p, a))
            else:
                res.append(map_projection(tmp, j, p, a))
            continue
        
        for i in range(len(res)):
            if is_camera:
                res.append(map_with_camera_matrix(res[i], j, p, a))
            else:
                res.append(map_projection(res[i], j, p, a))
    
    res = np.array(res)
    res[:, 1:] -= res[:, :1]
    if remove_first:
        return res[1:]
    else:
        return res

def hypothesize_torch(pose_3d):
    res = torch.zeros(pose_3d.shape[0], 16, pose_3d.shape[2], pose_3d.shape[3], dtype=torch.float32).cuda()
    tmp = pose_3d.clone()
    
    for i_pose in range(pose_3d.shape[0]):
        res[i_pose, 0] = tmp[i_pose,0]
        i_hypo = 1
        for idx, j, p, a in zip(idx_list, joint_list, parent_list, affect_list):
            for i in range(idx):
                res[i_pose, i_hypo] = map_projection_torch(res[i_pose, i], j, p, a)
                i_hypo += 1
    
    return res

def generate_hypothesis(pose_3d, is_camera=False, joint=[], parent=[], affect=[]):
    tmp = pose_3d.copy()
    for j, p, a in zip(joint, parent, affect):
        if is_camera:
            tmp = map_with_camera_matrix(tmp, j, p, a)
        else:
            tmp = map_projection(tmp, j, p, a)
            
    return tmp
