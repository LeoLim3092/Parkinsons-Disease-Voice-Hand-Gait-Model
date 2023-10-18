import torch
import numpy as np
import sys
import os.path as osp


pre_dir = osp.join(osp.dirname(osp.realpath(__file__)), '../..')
sys.path.insert(0, pre_dir)
from common.camera import normalize_screen_coordinates, camera_to_world
from common.generators import *
sys.path.pop(0)


joints_left, joints_right = [4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]
kps_left, kps_right = [4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]
rot = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)

trans = torch.tensor([])
map_list = [0,8,11,12,13,14,15,16,4,5,6,1,2,3,9]
is_init = False

def eval_data_prepare(receptive_field, inputs_2d):
    inputs_2d_p = torch.squeeze(inputs_2d)
    out_num = inputs_2d_p.shape[0] - receptive_field + 1
    eval_input_2d = torch.empty(out_num, receptive_field, inputs_2d_p.shape[1], inputs_2d_p.shape[2])
    for i in range(out_num):
        eval_input_2d[i,:,:,:] = inputs_2d_p[i:i+receptive_field, :, :]
    return eval_input_2d

def evaluate(test_generator, model_pos, pad):
    prediction = []

    with torch.no_grad():
        for _, _, batch_2d in test_generator.next_epoch():

            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            inputs_2d = eval_data_prepare(pad*2+1, inputs_2d)
            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()

            # Positional model
            predicted_3d_pos = model_pos(inputs_2d)

            # Test-time augmentation (if enabled)
            if test_generator.augment_enabled():
                # Undo flipping and take average with non-flipped version
                predicted_3d_pos[1, :, :, 0] *= -1
                predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :, joints_right + joints_left]
                predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)
            
            prediction.append(predicted_3d_pos.squeeze().cpu().numpy())
        return prediction


def gen_pose(kpts, valid_frames, width, height, model_pos, pad, causal_shift=0):
    assert len(kpts.shape) == 4, 'The shape of kpts: {}'.format(kpts.shape)
    assert kpts.shape[0] == len(valid_frames)

    norm_seqs = []
    for index, frames in enumerate(valid_frames):
        seq_kps = kpts[index, frames]
        norm_seq_kps = normalize_screen_coordinates(seq_kps, w=width, h=height)
        norm_seqs.append(norm_seq_kps)
    gen = UnchunkedGenerator(None, None, norm_seqs, pad=pad, causal_shift=causal_shift, augment=True,
                             kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    prediction = evaluate(gen, model_pos, pad)
    # return prediction

    prediction_to_world = []
    for i in range(len(prediction)):
        sub_prediction = prediction[i]
        
        sub_prediction = camera_to_world(sub_prediction, R=rot, t=0)

        # sub_prediction[:, :, 2] -= np.expand_dims(np.amin(sub_prediction[:, :, 2], axis=1), axis=1).repeat([17], axis=1)
        # sub_prediction[:, :, 2] -= np.amin(sub_prediction[:, :, 2])

        prediction_to_world.append(sub_prediction)

    # prediction_to_world = np.asarray(prediction_to_world, dtype=np.float32)
    return prediction_to_world

def initTrans():
    global trans, map_list
    trans = torch.zeros(15,17)
    
    for i in range(15):
        trans[i][map_list[i]] = 1
    
    if torch.cuda.is_available():
        trans = trans.cuda()

def evaluate_test(test_generator, model_pos):
    global trans, is_init
    if not is_init:
        initTrans()
        is_init = True
    
    prediction = []

    with torch.no_grad():
        for _, _, batch_2d in test_generator.next_epoch():

            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            # print(inputs_2d.shape)
            # torch.Size([2, 103, 17, 2])
            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()
            inputs_2d = torch.matmul(trans, inputs_2d)

            # Positional model
            predicted_3d_pos = model_pos(inputs_2d)

            # Test-time augmentation (if enabled)
            if test_generator.augment_enabled():
                # Undo flipping and take average with non-flipped version
                predicted_3d_pos[1, :, :, 0] *= -1
                predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :, joints_right + joints_left]
                predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)

            prediction.append(predicted_3d_pos.squeeze(0).cpu().numpy())

        return prediction

def gen_pose_test(kpts, valid_frames, width, height, model_pos, pad, causal_shift=0):
    assert len(kpts.shape) == 4, 'The shape of kpts: {}'.format(kpts.shape)
    assert kpts.shape[0] == len(valid_frames)

    norm_seqs = []
    for index, frames in enumerate(valid_frames):
        seq_kps = kpts[index, frames]
        norm_seq_kps = normalize_screen_coordinates(seq_kps, w=width, h=height)
        norm_seqs.append(norm_seq_kps)

    kps_left_test = [2, 3, 4, 8, 9, 10]
    kps_right_test = [5, 6, 7, 11, 12, 13]
    gen = UnchunkedGenerator(None, None, norm_seqs, pad=pad, causal_shift=causal_shift, augment=False,
                             kps_left=kps_left_test, kps_right=kps_right_test,
                             joints_left=kps_left_test, joints_right=kps_right_test)
    prediction = evaluate_test(gen, model_pos)

    prediction_to_world = []
    for i in range(len(prediction)):
        sub_prediction = prediction[i]

        sub_prediction = camera_to_world(sub_prediction, R=rot, t=0)

        # sub_prediction[:, :, 2] -= np.expand_dims(np.amin(sub_prediction[:, :, 2], axis=1), axis=1).repeat([17], axis=1)
        # sub_prediction[:, :, 2] -= np.amin(sub_prediction[:, :, 2])

        prediction_to_world.append(sub_prediction)

    # prediction_to_world = np.asarray(prediction_to_world, dtype=np.float32)
    return prediction_to_world

def gen_pose_frame(kpts, width, height, model_pos, pad, causal_shift=0):
    # kpts: (M, T, N, 2)
    norm_seqs = []
    for kpt in kpts:
        norm_kpt = normalize_screen_coordinates(kpt, w=width, h=height)
        norm_seqs.append(norm_kpt)

    gen = UnchunkedGenerator(None, None, norm_seqs, pad=pad, causal_shift=causal_shift, augment=True,
                             kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    prediction = evaluate(gen, model_pos)

    prediction_to_world = []
    for i in range(len(prediction)):
        sub_prediction = prediction[i][0]
        sub_prediction = camera_to_world(sub_prediction, R=rot, t=0)
        sub_prediction[:, 2] -= np.amin(sub_prediction[:, 2])
        prediction_to_world.append(sub_prediction)

    return prediction_to_world


def gen_pose_frame_(kpts, width, height, model_pos, pad, causal_shift=0):
        # input (N, 17, 2) return (N, 17, 3)
        if not isinstance(kpts, np.ndarray):
            kpts = np.array(kpts)

        keypoints = normalize_screen_coordinates(kpts[..., :2], w=width, h=height)

        input_keypoints = keypoints.copy()
        # test_time_augmentation True
        from common.generators import UnchunkedGenerator
        gen = UnchunkedGenerator(None, None, [input_keypoints], pad=pad, causal_shift=causal_shift,
                                 augment=True, kps_left=kps_left, kps_right=kps_right,
                                 joints_left=joints_left, joints_right=joints_right)
        prediction = evaluate(gen, model_pos)
        prediction = camera_to_world(prediction[0], R=rot, t=0)
        prediction[:, :, 2] -= np.min(prediction[:, :, 2])
        return prediction
