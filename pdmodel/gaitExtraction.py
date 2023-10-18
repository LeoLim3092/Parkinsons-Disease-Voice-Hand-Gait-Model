# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
import cv2
import mmcv
import copy
import numpy as np
from mmpose.apis import (inference_top_down_pose_model,
                    init_pose_model, process_mmdet_results,
                    vis_pose_result)
from mmpose.datasets import DatasetInfo

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

from .tools.utils import generate_2d_result
from .gen_skes import generate_skeletons
import settings


def gait_extraction(video_path, out_video_root="", save_out_video=False):
    """Visualize the demo video (support both single-frame and multi-frame).
    Using mmdet to detect the human.
    """

    thickness = 1
    radius = 4
    kpt_thr = 0.3
    bbox_thr = 0.3
    det_cat_id = 1
    det_config = settings.det_config_pth
    det_checkpoint = settings.det_checkpoint_pth
    pose_config = settings.pose_config_pth
    pose_checkpoint = settings.pose_checkpoint_pth
    pose_det_results_list = []
    person_results_list = []
    print('Initializing gait model...')
    # build the detection model from a config file and a checkpoint file
    det_model = init_detector(det_config, det_checkpoint, device="cuda:0")

    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(pose_config, pose_checkpoint, device="cuda:0")

    dataset = pose_model.cfg.data['test']['type']
    # get datasetinfo
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    # read video
    video = mmcv.VideoReader(video_path)
    assert video.opened, f'Failed to load video file {video_path}'

    if out_video_root == '':
        save_out_video = False
    else:
        os.makedirs(out_video_root, exist_ok=True)

    if save_out_video:
        fps = video.fps
        size = (video.width, video.height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(
            os.path.join(out_video_root,
                         f'vis_gait_{os.path.basename(video_path)}'), fourcc,
            fps, size)

    # whether to return heatmap, optional
    return_heatmap = False

    # return the output of some desired layers,
    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(video)):
        # get the detection results of current frame
        # the resulting box is (x1, y1, x2, y2)

        mmdet_results = inference_detector(det_model, cur_frame)
        # keep the person class bounding boxes.
        person_results = process_mmdet_results(mmdet_results, det_cat_id)

        # test a single image, with a list of bboxes.
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            cur_frame,
            person_results,
            bbox_thr=bbox_thr,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        if save_out_video:
            # show the results
            vis_frame = vis_pose_result(
                pose_model,
                cur_frame,
                pose_results,
                dataset=dataset,
                dataset_info=dataset_info,
                kpt_score_thr=kpt_thr,
                radius=radius,
                thickness=thickness,
                show=False)

            videoWriter.write(vis_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        pose_det_results_list.append(copy.deepcopy(pose_results))
        person_results_list.append(copy.deepcopy(person_results))

    out_pose_path = out_video_root + f'pose_results_{os.path.basename(video_path)[:-4]}.npy'
    out_person_path = out_video_root + f'person_results_{os.path.basename(video_path)[:-4]}.npy'

    np.save(out_pose_path, np.array(pose_det_results_list))
    np.save(out_person_path, np.array(person_results_list))
    path_2d = f'{out_video_root}2d_{os.path.basename(video_path)[:-4]}.npy'

    generate_2d_result(out_pose_path, video_path=video_path, output_folder=out_video_root, save_path=path_2d)

    generate_skeletons(
        width=1000,
        height=1000,
        npy_path=path_2d,
        saving_path=f"{out_video_root}3d_{os.path.basename(video_path)[:-4]}.npz"
    )

    if save_out_video:
        videoWriter.release()


def gait_checking(video_path):

    """Visualize the demo video (support both single-frame and multi-frame).
    Using mmdet to detect the human.
    """
    det_cat_id = 1
    det_config = settings.det_config_pth
    det_checkpoint = settings.det_checkpoint_pth

    # build the detection model from a config file and a checkpoint file
    det_model = init_detector(det_config, det_checkpoint, device="cuda:0")
    # read video
    video = mmcv.VideoReader(video_path)
    fps = video.fps
    frame_count = len(video)
    video_length_seconds = frame_count / fps

    assert video.opened, f'Failed to load video file {video_path}'
    predict_bbox_thershold = 0

    if video_length_seconds < 30:
        return "too short!"

    for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(video)):
        # get the detection results of current frame
        # the resulting box is (x1, y1, x2, y2)

        mmdet_results = inference_detector(det_model, cur_frame)
        # keep the person class bounding boxes.
        person_results = process_mmdet_results(mmdet_results, det_cat_id)

        if person_results:
            predict_bbox_thershold += person_results[0]["bbox"][-1]
        else:
            predict_bbox_thershold += 0

        if frame_id == int(5 * fps):
            return predict_bbox_thershold / frame_id


if __name__ == '__main__':
    video_path = "/mnt/pd_app/walk/20200806_3C.mp4"
    #video_path = "/mnt/pd_app/results/test/test.mp4"
    out_video_root = "/mnt/pd_app/results/test/"
    gait_extraction(video_path, out_video_root)
