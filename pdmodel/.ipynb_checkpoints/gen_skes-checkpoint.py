import sys
import os.path as osp
sys.path.insert(0, osp.dirname(osp.realpath(__file__)))
from tools.utils import get_path
from common.model_poseformer import *
from common.skeleton import Skeleton
from common.generators import *
from tools.preprocess import h36m_coco_format, revise_kpts
from tools.inference import gen_pose
import settings

cur_dir, chk_root, data_root, lib_root, output_root = get_path(__file__)
sys.path.pop(0)

skeleton = Skeleton(parents=[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15],
                    joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
                    joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])

joints_left, joints_right = [4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]
kps_left, kps_right = [4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]
rot = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)
keypoints_metadata = {'keypoints_symmetry': (joints_left, joints_right), 'layout_name': 'Human3.6M', 'num_joints': 17}
width, height = (1920, 1080)


def load_model_layer():
    chk = settings.chk_pth

    print('Loading model ...')
    model_pos = PoseTransformer(num_frame=81, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
        num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_path_rate=0.1)
    if torch.cuda.is_available():
        model_pos = nn.DataParallel(model_pos)
        model_pos = model_pos.cuda()

    # Loading pre-trained model
    checkpoint = torch.load(chk)
    model_pos.load_state_dict(checkpoint['model_pos'])

    model_pos = model_pos.eval()

    return model_pos

def generate_skeletons(width, height, npy_path, saving_path=None):
    raw = np.load(npy_path, allow_pickle=True)
    keypoints = []
    scores = []
    for i in range(len(raw)):
        kpt = raw[i]['keypoint']
        keypoints.append(kpt[:, :2])
        scores.append(kpt[:, 2])
    
    keypoints = np.array([keypoints])
    scores = np.array([scores])
    
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
    kpts = revise_kpts(keypoints, scores, valid_frames)
    valid_frames = [[i for i in np.arange(kpts.shape[1])]]
    
    # Loading 3D pose model
    model_pos = load_model_layer()
    
    # Generating 3D poses
    print('Generating 3D human pose ...')
    prediction = gen_pose(kpts, valid_frames, width, height, model_pos, pad=40, causal_shift=0)

    # Adding absolute distance to 3D poses and rebase the height
    prediction[0][:, :, 2] -= np.amin(prediction[0][:, :, 2])
    
    print('Saving 3D reconstruction...')
    if saving_path is not None:
        np.savez_compressed(saving_path, reconstruction=prediction)
    else:
        output_npz = './output/' + npy_path.split('/')[-1].split('.')[0] + '.npz'
        np.savez_compressed(output_npz, reconstruction=prediction)
    print('Completing saving...')


if __name__ == "__main__":
    npy_path = 'processed_20200430_1C.npy'
    generate_skeletons(
        width=1000,
        height=1000,
        npy_path=npy_path
    )
    