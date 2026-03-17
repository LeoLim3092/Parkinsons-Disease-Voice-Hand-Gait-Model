import cv2 
import numpy as np 
from . import mysulplotter as plotter 
import os
from tqdm import tqdm
 
def mkdir(fname):
    dirname = os.path.dirname(fname)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

joints = [[8,9],[9,10], [8,14],[14,15],[15,16], [8,11],[12,13],[11,12], [8,7],[7,0], [4,5],[5,6],[0,4], [0,1],[1,2],[2,3]]
colors = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9']
joints32 = [[13,14],[14,15], [13,25],[25,26],[26,27], [13,17],[17,18],[18,19],
            [13,12],[12,0], [0,6],[6,7],[7,8], [0,1],[1,2],[2,3]]

def plot_skeleton(img, pts):
    img = img.copy()
    for j in joints:
        x1 = int(pts[j[0],0])
        x2 = int(pts[j[1],0])
        y1 = int(pts[j[0],1])
        y2 = int(pts[j[1],1])
        cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 2)
    return img 

plt = plotter.Plotter3D(usebuffer=False, no_margin=True, azim=181, elev=25)
# plt = plotter.Plotter3D(usebuffer=True, no_margin=True, axis_tick='off', azim=180, elev=0)
def plot_skeleton3d(inp, shift):
    plt.clear()
    # plt.set_title('%.2f, %.2f' % (shift[0], shift[1]))
    if isinstance(inp, np.ndarray):
        inp = [inp]
    for i in range(len(inp)):
        pts = inp[i]
        for iddd,j in enumerate(joints):
            collllor = colors[iddd%len(colors)]
            xs = [pts[j[0],2]+shift[1], pts[j[1],2]+shift[1]]
            ys = [-pts[j[0],0]-shift[0], -pts[j[1],0]-shift[0]]
            zs = [-pts[j[0],1], -pts[j[1],1]]
            lims = [[-2.0,2.0], [-2.0,2.0], [-1,2.0]]
            # lims=None
            zorder = 3
            # if (15 in j):
            # 	if pts[15,2]>pts[0,2]:
            # 		zorder = 2
            # 	else:
            # 		zorder = 5
            # if (12 in j):
            # 	if pts[12,2]>pts[0,2]:
            # 		zorder = 2
            # 	else:
            # 		zorder = 4
            # if (5 in j):
            # 	if pts[5,2]>pts[2,2]:
            # 		zorder = 2
            # 	else:
            # 		zorder = 4
            # if (2 in j):
            # 	if pts[2,2]>pts[5,2]:
            # 		zorder = 2
            # 	else:
            # 		zorder = 4
            plt.plot(xs,ys,zs,lims=lims, marker='o', linewidth=3, zorder=zorder, markersize=2, color=collllor)
    img = plt.update(require_img=True)
    return img 

def plot_skeleton3d_32joints(inp):
    plt.clear()
    if isinstance(inp, np.ndarray):
        inp = [inp]
    for i in range(len(inp)):
        pts = inp[i]
        for iddd,j in enumerate(joints32):
            collllor = colors[iddd%len(colors)]
            xs = [pts[j[0],1], pts[j[1],1]]
            ys = [pts[j[0],0], pts[j[1],0]]
            zs = [pts[j[0],2], pts[j[1],2]]
            lims = [[-2.0,2.0], [-2.0,2.0], [0,2.0]]
            zorder = 3
            plt.plot(xs,ys,zs,lims=lims, marker='o', linewidth=3, zorder=zorder, markersize=2, color=collllor)
    img = plt.update(require_img=True)
    return img 

# ugly code here
def draw_skeleton_raw(img, kpts):
    cmap = color_map()
    for idx,pts in enumerate(kpts):
        color = cmap[idx+1]
        # color = tuple(color)
        # print(color)
        pts = np.float32(pts)
        x0 = int(pts[0,0])
        y0 = int(pts[0,1])
        for j in joints:    
            x1 = int(pts[j[0],0])
            x2 = int(pts[j[1],0])
            y1 = int(pts[j[0],1])
            y2 = int(pts[j[1],1])
            cv2.line(img, (x1,y1), (x2,y2), (int(color[0]), int(color[1]), int(color[2])),3)
    return img

def draw_skeleton(img, kpts, ids):
    cmap = color_map()
    for idx,pts in enumerate(kpts):
        color = cmap[ids[idx]+1]
        # color = tuple(color)
        # print(color)
        pts = np.float32(pts)
        x0 = int(pts[0,0])
        y0 = int(pts[0,1])
        font = cv2.FONT_HERSHEY_SIMPLEX
        # img = cv2.circle(img,(x0,y0), 63, (0,0,255), -1)
        cv2.putText(img,'ID:%d'%ids[idx],(x0,y0), font, 1,(0,0,255),2,cv2.LINE_AA)
        for j in joints:
            x1 = int(pts[j[0],0])
            x2 = int(pts[j[1],0])
            y1 = int(pts[j[0],1])
            y2 = int(pts[j[1],1])
            cv2.line(img, (x1,y1), (x2,y2), (int(color[0]), int(color[1]), int(color[2])),3)
    return img 

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)
    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3
        cmap[i] = np.array([b,g,r])
    cmap = cmap/255 if normalized else cmap
    return cmap


# plot scene functions 
def fill_2d(pts):
    for i in range(len(pts)):
        if pts[i][0][0]==0 and pts[i][0][1]==0:
            pts[i] = pts[i-1]
    return pts 

def low_pass(data, cutoff):
    assert cutoff<data.shape[0]//2,'cutoff should be less than half seq'
    datahead = data[0]
    datatail = data[-1]
    datahead = np.stack([datahead] * 10, axis=0)
    datatail = np.stack([datatail] * 10, axis=0)
    data = np.concatenate([datahead, data, datatail], axis=0)
    for pt in range(data.shape[1]):
        for dim in range(data.shape[2]):
            pts1 = data[:,pt,dim]
            x = np.array(list(range(len(pts1))))

            fft1 = np.fft.fft(pts1)
            fft1_amp = np.fft.fftshift(fft1)
            fft1_amp = np.abs(fft1_amp)

            fft1[cutoff:-cutoff] = 0
            recover = np.fft.ifft(fft1)
            data[:,pt,dim] = recover
    data = data[10:-10]
    return data 

def order_data_by_frame(data):
    ids = list(range(len(data)))
    # get max length 
    maxlength = 0
    for i in ids:
        pts2d, ptsrel, ptsroot, start = data[i]
        pts2d = fill_2d(pts2d)
        # if len(pts2d)>40:
        # 	pts2d = low_pass(pts2d, 20)
        length = start + len(pts2d)
        if length > maxlength:
            maxlength = length
    # create empty list 
    res = [[] for _ in range(maxlength)] 
    for i in ids:
        pts2d, ptsrel, ptsroot, start = data[i]
        # if len(pts2d) < 20:
        # 	continue
        for j in range(len(pts2d)):
            res[j+start].append([pts2d[j], ptsrel[j], ptsroot[j], i]) # Add id here if useful
    return res 

def get_minmax(data):
    ids = list(range(len(data)))
    minmaxroot = [9999, -9999]
    minmax2d = [9999, -9999, 9999, -9999]
    for i in ids:
        pts2d, _, ptsroot, _ = data[i]
        # print(pts2d.shape)
        pts2dx = pts2d[:,0,0]
        pts2dy = pts2d[:,0,1]
        if minmax2d[0]>pts2dx.min():
            minmax2d[0] = pts2dx.min()
        if minmax2d[1]<pts2dx.max():
            minmax2d[1] = pts2dx.max()
        if minmax2d[2]>pts2dy.min():
            minmax2d[2] = pts2dy.min()
        if minmax2d[3]<pts2dy.max():
            minmax2d[3] = pts2dy.max()

        if minmaxroot[0]>ptsroot.min():
            minmaxroot[0] = ptsroot.min()
        if minmaxroot[1]<ptsroot.max():
            minmaxroot[1] = ptsroot.max()
    return minmax2d, minmaxroot

def normalize_pts3d(centerx, centery, dep, minmax2d, minmaxroot):
    center2d = [0.5 * (minmax2d[1] + minmax2d[0]), 0.5 * (minmax2d[2] + minmax2d[3])]
    centerroot = 0.5 * minmaxroot[0] + 0.5 * minmaxroot[1]

    centerx = (centerx - center2d[0]) *10 / (minmax2d[1] - minmax2d[0])
    centery = (centery - center2d[1]) *10 / (minmax2d[3] - minmax2d[2])

    dep = (dep - centerroot) * 12 / (minmaxroot[1] - minmaxroot[0]) + 6

    return centerx, centery, dep 

def plot_sequence(pts, adjust, outpath):
    for i in tqdm(range(len(pts))):
        p = pts[i]
        adjust[i][0] *= 0.005
        adjust[i][1] *= 0.0025
        img = plot_skeleton3d(p, adjust[i])
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(outpath, '%08d.png'%i), img)

def output_video_32joints(pts, outpath, FPS):
    frames = []
    for i in tqdm(range(len(pts))):
        p = pts[i]
        img = plot_skeleton3d_32joints(p)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        frames.append(img)

    height, width, layers = frames[0].shape
    video = cv2.VideoWriter(outpath, 0, FPS, (width, height))
    for i in range(len(frames)):
        video.write(frames[i])

    cv2.destroyAllWindows()
    video.release()

