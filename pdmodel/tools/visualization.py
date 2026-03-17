import subprocess as sp
from tqdm import tqdm
import cv2
import numpy as np

import pdmodel.tools.mysulplotter as plotter
import matplotlib.pyplot as plt

colors = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9']
plt3D = plotter.Plotter3D(usebuffer=False, no_margin=True, azim=66, elev=70)
bones = [
    [8,9],[9,10], [8,14],[14,15],[15,16], [8,11],[11,12],[12,13],
    [8,7],[7,0], [0,4],[4,5],[5,6], [0,1],[1,2],[2,3]
]

def get_fps(filename):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', filename]
    with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
        for line in pipe.stdout:
            a, b = line.decode().strip().split('/')
            return int(a) / int(b)

def plotSke(pts):
    plt3D.clear()
    radius = 1.7
    for i, p in enumerate(bones):
        color = colors[i%len(colors)]
        xs = [pts[p[0]][0], pts[p[1]][0]]
        ys = [pts[p[0]][1], pts[p[1]][1]]
        zs = [pts[p[0]][2], pts[p[1]][2]]
        lims = [[-radius,radius], [-radius,radius], [0,radius]]
        zorder = 3
        plt3D.plot(xs,ys,zs,lims=lims, zdir='z', marker='o', linewidth=3, zorder=zorder, markersize=2, color=color)
    img = plt3D.update(require_img=True)
    return img

def draw(result, video_path, name):
    fps = get_fps(video_path)
    k = len(result)
    n = len(result[0])
    
    print('Drawing...')
    frames = []
    cap = cv2.VideoCapture(video_path)
    for i in tqdm(range(n)):
        success, frame = cap.read()
        
        fig, axes = plt.subplots(1, k+1, figsize=(5+4*k,5), constrained_layout=True)
        axes[0].imshow(frame)
        for p in range(k):
            img = plotSke(result[p][i])
            axes[p+1].imshow(img)
        
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
                    
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
        
        plt.close(fig)

    height, width, layers = frames[0].shape
    video = cv2.VideoWriter(name, 0, fps, (width, height))
    print('Generating video...')
    for i in range(len(frames)):
        video.write(frames[i])

    cv2.destroyAllWindows()
    video.release()