import subprocess as sp
from tqdm import tqdm
import cv2
import numpy as np
import os
import sys

import matplotlib.pyplot as plt


colors = ['k','k', 'b','b','b', 'k','k','k', 'k','k', 'k','k','k', 'b','b','b']
bones = [
    [8,9],[9,10], [8,14],[14,15],[15,16], [8,11],[11,12],[12,13],
    [8,7],[7,0], [0,4],[4,5],[5,6], [0,1],[1,2],[2,3]
]

class Plotter3D():
    def __init__(self, usebuffer=False, elev=None, azim=None, axis='on', axis_tick='on', no_margin=False):
        fig = plt.figure()
        self.ax = fig.add_subplot(111, projection='3d')
        self.axis_tick = axis_tick

        if no_margin:
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.margins(0,0,0)
        self.ax.view_init(elev=elev, azim=azim)
        self.ax.axis(axis)
        if axis_tick=='off':
            self.ax.set_xticklabels([])
            self.ax.set_yticklabels([])
            self.ax.set_zticklabels([])
        self.lines = []
        self.lines_buff = []
        self.line_pos = 0
        self.usebuffer = usebuffer
        self.fig = fig 

    def show(self, ion=True):
        if ion:
            plt.ion()
        plt.show(block=True)

    def clear(self):
        self.ax.clear()
        if self.axis_tick=='off':
            self.ax.set_xticklabels([])
            self.ax.set_yticklabels([])
            self.ax.set_zticklabels([])
        
    def plot(self, xs,ys,zs, lims=None, **kwargs):
        if lims is not None:
            self.ax.set_xlim(lims[0])
            self.ax.set_ylim(lims[1])
            self.ax.set_zlim(lims[2])

        if (len(self.lines)==0) or (not self.usebuffer):
            a = self.ax.plot(xs, ys, zs, **kwargs)
            self.lines_buff.append(a)
        else:
            line = self.lines[self.line_pos][0]
            line.set_data(xs,ys)
            line.set_3d_properties(zs)
            self.line_pos += 1
        
    def set_title(self, title):
        self.ax.set_title(title)
        
    def update(self, require_img=False):
        try:
            self.ax.set_proj_type('persp')
            self.ax.draw_artist(self.ax.patch)
            for line in self.lines:
                self.ax.draw_artist(line)
            self.fig.canvas.update()
        except:
            self.ax.set_proj_type('persp')
            self.fig.canvas.draw()

        if require_img:
            # image = np.fromstring(self.fig.canvas.tostring_rgb(), dtype='uint8')
            s, (width, height) = self.fig.canvas.print_to_buffer()
            image = np.fromstring(s, np.uint8).reshape((height, width, 4))

        self.fig.canvas.flush_events()
        if len(self.lines)==0:
            self.lines = self.lines_buff
            self.lines_buff = []
        self.line_pos = 0

        if require_img:
            return image


plt3D = Plotter3D(usebuffer=False, no_margin=True, axis='off', axis_tick='off', azim=66, elev=15)
def plotSke(pts):
    plt3D.clear()
    radius = 1.7
    for i, p in enumerate(bones):
        color = colors[i]
        xs = [pts[p[0]][0], pts[p[1]][0]]
        ys = [pts[p[0]][1], pts[p[1]][1]]
        zs = [pts[p[0]][2], pts[p[1]][2]]
        lims = [[-radius,radius], [-radius,radius], [0,radius]]
        zorder = 3
        plt3D.plot(xs,ys,zs,lims=lims, zdir='z', marker='o', linewidth=3, zorder=zorder, markersize=2, color=color)
    img = plt3D.update(require_img=True)
    return img


def draw(result, re_kpts=None, video_path=None, fps=30, name='output.mp4'):
    k = len(result)
    n = len(result[0])
    
    print('Drawing...')
    frames = []
    
    if video_path is not None:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(5)
    
    for i in tqdm(range(n)):
        if re_kpts is None or video_path is None:
            fig, axes = plt.subplots(1, k, figsize=(10+10*k,10), constrained_layout=True)
            if k == 1:
                axes = [axes]
        else:
            success, frame = cap.read()
            fig, axes = plt.subplots(1, k+1, figsize=(10+10*k,10), constrained_layout=True)
            
            for kpt in re_kpts[i]:
                cv2.circle(frame, (int(kpt[0]), int(kpt[1])), radius=5, color=(0, 0, 255), thickness=-1)
            axes[0].imshow(frame)
        
        for p in range(k):
            img = plotSke(result[p][i])
            if re_kpts is None:
                axes[p].imshow(img)
            else:
                axes[p+1].imshow(img)
        
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
        
        axes[-1].set_title('predicted', pad=35)
        if k==2:
            axes[-2].set_title('gt', pad=35)
        
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
        
        plt.close(fig)

    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(name, fourcc, fps, (width, height))
    print('Generating video...')
    for i in range(len(frames)):
        video.write(frames[i])

    cv2.destroyAllWindows()
    video.release()

if __name__ == '__main__':
    '''
    result:     3D poses in the shape of (k, n, 17, 3)
                k: the number of skeletons to be shown. You may show groundtruth and predicted skeletons simultaneously.
                n: the length of the video
    re_kpts:    2D keypoints in the shape of (n, 17, 2)
    video_path: path to the video source
    fps:        fps of the output video
    name:       name of the output viedeo
    '''
    
    # result = np.random.rand(1, 1715, 17, 3)
    result = np.load('3d_IMG_9986.npz')['reconstruction']
    draw(result, re_kpts=None, video_path=None, fps=30, name='output.mp4')