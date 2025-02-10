# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------


from collections import defaultdict
from glob import glob
import json
import os
import cv2
import numpy as np
import subprocess
import random
from tqdm import tqdm
from PIL import Image, ImageDraw

from scipy.optimize import linear_sum_assignment as linear_assignment

# 计算两个box的IOU
def bboxes_iou(bboxes1,bboxes2):
	bboxes1 = np.transpose(bboxes1)
	bboxes2 = np.transpose(bboxes2)

	# 计算两个box的交集：交集左上角的点取两个box的max，交集右下角的点取两个box的min
	int_ymin = np.maximum(bboxes1[0][:, None], bboxes2[0])
	int_xmin = np.maximum(bboxes1[1][:, None], bboxes2[1])
	int_ymax = np.minimum(bboxes1[2][:, None], bboxes2[2])
	int_xmax = np.minimum(bboxes1[3][:, None], bboxes2[3])

	# 计算两个box交集的wh：如果两个box没有交集，那么wh为0(按照计算方式wh为负数，跟0比较取最大值)
	int_h = np.maximum(int_ymax-int_ymin,0.)
	int_w = np.maximum(int_xmax-int_xmin,0.)

	# 计算IOU
	int_vol = int_h * int_w # 交集面积
	vol1 = (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1]) # bboxes1面积
	vol2 = (bboxes2[2] - bboxes2[0]) * (bboxes2[3] - bboxes2[1]) # bboxes2面积
	IOU = int_vol / (vol1[:, None] + vol2 - int_vol) # IOU=交集/并集
	return IOU

def get_color(i):
    return [(i * 23 * j + 43) % 255 for j in range(3)]


def show_gt(img_list, output="output.mp4"):
    h, w, _ = cv2.imread(img_list[0]).shape
    command = [
        "anaconda3/envs/detrex/bin/ffmpeg",
        '-y',  # overwrite output file if it exists
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-s', f'{w}x{h}',  # size of one frame
        '-pix_fmt', 'bgr24',
        '-r', '20',  # frames per second
        '-i', '-',  # The imput comes from a pipe
        '-s', f'{w//2*2}x{h//2*2}',
        '-an',  # Tells FFMPEG not to expect any audio
        '-loglevel', 'error',
        # '-crf', '26',
        '-b:v', '0',
        '-pix_fmt', 'yuv420p'
    ]
    # writing_process = subprocess.Popen(command + [output], stdin=subprocess.PIPE)
    fps = 16 
    size = (w,h) 
    videowriter = cv2.VideoWriter(output,cv2.VideoWriter_fourcc('M','J','P','G'), fps, size)


    for i, path in enumerate(tqdm(sorted(img_list))):
        im = cv2.imread(path)
        det_bboxes = []
        motr_bboxes = []
        for det in det_db[path.replace('data/', '').replace('.jpg', '.txt').replace('dancetrack/', 'DanceTrack/')]:
            x1, y1, w, h, s = map(float, det.strip().split(','))
            x1, y1, w, h = map(int, [x1, y1, w, h])
            im = cv2.rectangle(im, (x1, y1), (x1+w, y1+h), (255, 255, 255), 2)
            im = cv2.putText(im, '%0.2f'%s, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            det_bboxes.append([x1, y1, x1+w, y1+h])

        det_bboxes = np.array(det_bboxes)
        motr_bboxes = np.array(motr_bboxes)
        ious = bboxes_iou(det_bboxes, motr_bboxes)
        matching = linear_assignment(-ious)
        matched = sum(ious[matching[0], matching[1]] > 0.5)
        im = cv2.putText(im, f"{matched}/{len(det_bboxes)}/{len(motr_bboxes)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, get_color(j), 3)
        cv2.putText(im, "{}".format(os.path.basename(path)[:-4]), (120,120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 6)
        # writing_process.stdin.write(im.tobytes())
        videowriter.write(im)
        
    videowriter.release()


if __name__ == '__main__':

    labels_full = defaultdict(lambda : defaultdict(list))
    imgid2name = defaultdict()
    def _add_mot_folder(mot_path, split_dir):
        print("Adding", split_dir)
        labels = json.load(open(os.path.join(mot_path, split_dir)))
        for ann in labels['images']:
            imgid2name[ann['id']] = ann['file_name']
        for ann in labels['annotations']:
            vid = ann['video_id']
            t = ann['image_id']
            x, y, w, h = ann['bbox']
            i = ann['track_id']
            crowd = ann['iscrowd']
            cl = ann['category_id']
            labels_full[vid][t].append([x, y, w, h, i, crowd, cl])
        return labels_full, imgid2name
    
    mot_path = 'data/'
    labels_full, imgid2name = _add_mot_folder(mot_path, 'tao/annotations/train.json')
    indices = []
    vid_files = list(labels_full.keys())
    for vid in vid_files:
        t_min = min(labels_full[vid].keys())
        t_max = max(labels_full[vid].keys()) + 1
        for t in range(t_min, t_max):
            indices.append((vid, t))
          
    vid_old = None
    random.shuffle(vid_files)
    videowriter = None
    for vid in vid_files:
        print(vid)
        t_min = min(labels_full[vid].keys())
        t_max = max(labels_full[vid].keys()) + 1
        for idx in range(t_min, t_max):
        # vid, idx = indices[idx]
            img_path = os.path.join(mot_path, 'tao/frames', imgid2name[idx])
            img = Image.open(img_path)
            if vid != vid_old:
                vid_old = vid
                w, h = img._size
                fps = 1
                size = (w,h) 
                if videowriter is not None:
                    videowriter.release()
                videowriter = cv2.VideoWriter('tmp/'+imgid2name[idx].split('/')[-2]+'.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fps, size)
            im = np.array(img)
            for *xywh, id, crowd, cl in labels_full[vid][idx]:
                x1, y1, w, h = xywh
                x1, y1, w, h = map(int, [x1, y1, w, h])
                im = cv2.rectangle(im, (x1, y1), (x1+w, y1+h), (255, 255, 255), 2)
                im = cv2.putText(im, '%d'%id, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            videowriter.write(im)
    
    videowriter.release()
    