import json
import os
import torch
from lib.config import CONF
import numpy as np
import torch.nn.functional as F
import threading
from plyfile import PlyData, PlyElement
from utils.val_helper import decode_caption, check_candidates, organize_candidates, prepare_corpus, collect_results_cpu, \
    save_pred_instances, save_gt_instances
import datetime

import lib.capeval.bleu.bleu as capblue
import lib.capeval.cider.cider as capcider
import lib.capeval.rouge.rouge as caprouge
import lib.capeval.meteor.meteor as capmeteor

# caption_decoded= {'test': ['sos there is a black lujiachen eos']}
# gt_caption={'test': ['sos there is a black dog eos']}
#
# print(threading.Lock())
#
# bleu = capblue.Bleu(4).compute_score(gt_caption,caption_decoded)
# cider = capcider.Cider().compute_score(gt_caption,caption_decoded)
# rouge = caprouge.Rouge().compute_score(gt_caption,caption_decoded)
# meteor = capmeteor.Meteor().compute_score(gt_caption, caption_decoded)
# print(bleu)
# print(cider)
# print(rouge)
# print(meteor)


# print('test' in caption_decoded.keys())

# a = np.zeros(10)
# print(np.where(a == 1)[0].shape[0])
#
# plydata = PlyData.read('/home/luk/DenseCap/data/scannet/val/scene0427_00_vh_clean_2.ply')
# num_verts = plydata['vertex'].count
#
# print(plydata['vertex'].data)
#
# lines = open('/home/luk/DenseCap/data/scannet/val/scene0427_00.txt').readlines()
# axis_align_matrix = None
# for line in lines:
#     if 'axisAlignment' in line:
#         axis_align_matrix = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]
#
# print(axis_align_matrix)
# axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
#
# vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
# vertices[:, 0] = plydata['vertex'].data['x']
# vertices[:, 1] = plydata['vertex'].data['y']
# vertices[:, 2] = plydata['vertex'].data['z']
#
# pts = np.ones((vertices.shape[0], 4))
# pts[:, 0:3] = vertices[:, 0:3]  # using homogeneous coordinates
# pts = np.dot(pts, axis_align_matrix.transpose())  # Nx4
# aligned_vertices = np.copy(vertices)
# aligned_vertices[:, 0:3] = pts[:, 0:3]
#
# aligned_vertices[:, 0:3] = aligned_vertices[:, 0:3] - aligned_vertices[:, 0:3].mean(0)
# vertices[:, 0:3] = vertices[:, 0:3] - vertices[:, 0:3].mean(0)
#
# plydata['vertex'].data['x'] = vertices[:, 0]
# plydata['vertex'].data['y'] = vertices[:, 1]
# plydata['vertex'].data['z'] = vertices[:, 2]
#
# print(plydata['vertex'].data)
#
# plydata.write('test0427.ply')
#
# box_max = np.ones(3)
# box_min = np.ones(3) * -1
# color = np.random.rand(3) * 255
# print(box_min)
# print(box_max)
# print(color)

# corpus_path = os.path.join(CONF.PATH.OUTPUT, "corpus_train.json")
# if not os.path.exists(corpus_path):
#     print("preparing corpus...")
#     raw_data = json.load(open(os.path.join(CONF.PATH.DATA, "scanrefer/ScanRefer_filtered_train.json")))
#     corpus = prepare_corpus(raw_data, CONF.TRAIN.MAX_DES_LEN)
#     with open(corpus_path, "w") as f:
#         json.dump(corpus, f, indent=4)

# mask = torch.ones(2,4)
# mask[1,1] = 0
# mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1).reshape(-1)
#
# print(mask)

# caption_decoded = {0: ['2 19 6 7 32 76 41 5 8 29 7 142 206 28 7 53 11 8 5 3'], 1: ['2 4 109 6 40 167 4 18 27 9 4 24 5 4 63 12 6 55 28 78 237 11 141 5 3'], 2: ['2 4 137 86 6 20 14 6 30 10 7 189 86 5 8 29 7 251 168 10 77 17 14 78 149 6 7 251 168 11 3'], 3: ['2 22 14 23 12 5 4 12 6 75 58 7 23 16 5 3'], 4: ['2 13 6 7 23 41 5 8 6 59 7 104 16 5 3'], 5: ['2 13 6 7 55 137 86 5 8 6 11 4 85 5 3'], 6: ['2 13 6 7 55 137 86 5 8 6 15 4 36 5 3'], 7: ['2 13 6 7 12 5 8 6 159 166 49 4 16 5 3']}
#
# gt_caption = {0: ['2 13 6 7 442 41 5 8 6 59 7 84 16 5 3', '2 467 1142 467 197 41 28 350 1321 11 4 67 553 5 41 29 247 76 54 14 7 84 16 15 34 9 8 5 3', '2 4 41 6 15 4 36 9 45 398 5 4 41 29 7 84 16 28 52 193 653 15 34 9 8 5 3', '2 149 4 89 96 101 11 4 21 1 15 4 36 9 4 24 62 6 143 153 78 257 31 6 7 106 154 117 5 4 106 3', '2 4 41 6 68 7 84 16 62 6 30 10 45 76 54 5 4 41 6 154 14 29 7 460 92 5 3'], 1: ['2 4 55 86 6 48 4 225 14 4 60 1 5 8 6 11 4 17 9 7 252 86 5 3', '2 7 253 55 137 86 5 8 6 111 10 4 139 5 3', '2 13 6 7 55 137 86 5 8 6 10 4 17 9 74 86 5 3', '2 19 6 7 32 55 137 86 5 8 6 17 9 74 55 137 86 5 3', '2 13 6 7 55 137 86 5 8 6 11 4 17 9 74 86 5 3'], 2: ['2 4 37 38 6 4 92 22 5 8 6 40 43 4 34 9 4 24 151 4 34 25 5 8 6 274 151 4 34 25 31 3', '2 13 6 7 32 37 38 5 8 6 10 4 18 9 4 88 5 3', '2 19 6 7 99 37 38 5 8 6 15 34 9 4 25 14 30 10 4 33 5 3', '2 19 6 7 99 37 38 5 8 6 15 34 9 201 14 30 10 4 33 5 3', '2 4 37 38 6 10 4 17 9 4 25 5 4 37 38 6 15 34 9 4 39 5 3'], 3: ['2 4 391 12 15 49 4 234 9 4 16 5 8 6 11 4 223 27 9 4 24 160 4 25 31 14 450 4 20 158 5 3', '2 19 6 78 63 12 15 4 105 9 13 24 2125 67 120 38 246 385 5 8 79 15 34 9 7 100 405 26 14 29 45 3', '2 13 6 4 391 12 11 4 17 27 9 4 24 49 4 34 9 4 24 5 8 6 165 49 4 20 158 5 3', '2 4 12 6 15 4 70 9 4 16 5 4 12 6 11 4 17 27 9 4 16 5 3'], 4: ['2 13 6 7 442 41 5 8 6 59 7 84 16 5 3', '2 467 1142 467 197 41 28 350 1321 11 4 67 553 5 41 29 247 76 54 14 7 84 16 15 34 9 8 5 3', '2 4 41 6 15 4 36 9 45 398 5 4 41 29 7 84 16 28 52 193 653 15 34 9 8 5 3', '2 149 4 89 96 101 11 4 21 1 15 4 36 9 4 24 62 6 143 153 78 257 31 6 7 106 154 117 5 4 106 3', '2 4 41 6 68 7 84 16 62 6 30 10 45 76 54 5 4 41 6 154 14 29 7 460 92 5 3'], 5: ['2 4 55 86 6 48 4 225 14 4 60 1 5 8 6 11 4 17 9 7 252 86 5 3', '2 7 253 55 137 86 5 8 6 111 10 4 139 5 3', '2 13 6 7 55 137 86 5 8 6 10 4 17 9 74 86 5 3', '2 19 6 7 32 55 137 86 5 8 6 17 9 74 55 137 86 5 3', '2 13 6 7 55 137 86 5 8 6 11 4 17 9 74 86 5 3'], 6: ['2 4 37 38 6 4 92 22 5 8 6 40 43 4 34 9 4 24 151 4 34 25 5 8 6 274 151 4 34 25 31 3', '2 13 6 7 32 37 38 5 8 6 10 4 18 9 4 88 5 3', '2 19 6 7 99 37 38 5 8 6 15 34 9 4 25 14 30 10 4 33 5 3', '2 19 6 7 99 37 38 5 8 6 15 34 9 201 14 30 10 4 33 5 3', '2 4 37 38 6 10 4 17 9 4 25 5 4 37 38 6 15 34 9 4 39 5 3'], 7: ['2 4 391 12 15 49 4 234 9 4 16 5 8 6 11 4 223 27 9 4 24 160 4 25 31 14 450 4 20 158 5 3', '2 19 6 78 63 12 15 4 105 9 13 24 2125 67 120 38 246 385 5 8 79 15 34 9 7 100 405 26 14 29 45 3', '2 13 6 4 391 12 11 4 17 27 9 4 24 49 4 34 9 4 24 5 8 6 165 49 4 20 158 5 3', '2 4 12 6 15 4 70 9 4 16 5 4 12 6 11 4 17 27 9 4 16 5 3']}

gt_caption = {1: [
    'sos this is a long kitchen counter . it is grey . there are ovens under it . eos',
    'sos this is a long kitchen counter . it is below two windows . eos',
    'sos it is a light gray counter . it sit on top of wooden cabinets that go along one side of the kitchen . it is on the same side of eos',
    'sos this is a black kitchen counter . it is on top of a kitchen cabinet . eos',
    'sos there is a long kitchen counter . it has a window on both sides and a sink to its left . eos'],
    2: [
        "sos it is a tall gray trash can . the trash can sits along the wall in the kitchen next to the console table that is under the tv . eos",
        "sos it is a gray trash can . the trash can sits in the corner by where the tv is . eos",
        "sos this is a round trash can . it is in the corner of the room . eos",
        "sos a gray trash can in the middle of the pillar and a small brown table . above the small table is a black tv , in front of the trash eos",
        "sos this is a gray trash can . it is to the right of a table . eos"],
    3: [
        "sos eos"],
    4: [
        "sos the chair is one of four chairs facing the table . it is the second chair from the left . eos",
        "sos the chair is at the table . there are two chairs to the right of it , and one chair to the left of it . eos",
        "sos you are looking for the chair on the side of the table near the ovens . the chair will be the one in the center . eos",
        "sos a brown chair . it is placed next to a table . in the right - to - left direction it is the third chair . eos"],
    5: [
        "sos the chair is one of four chairs facing the table . it is the second chair from the left . eos",
        "sos the chair is at the table . there are two chairs to the right of it , and one chair to the left of it . eos",
        "sos you are looking for the chair on the side of the table near the ovens . the chair will be the one in the center . eos",
        "sos a brown chair . it is placed next to a table . in the right - to - left direction it is the third chair . eos"],
}

caption_decoded = {1: ['sos eos'], 2: ['sos eos'], 3: ['sos eos'], 4: ['sos eos'], 5: ['sos eos']}

# bleu = capblue.Bleu(4).compute_score(gt_caption, caption_decoded)
# cider = capcider.Cider().compute_score(gt_caption, caption_decoded)
# rouge = caprouge.Rouge().compute_score(gt_caption, caption_decoded)
# meteor = capmeteor.Meteor().compute_score(gt_caption, caption_decoded)
# print(bleu)
# print(cider)
# print(rouge)
# print(meteor)

# a = [1,2,3,4]
# print(np.asarray(a).mean())

# a = torch.ones(4).long()
# a[0] = -1
# print(a)
#
# b = torch.zeros(4).long()
# for i in range(4):
#     b[i] = max(a[i], 0)
#
# print(b)

# current_time = datetime.datetime.now()
# time_string = current_time.strftime("%H:%M:%S")
# print(time_string)

a= torch.zeros(2,10)
a[0,0]=1
print(a)
print(a.repeat(1,2).reshape(-1,10))