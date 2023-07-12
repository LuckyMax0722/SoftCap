"""Modified from SparseConvNet data preparation: https://github.com/facebookres
earch/SparseConvNet/blob/master/examples/ScanNet/prepare_data.py."""

import argparse
import glob
import json
import multiprocessing as mp

import numpy as np
import plyfile
import scannet_util
import torch

# Map relevant classes to {0,1,...,19}, and ignored classes to -100
remapper = np.ones(150) * (-100)
for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
    remapper[x] = i
remapper[13] = 19
remapper[15] = 19
remapper[17] = 19
remapper[18] = 19
remapper[19] = 19
remapper[20] = 19
remapper[21] = 19
remapper[23] = 19
remapper[25] = 19
remapper[26] = 19
remapper[27] = 19
remapper[29] = 19
remapper[30] = 19
remapper[31] = 19
remapper[32] = 19
remapper[35] = 19
remapper[37] = 19
remapper[38] = 19
remapper[40] = 19

parser = argparse.ArgumentParser()
parser.add_argument('--data_split', help='data split (train / val / test)', default='train')
opt = parser.parse_args()

split = opt.data_split

print('data split: {}'.format(split))
files = sorted(glob.glob(split + '/*_vh_clean_2.ply'))

if opt.data_split != 'test':
    files2 = sorted(glob.glob(split + '/*_vh_clean_2.labels.ply'))
    files3 = sorted(glob.glob(split + '/*_vh_clean_2.0.010000.segs.json'))
    files4 = sorted(glob.glob(split + '/*[0-9].aggregation.json'))
    assert len(files) == len(files2)
    assert len(files) == len(files3)
    assert len(files) == len(files4), '{} {}'.format(len(files), len(files4))


def f_test(fn):
    print(fn)

    f = plyfile.PlyData().read(fn)
    points = np.array([list(x) for x in f.elements[0]])
    coords = np.ascontiguousarray(points[:, :3] - points[:, :3].mean(0))
    colors = np.ascontiguousarray(points[:, 3:6]) / 127.5 - 1

    torch.save((coords, colors), fn[:-15] + '_inst_nostuff.pth')
    print('Saving to ' + fn[:-15] + '_inst_nostuff.pth')


def f(fn):
    fn2 = fn[:-3] + 'labels.ply'
    fn3 = fn[:-15] + '_vh_clean_2.0.010000.segs.json'
    fn4 = fn[:-15] + '.aggregation.json'
    print(fn)

    f = plyfile.PlyData().read(fn)
    points = np.array([list(x) for x in f.elements[0]])
    coords = np.ascontiguousarray(points[:, :3] - points[:, :3].mean(0))
    colors = np.ascontiguousarray(points[:, 3:6]) / 127.5 - 1

    aligned_coords = np.ascontiguousarray(points[:, :3]).copy()
    lines = open(fn[:-15] + '.txt').readlines()
    axis_align_matrix = None

    # check if there is axisAlignment data (only train scene has)
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]

    # coordinatae transformation with axix_align_matrix
    if axis_align_matrix != None:
        axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
        pts = np.ones((aligned_coords.shape[0], 4))
        pts[:, 0:3] = aligned_coords[:, 0:3]  # using homogeneous coordinates
        pts = np.dot(pts, axis_align_matrix.transpose())  # Nx4
        aligned_coords[:, 0:3] = pts[:, 0:3]
        aligned_coords = aligned_coords[:, :3] - aligned_coords[:, :3].mean(0)
    else:
        print("No axis alignment matrix found")
        aligned_coords = coords  # for test scene, no transformation on coordinates

    f2 = plyfile.PlyData().read(fn2)
    sem_labels = np.array(remapper[np.array(f2.elements[0]['label'])])

    with open(fn3) as jsondata:
        d = json.load(jsondata)
        seg = d['segIndices']
    segid_to_pointid = {}
    for i in range(len(seg)):
        if seg[i] not in segid_to_pointid:
            segid_to_pointid[seg[i]] = []
        segid_to_pointid[seg[i]].append(i)

    instance_segids = []
    labels = []
    object_segids = []

    with open(fn4) as jsondata:
        d = json.load(jsondata)
        for x in d['segGroups']:
            # valid instance
            if scannet_util.g_raw2scannetv2[x['label']] != 'wall' and scannet_util.g_raw2scannetv2[
                x['label']] != 'floor':
                instance_segids.append(x['segments'])
                labels.append(x['label'])
                assert (x['label'] in scannet_util.g_raw2scannetv2.keys())
            # all objects
            object_segids.append(x['segments'])

    if (fn == 'val\scene0217_00_vh_clean_2.ply'
            and instance_segids[0] == instance_segids[int(len(instance_segids) / 2)]):
        instance_segids = instance_segids[:int(len(instance_segids) / 2)]
        object_segids = object_segids[:int(len(object_segids) / 2)]

    # check for instance
    check = []
    for i in range(len(instance_segids)):
        check += instance_segids[i]
    assert len(np.unique(check)) == len(check)

    # check for objects
    check = []
    for i in range(len(object_segids)):
        check += object_segids[i]
    assert len(np.unique(check)) == len(check)

    # instance_labels (not including wall,floor)
    instance_labels = np.ones(sem_labels.shape[0]) * -100
    for i in range(len(instance_segids)):
        segids = instance_segids[i]
        pointids = []
        for segid in segids:
            pointids += segid_to_pointid[segid]
        instance_labels[pointids] = i
        assert (len(np.unique(sem_labels[pointids])) == 1)

    # object_labels (it's the same as in scanrefer dataset but different from instance labels)
    object_labels = np.ones(sem_labels.shape[0]) * -100
    for i in range(len(object_segids)):
        segids = object_segids[i]
        pointids = []
        for segid in segids:
            pointids += segid_to_pointid[segid]
        object_labels[pointids] = i
        assert (len(np.unique(sem_labels[pointids])) == 1)

    num_instances = np.int(np.max(object_labels) + 1)
    instance_bboxes = np.zeros((num_instances, 8))  # also include object id
    aligned_instance_bboxes = np.zeros((num_instances, 8))  # also include object id
    for obj_id in range(num_instances):
        idx = object_labels == obj_id
        label_id = sem_labels[np.where(object_labels == obj_id)[0][0]]
        # bboxes in the original meshes
        obj_pc = coords[idx, 0:3]
        if len(obj_pc) == 0:
            continue
        # Compute axis aligned box
        # An axis aligned bounding box is parameterized by
        # (cx,cy,cz) and (dx,dy,dz) and label id
        # where (cx,cy,cz) is the center point of the box,
        # dx is the x-axis length of the box.
        xmin = np.min(obj_pc[:, 0])
        ymin = np.min(obj_pc[:, 1])
        zmin = np.min(obj_pc[:, 2])
        xmax = np.max(obj_pc[:, 0])
        ymax = np.max(obj_pc[:, 1])
        zmax = np.max(obj_pc[:, 2])
        bbox = np.array(
            [(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2, xmax - xmin, ymax - ymin, zmax - zmin, label_id,
             obj_id])  # also include object id
        instance_bboxes[obj_id, :] = bbox

        # bboxes in the aligned meshes
        obj_pc = aligned_coords[idx, 0:3]
        if len(obj_pc) == 0:
            continue
        # Compute axis aligned box
        # An axis aligned bounding box is parameterized by
        # (cx,cy,cz) and (dx,dy,dz) and label id
        # where (cx,cy,cz) is the center point of the box,
        # dx is the x-axis length of the box.
        xmin = np.min(obj_pc[:, 0])
        ymin = np.min(obj_pc[:, 1])
        zmin = np.min(obj_pc[:, 2])
        xmax = np.max(obj_pc[:, 0])
        ymax = np.max(obj_pc[:, 1])
        zmax = np.max(obj_pc[:, 2])
        bbox = np.array(
            [(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2, xmax - xmin, ymax - ymin, zmax - zmin, label_id,
             obj_id])  # also include object id
        aligned_instance_bboxes[obj_id, :] = bbox

    # filter data
    IGN_LABEL = np.array([-100, 0, 1])  # instance bbox doesn't contain ceiling, wall, floor
    bbox_mask = np.logical_not(np.in1d(instance_bboxes[:, -2], IGN_LABEL))
    instance_bboxes = instance_bboxes[bbox_mask, :]
    aligned_instance_bboxes = aligned_instance_bboxes[bbox_mask, :]

    torch.save((coords, colors, sem_labels, instance_labels, object_labels, aligned_coords, instance_bboxes,
                aligned_instance_bboxes), fn[:-15] + '_inst_nostuff.pth')
    print('num of points is:', coords.shape[0])
    print('num of all objects is:', np.int(np.max(object_labels) + 1))
    print('num of valid objects is:', instance_bboxes.shape[0])
    print('num of instances is:', np.int(np.max(instance_labels) + 1))
    print('Saving to ' + fn[:-15] + '_inst_nostuff.pth')

    # object_labels: contain all objects, the label are same with the object id in ScanRefer dataset
    # instance_labels: doesn't contain objects belong to wall, floor
    # instance_bboxes: doesn't contain objects belong to wall, floor, ceiling


for fn in files:
    f(fn)

# p = mp.Pool(processes=mp.cpu_count())
# if opt.data_split == 'test':
#     p.map(f_test, files)
# else:
#     p.map(f, files)
# p.close()
# p.join()
