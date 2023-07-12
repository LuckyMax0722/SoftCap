import os
import sys
import time
import h5py
import json
import pickle
import numpy as np
import multiprocessing as mp
import torch
import scipy.interpolate
import scipy.ndimage
import math

from itertools import chain
from collections import Counter
from torch.utils.data import Dataset
from ops import voxelization_idx

sys.path.append(os.path.join(os.getcwd(), "lib"))  # HACK add the lib folder
from lib.config import CONF
from utils.pc_utils import random_sampling, rotx, roty, rotz
from utils.box_util import get_3d_box, get_3d_box_batch
from data.scannet.model_util_scannet import rotate_aligned_boxes, ScannetDatasetConfig, rotate_aligned_boxes_along_axis

from copy import deepcopy

# data setting
DC = ScannetDatasetConfig()
MAX_NUM_OBJ = 128
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])

# data path
SCANNET_V2_TSV = os.path.join(CONF.PATH.SCANNET_META, "scannetv2-labels.combined.tsv")
SCANREFER_VOCAB = os.path.join(CONF.PATH.DATA, "Scanrefer_vocabulary.json")
VOCAB = os.path.join(CONF.PATH.DATA, "{}_vocabulary.json")  # dataset_name
# SCANREFER_VOCAB_WEIGHTS = os.path.join(CONF.PATH.DATA, "ScanRefer_vocabulary_weights.json")
VOCAB_WEIGHTS = os.path.join(CONF.PATH.DATA, "{}_vocabulary_weights.json")  # dataset_name
# MULTIVIEW_DATA = os.path.join(CONF.PATH.SCANNET_DATA, "enet_feats.hdf5")
MULTIVIEW_DATA = CONF.MULTIVIEW
GLOVE_PICKLE = os.path.join(CONF.PATH.DATA, "glove.p")


def get_scanrefer(model=None):
    scanrefer_train = json.load(open(os.path.join(CONF.PATH.DATA, "scanrefer/ScanRefer_filtered_train.json")))
    scanrefer_eval_train = json.load(open(os.path.join(CONF.PATH.DATA, "scanrefer/ScanRefer_filtered_train.json")))
    scanrefer_eval_val = json.load(open(os.path.join(CONF.PATH.DATA, "scanrefer/ScanRefer_filtered_val.json")))

    SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "scanrefer/ScanRefer_filtered_train.json")))

    if model == 'debug':
        scanrefer_train = [SCANREFER_TRAIN[0]]
        scanrefer_eval_train = [SCANREFER_TRAIN[0]]
        scanrefer_eval_val = [SCANREFER_TRAIN[0]]

    # get initial scene list
    train_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_train])))  # 562 train scenes
    val_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_eval_val])))  # 141 val scenes

    # filter data in chosen scenes
    new_scanrefer_train = []
    for data in scanrefer_train:
        if data["scene_id"] in train_scene_list:
            new_scanrefer_train.append(data)

    # eval on train
    new_scanrefer_eval_train = []
    for scene_id in train_scene_list:
        data = deepcopy(SCANREFER_TRAIN[0])
        data["scene_id"] = scene_id
        new_scanrefer_eval_train.append(data)

    # eval on val
    new_scanrefer_eval_val = []
    for scene_id in val_scene_list:
        data = deepcopy(scanrefer_eval_val[0])
        data["scene_id"] = scene_id
        new_scanrefer_eval_val.append(data)

    # all scanrefer scene
    all_scene_list = train_scene_list + val_scene_list

    print("using ScanRefer dataset")
    print("train on {} samples from {} scenes".format(len(new_scanrefer_train), len(train_scene_list)))
    print("eval on {} scenes from train and {} scenes from val".format(len(new_scanrefer_eval_train),
                                                                       len(new_scanrefer_eval_val)))
    return new_scanrefer_train, new_scanrefer_eval_train, new_scanrefer_eval_val, all_scene_list


class ReferenceDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def _get_raw2label(self):
        # mapping
        scannet_labels = DC.type2class.keys()
        scannet2label = {label: i for i, label in enumerate(scannet_labels)}

        lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2label = {}
        for i in range(len(lines)):
            label_classes_set = set(scannet_labels)
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = elements[7]
            if nyu40_name not in label_classes_set:
                raw2label[raw_name] = scannet2label['others']
            else:
                raw2label[raw_name] = scannet2label[nyu40_name]

        return raw2label

    def _get_unique_multiple_lookup(self):
        all_sem_labels = {}
        cache = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            object_name = " ".join(data["object_name"].split("_"))
            ann_id = data["ann_id"]

            if scene_id not in all_sem_labels:
                all_sem_labels[scene_id] = []

            if scene_id not in cache:
                cache[scene_id] = {}

            if object_id not in cache[scene_id]:
                cache[scene_id][object_id] = {}
                try:
                    all_sem_labels[scene_id].append(self.raw2label[object_name])
                except KeyError:
                    all_sem_labels[scene_id].append(17)

        # convert to numpy array
        all_sem_labels = {scene_id: np.array(all_sem_labels[scene_id]) for scene_id in all_sem_labels.keys()}

        unique_multiple_lookup = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            object_name = " ".join(data["object_name"].split("_"))
            ann_id = data["ann_id"]

            try:
                sem_label = self.raw2label[object_name]
            except KeyError:
                sem_label = 17

            unique_multiple = 0 if (all_sem_labels[scene_id] == sem_label).sum() == 1 else 1

            # store
            if scene_id not in unique_multiple_lookup:
                unique_multiple_lookup[scene_id] = {}

            if object_id not in unique_multiple_lookup[scene_id]:
                unique_multiple_lookup[scene_id][object_id] = {}

            if ann_id not in unique_multiple_lookup[scene_id][object_id]:
                unique_multiple_lookup[scene_id][object_id][ann_id] = None

            unique_multiple_lookup[scene_id][object_id][ann_id] = unique_multiple

        return unique_multiple_lookup

    def _tranform_des(self):
        lang = {}
        label = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            ann_id = data["ann_id"]

            if scene_id not in lang:
                lang[scene_id] = {}
                label[scene_id] = {}

            if object_id not in lang[scene_id]:
                lang[scene_id][object_id] = {}
                label[scene_id][object_id] = {}

            if ann_id not in lang[scene_id][object_id]:
                lang[scene_id][object_id][ann_id] = {}
                label[scene_id][object_id][ann_id] = {}

            # trim long descriptions
            tokens = data["token"][:CONF.TRAIN.MAX_DES_LEN]

            # tokenize the description
            tokens = ["sos"] + tokens + ["eos"]
            embeddings = np.zeros((CONF.TRAIN.MAX_DES_LEN + 2, 300))
            labels = np.zeros((CONF.TRAIN.MAX_DES_LEN + 2))

            # load
            for token_id in range(len(tokens)):
                token = tokens[token_id]
                try:
                    embeddings[token_id] = self.glove[token]
                    labels[token_id] = self.vocabulary["word2idx"][token]
                except KeyError:
                    embeddings[token_id] = self.glove["unk"]
                    labels[token_id] = self.vocabulary["word2idx"]["unk"]

            # store
            lang[scene_id][object_id][ann_id] = embeddings
            label[scene_id][object_id][ann_id] = labels

        return lang, label

    def _build_vocabulary(self, dataset_name):
        vocab_path = VOCAB.format(dataset_name)
        if os.path.exists(vocab_path):
            self.vocabulary = json.load(open(vocab_path))
        else:
            if self.split == "train":
                all_words = chain(*[data["token"][:CONF.TRAIN.MAX_DES_LEN] for data in self.scanrefer])
                word_counter = Counter(all_words)
                word_counter = sorted([(k, v) for k, v in word_counter.items() if k in self.glove], key=lambda x: x[1],
                                      reverse=True)
                word_list = [k for k, _ in word_counter]

                # build vocabulary
                word2idx, idx2word = {}, {}
                spw = ["pad_", "unk", "sos", "eos"]  # NOTE distinguish padding token "pad_" and the actual word "pad"
                for i, w in enumerate(word_list):
                    shifted_i = i + len(spw)
                    word2idx[w] = shifted_i
                    idx2word[shifted_i] = w

                # add special words into vocabulary
                for i, w in enumerate(spw):
                    word2idx[w] = i
                    idx2word[i] = w

                vocab = {
                    "word2idx": word2idx,
                    "idx2word": idx2word
                }
                json.dump(vocab, open(vocab_path, "w"), indent=4)

                self.vocabulary = vocab

    def _build_frequency(self, dataset_name):
        vocab_weights_path = VOCAB_WEIGHTS.format(dataset_name)
        if os.path.exists(vocab_weights_path):
            with open(vocab_weights_path) as f:
                weights = json.load(f)
                self.weights = np.array([v for _, v in weights.items()])
        else:
            all_tokens = []
            for scene_id in self.lang_ids.keys():
                for object_id in self.lang_ids[scene_id].keys():
                    for ann_id in self.lang_ids[scene_id][object_id].keys():
                        all_tokens += self.lang_ids[scene_id][object_id][ann_id].astype(int).tolist()

            word_count = Counter(all_tokens)
            word_count = sorted([(k, v) for k, v in word_count.items()], key=lambda x: x[0])

            # frequencies = [c for _, c in word_count]
            # weights = np.array(frequencies).astype(float)
            # weights = weights / np.sum(weights)
            # weights = 1 / np.log(1.05 + weights)

            weights = np.ones((len(word_count)))

            self.weights = weights

            with open(vocab_weights_path, "w") as f:
                weights = {k: v for k, v in enumerate(weights)}
                json.dump(weights, f, indent=4)

    def _load_data(self, dataset_name):
        print("loading data...")
        # load language features
        self.glove = pickle.load(open(GLOVE_PICKLE, "rb"))
        self._build_vocabulary(dataset_name)
        self.num_vocabs = len(self.vocabulary["word2idx"].keys())
        self.lang, self.lang_ids = self._tranform_des()
        self._build_frequency(dataset_name)

        # add scannet data
        self.scene_list = sorted(list(set([data["scene_id"] for data in self.scanrefer])))

        # load scene data
        self.scene_data = {}
        for scene_id in self.scene_list:
            self.scene_data[scene_id] = {}
            filename = os.path.join(CONF.PATH.SCANNET, self.split, scene_id + '_inst_nostuff.pth')
            coords, colors, sem_labels, instance_labels, object_labels, aligned_coords, instance_bboxes, aligned_instance_bboxes = torch.load(
                filename)
            self.scene_data[scene_id]['coords'] = coords
            self.scene_data[scene_id]['colors'] = colors
            self.scene_data[scene_id]['sem_labels'] = sem_labels
            self.scene_data[scene_id]['instance_labels'] = instance_labels
            self.scene_data[scene_id]['object_labels'] = object_labels
            self.scene_data[scene_id]['aligned_coords'] = aligned_coords
            self.scene_data[scene_id]['instance_bboxes'] = instance_bboxes
            self.scene_data[scene_id]['aligned_instance_bboxes'] = aligned_instance_bboxes

        # prepare class mapping
        lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2nyuid = {}
        for i in range(len(lines)):
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = int(elements[4])
            raw2nyuid[raw_name] = nyu40_name

        # store
        self.raw2nyuid = raw2nyuid
        self.raw2label = self._get_raw2label()
        self.unique_multiple_lookup = self._get_unique_multiple_lookup()

    # 对整体点云和bounding box进行平移
    def _translate(self, point_set, bbox):
        # unpack
        coords = point_set[:, :3]

        # translation factors
        x_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        y_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        z_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        factor = [x_factor, y_factor, z_factor]

        # dump
        coords += factor
        point_set[:, :3] = coords
        bbox[:, :3] += factor

        return point_set, bbox


class ScannetReferenceDataset(ReferenceDataset):

    def __init__(self, scanrefer, scanrefer_all_scene,
                 split="train",
                 num_points=40000,
                 augment=False,
                 voxel_cfg=CONF.voxel_cfg,
                 scan2cad_rotation=json.load(open(os.path.join(CONF.PATH.SCAN2CAD, "scannet_instance_rotations.json")))):

        # NOTE only feed the scan2cad_rotation when on the training mode and train split

        self.scanrefer = scanrefer
        self.scanrefer_all_scene = scanrefer_all_scene  # all scene_ids in scanrefer
        self.split = split
        self.num_points = num_points
        self.augment = augment
        self.voxel_cfg = voxel_cfg
        self.scan2cad_rotation = scan2cad_rotation
        # load data
        self._load_data('Scanrefer')

    def __len__(self):
        return len(self.scanrefer)

    def elastic(self, x, gran, mag):
        blur0 = np.ones((3, 1, 1)).astype('float32') / 3
        blur1 = np.ones((1, 3, 1)).astype('float32') / 3
        blur2 = np.ones((1, 1, 3)).astype('float32') / 3

        bb = np.abs(x).max(0).astype(np.int32) // gran + 3
        noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        ax = [np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in bb]
        interp = [
            scipy.interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0)
            for n in noise
        ]

        def g(x_):
            return np.hstack([i(x_)[:, None] for i in interp])

        return x + g(x) * mag

    def dataAugment(self, xyz, jitter=False, flip=False, rot=False, scale=False, prob=1.0):
        m = np.eye(3)
        if jitter and np.random.rand() < prob:
            m += np.random.randn(3, 3) * 0.1
        if flip and np.random.rand() < prob:
            m[0][0] *= np.random.randint(0, 2) * 2 - 1
        if rot and np.random.rand() < prob:
            theta = np.random.rand() * 0.2 * math.pi
            m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0],
                              [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])

        else:
            if self.augment:
                # Empirically, slightly rotate the scene can match the results from checkpoint
                theta = 0.35 * math.pi
                m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0],
                                  [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
        if scale and np.random.rand() < prob:
            scale_factor = np.random.uniform(0.95, 1.05)
            xyz = xyz * scale_factor
        return np.matmul(xyz, m)

    def crop(self, xyz, step=32):
        xyz_offset = xyz.copy()
        valid_idxs = xyz_offset.min(1) >= 0
        assert valid_idxs.sum() == xyz.shape[0]
        spatial_shape = np.array([self.voxel_cfg.spatial_shape[1]] * 3)
        room_range = xyz.max(0) - xyz.min(0)
        while (valid_idxs.sum() > self.voxel_cfg.max_npoint):
            step_temp = step
            if valid_idxs.sum() > 1e6:
                step_temp = step * 2
            offset = np.clip(spatial_shape - room_range + 0.001, None, 0) * np.random.rand(3)
            xyz_offset = xyz + offset
            valid_idxs = (xyz_offset.min(1) >= 0) * ((xyz_offset < spatial_shape).sum(1) == 3)
            spatial_shape[:2] -= step_temp
        return xyz_offset, valid_idxs

    def getCroppedInstLabel(self, instance_label, valid_idxs):
        instance_label = instance_label[valid_idxs]
        j = 0
        while (j < instance_label.max()):
            if (len(np.where(instance_label == j)[0]) == 0):
                instance_label[instance_label == instance_label.max()] = j
            j += 1
        return instance_label

    def transform_train(self, xyz, rgb, semantic_label, instance_label, object_id_labels, aug_prob=1.0):
        if self.augment == True:
            xyz_middle = self.dataAugment(xyz, True, True, True, True, aug_prob)
        else:
            xyz_middle = xyz
        xyz = xyz_middle * self.voxel_cfg.scale
        if np.random.rand() < aug_prob:
            xyz = self.elastic(xyz, 6, 40.)
            xyz = self.elastic(xyz, 20, 160.)
        xyz = xyz - xyz.min(0)
        max_tries = 5
        while (max_tries > 0):
            xyz_offset, valid_idxs = self.crop(xyz)
            if valid_idxs.sum() >= self.voxel_cfg.min_npoint:
                xyz = xyz_offset
                break
            max_tries -= 1
        if valid_idxs.sum() < self.voxel_cfg.min_npoint:
            return None
        xyz = xyz[valid_idxs]
        xyz_middle = xyz_middle[valid_idxs]
        rgb = rgb[valid_idxs]
        semantic_label = semantic_label[valid_idxs]
        object_id_labels = object_id_labels[valid_idxs]
        instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)
        return xyz, xyz_middle, rgb, semantic_label, instance_label, object_id_labels

    def transform_test(self, xyz, rgb, semantic_label, instance_label, object_id_labels):
        xyz_middle = self.dataAugment(xyz, False, False, False, False)
        xyz = xyz_middle * self.voxel_cfg.scale
        xyz -= xyz.min(0)
        valid_idxs = np.ones(xyz.shape[0], dtype=bool)
        instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)
        return xyz, xyz_middle, rgb, semantic_label, instance_label, object_id_labels

    def getInstanceInfo(self, xyz, instance_label, semantic_label):
        pt_mean = np.ones((xyz.shape[0], 3), dtype=np.float32) * -100.0
        instance_pointnum = []
        instance_cls = []
        # max(instance_num, 0) to support instance_label with no valid instance_id
        instance_num = max(int(instance_label.max()) + 1, 0)
        for i_ in range(instance_num):
            inst_idx_i = np.where(instance_label == i_)
            xyz_i = xyz[inst_idx_i]
            pt_mean[inst_idx_i] = xyz_i.mean(0)
            instance_pointnum.append(inst_idx_i[0].size)
            cls_idx = inst_idx_i[0][0]
            instance_cls.append(semantic_label[cls_idx])
        pt_offset_label = pt_mean - xyz
        return instance_num, instance_pointnum, instance_cls, pt_offset_label

    def __getitem__(self, idx):
        start = time.time()
        scene_id = self.scanrefer[idx]["scene_id"]
        object_id = int(self.scanrefer[idx]["object_id"])
        object_name = " ".join(self.scanrefer[idx]["object_name"].split("_"))
        ann_id = self.scanrefer[idx]["ann_id"]

        # get language features
        lang_feat = self.lang[scene_id][str(object_id)][ann_id]
        lang_len = len(self.scanrefer[idx]["token"]) + 2
        lang_len = lang_len if lang_len <= CONF.TRAIN.MAX_DES_LEN + 2 else CONF.TRAIN.MAX_DES_LEN + 2
        ann_list = list(self.lang_ids[scene_id][str(object_id)].keys())
        lang_id = []
        for i in ann_list:
            lang_id.append(self.lang_ids[scene_id][str(object_id)][i].astype(np.int64))

        coords = self.scene_data[scene_id]['coords']
        # coords = self.scene_data[scene_id]['aligned_coords']
        instance_bboxes = self.scene_data[scene_id]['instance_bboxes']
        # instance_bboxes = self.scene_data[scene_id]['aligned_instance_bboxes']

        colors = self.scene_data[scene_id]['colors']
        semantic_labels = self.scene_data[scene_id]['sem_labels']
        instance_labels = self.scene_data[scene_id]['instance_labels']
        object_labels = self.scene_data[scene_id]['object_labels']

        if coords.shape[0] > self.voxel_cfg.max_npoint and self.split == 'train':
            choices = np.random.choice(coords.shape[0], self.voxel_cfg.max_npoint, replace=False)
            coords = coords[choices, :]
            colors = colors[choices, :]
            semantic_labels = semantic_labels[choices]
            instance_labels = instance_labels[choices]
            object_labels = object_labels[choices]

        original_points = coords.copy()

        # --------------------------- FEAT used for SOFTGROUP -----------------------------
        if self.augment:
            data = self.transform_train(coords, colors, semantic_labels, instance_labels,
                                        object_labels, 1)
        else:
            data = self.transform_test(coords, colors, semantic_labels, instance_labels,
                                       object_labels)
        xyz, xyz_middle, rgb, semantic_labels, instance_labels, object_labels = data
        point_cloud = np.concatenate((xyz_middle, rgb), axis=1)

        info = self.getInstanceInfo(xyz_middle, instance_labels, semantic_labels)
        inst_num, inst_pointnum, inst_cls, pt_offset_label = info
        inst_cls = [x - 2 if x != -100 else x for x in inst_cls]

        if np.where(object_labels == object_id)[0].size != 0:
            instance_id = instance_labels[np.where(object_labels == object_id)[0][0]]
        else:
            instance_id = -100

        # ------------------------------- LABELS ------------------------------
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))
        num_bbox = 0

        ref_box_label = np.zeros(MAX_NUM_OBJ)  # bbox label for reference target
        ref_center_label = np.zeros(3)  # bbox center for reference target
        ref_size_class_label = 0
        ref_size_residual_label = np.zeros(3)  # bbox size residual for reference target
        ref_box_corner_label = np.zeros((8, 3))
        ref_size_label = np.zeros(3)

        num_bbox = instance_bboxes.shape[0] if instance_bboxes.shape[0] < MAX_NUM_OBJ else MAX_NUM_OBJ
        target_bboxes_mask[0:num_bbox] = 1
        target_bboxes[0:num_bbox, :] = instance_bboxes[:MAX_NUM_OBJ, 0:6]

        # NOTE: set size class as semantic class. Consider use size2class.
        # class_ind 为 semantic class（0-19）
        class_ind = instance_bboxes[:num_bbox, -2].astype(np.int64)
        size_classes[0:num_bbox] = class_ind
        size_residuals[0:num_bbox, :] = target_bboxes[0:num_bbox, 3:6] - DC.mean_size_arr[class_ind - 2,
                                                                         :]

        # construct the reference target label for each bbox
        for i, gt_id in enumerate(instance_bboxes[:num_bbox, -1]):
            if gt_id == object_id:
                ref_box_label[i] = 1
                ref_center_label = target_bboxes[i, 0:3]
                ref_size_label = target_bboxes[i, 3:6]
                ref_size_class_label = size_classes[i]
                ref_size_residual_label = size_residuals[i]

                # construct ground truth box corner coordinates
                ref_obb = DC.param2obb(ref_center_label, ref_size_class_label, ref_size_residual_label)
                ref_box_corner_label = get_3d_box(ref_obb[3:6], 0, ref_obb[0:3])

        # construct all GT bbox corners
        all_obb = DC.param2obb_batch(target_bboxes[:num_bbox, 0:3], size_classes[:num_bbox].astype(np.int64),
                                     size_residuals[:num_bbox])
        all_box_corner_label = get_3d_box_batch(all_obb[:, 3:6], np.zeros(num_bbox), all_obb[:, 0:3])

        # store
        gt_box_corner_label = np.zeros((MAX_NUM_OBJ, 8, 3))
        gt_box_masks = np.zeros((MAX_NUM_OBJ,))
        gt_box_object_ids = np.zeros((MAX_NUM_OBJ,))

        gt_box_corner_label[:num_bbox] = all_box_corner_label
        gt_box_masks[:num_bbox] = 1
        gt_box_object_ids[:num_bbox] = instance_bboxes[:, -1]

        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        target_object_ids = np.zeros((MAX_NUM_OBJ,))  # object ids of all objects
        try:
            target_bboxes_semcls[0:num_bbox] = instance_bboxes[:, -2][0:num_bbox]
            target_object_ids[0:num_bbox] = instance_bboxes[:, -1][0:num_bbox]
        except KeyError:
            pass

        object_cat = self.raw2label[object_name] if object_name in self.raw2label else 17

        num_points = torch.tensor(xyz_middle.shape[0])

        # object rotations
        scene_object_rotations = np.zeros((MAX_NUM_OBJ, 3, 3))
        scene_object_rotation_masks = np.zeros((MAX_NUM_OBJ,))  # NOTE this is not object mask!!!
        # if scene is not in scan2cad annotations, skip
        # if the instance is not in scan2cad annotations, skip
        if self.scan2cad_rotation and scene_id in self.scan2cad_rotation:
            for i, gt_id in enumerate(instance_bboxes[:num_bbox, -1].astype(int)):
                try:
                    rotation = np.array(self.scan2cad_rotation[scene_id][str(gt_id)])

                    scene_object_rotations[i] = rotation
                    scene_object_rotation_masks[i] = 1
                except KeyError:
                    pass

        data_dict = {}
        # dataset
        # ----------------------------------------------------------------------
        data_dict["dataset_idx"] = np.array(idx).astype(np.int64)
        data_dict["num_points"] = num_points

        # softgroup
        # ----------------------------------------------------------------------
        data_dict["scan_id"] = scene_id
        data_dict["coord"] = torch.from_numpy(xyz).long()
        data_dict["coord_float"] = torch.from_numpy(xyz_middle)
        data_dict["feat"] = torch.from_numpy(rgb).float()  # rgb color
        data_dict["semantic_label"] = torch.from_numpy(semantic_labels)
        data_dict["instance_label"] = torch.from_numpy(np.array(instance_labels).astype(np.int64))
        data_dict["object_label"] = torch.from_numpy(np.array(object_labels).astype(np.int64))
        data_dict["inst_num"] = np.array(inst_num).astype(np.int64)
        data_dict["inst_pointnum"] = np.array(inst_pointnum).astype(np.int64)
        data_dict["inst_cls"] = np.array(inst_cls).astype(np.int64)
        data_dict["pt_offset_label"] = torch.from_numpy(pt_offset_label)

        # point-cloud data
        # ----------------------------------------------------------------------
        data_dict['original_point'] = original_points.astype(np.float32)
        data_dict["point_clouds"] = point_cloud.astype(np.float32)  # point cloud data including features
        data_dict["pcl_color"] = colors
        data_dict["object_id"] = torch.from_numpy(np.array(int(object_id)).astype(np.int64))
        data_dict["instance_id"] = torch.from_numpy(np.array(int(instance_id)).astype(np.int64))
        data_dict["object_cat"] = np.array(object_cat).astype(np.int64)
        data_dict["ann_id"] = np.array(int(ann_id)).astype(np.int64)

        # language description
        # ----------------------------------------------------------------------
        data_dict["lang_feat"] = torch.from_numpy(lang_feat.astype(np.float32))  # language feature vectors
        data_dict["lang_len"] = torch.from_numpy(np.array(lang_len).astype(np.int64))  # length of each description
        data_dict["lang_ids"] = lang_id
        data_dict["lang_ids_tensor"] = torch.from_numpy(
            np.array(self.lang_ids[scene_id][str(object_id)][ann_id]).astype(np.int64))

        # GT bounding box
        # ----------------------------------------------------------------------
        data_dict["num_bbox"] = np.array(num_bbox).astype(np.int64)
        data_dict["box_label_mask"] = target_bboxes_mask.astype(
            np.float32)  # (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
        data_dict["center_label"] = torch.from_numpy(target_bboxes.astype(np.float32)[:,
                                                     0:3])  # (MAX_NUM_OBJ, 3) for GT box center XYZ
        data_dict["size_class_label"] = size_classes.astype(
            np.int64)  # (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
        data_dict["size_residual_label"] = size_residuals.astype(np.float32)  # (MAX_NUM_OBJ, 3)

        # GT bounding box corner
        # ----------------------------------------------------------------------
        data_dict["gt_box_corner_label"] = torch.from_numpy(gt_box_corner_label.astype(
            np.float64))  # (MAX_NUM_OBJ，8，3)
        data_dict["gt_box_masks"] = torch.from_numpy(gt_box_masks.astype(np.int64))  # (MAX_NUM_OBJ)
        data_dict["gt_box_object_ids"] = gt_box_object_ids.astype(np.int64)

        # ref bounding box
        # ----------------------------------------------------------------------
        data_dict["ref_box_label"] = ref_box_label.astype(np.int64)  # 0/1 reference labels for each object bbox
        data_dict["ref_center_label"] = torch.from_numpy(ref_center_label.astype(np.float32))
        data_dict["ref_size_class_label"] = np.array(int(ref_size_class_label)).astype(
            np.int64)
        data_dict["ref_size_residual_label"] = ref_size_residual_label.astype(np.float32)
        data_dict["ref_box_corner_label"] = ref_box_corner_label.astype(
            np.float64)  # target box corners NOTE type must be double
        data_dict['ref_size_label'] = torch.from_numpy(ref_size_label.astype(np.float32))

        # target
        # ----------------------------------------------------------------------
        data_dict["sem_cls_label"] = target_bboxes_semcls.astype(np.int64)  # (MAX_NUM_OBJ,)
        data_dict["scene_object_ids"] = torch.from_numpy(
            target_object_ids.astype(np.int64))  # (MAX_NUM_OBJ,)

        # unique_multiple
        data_dict["unique_multiple"] = np.array(self.unique_multiple_lookup[scene_id][str(object_id)][ann_id]).astype(
            np.int64)

        # object rotation
        data_dict["scene_object_rotations"] = torch.from_numpy(
            scene_object_rotations.astype(np.float32))  # (MAX_NUM_OBJ, 3, 3)
        data_dict["scene_object_rotation_masks"] = torch.from_numpy(
            scene_object_rotation_masks.astype(np.int64))  # (MAX_NUM_OBJ)

        data_dict["load_time"] = time.time() - start

        return data_dict

    def collate_fn(self, batch):
        scan_ids = []
        coords = []
        coords_float = []
        feats = []
        semantic_labels = []
        instance_labels = []
        object_labels = []
        instance_ids = []
        instance_pointnum = []  # (total_nInst), int
        instance_cls = []  # (total_nInst), long
        pt_offset_labels = []
        inst_nums = []
        total_inst_num = 0
        batch_id = 0
        num_points = []
        lang_ids = []

        for data in batch:
            if data is None:
                continue
            # get data from dataset
            scan_id = data["scan_id"]
            coord = data["coord"]
            coord_float = data["coord_float"]
            feat = data["feat"]
            semantic_label = data["semantic_label"]
            instance_label = data["instance_label"]
            object_label = data["object_label"]
            inst_num = data["inst_num"]
            inst_pointnum = data["inst_pointnum"]
            inst_cls = data["inst_cls"]
            pt_offset_label = data["pt_offset_label"]
            instance_id = data['instance_id']
            num_point = data["num_points"]
            lang_id = data['lang_ids']

            # append
            instance_label[np.where(instance_label != -100)] += total_inst_num
            instance_id += total_inst_num

            total_inst_num += inst_num
            scan_ids.append(scan_id)
            coords.append(torch.cat([coord.new_full((coord.size(0), 1), batch_id), coord], 1))
            coords_float.append(coord_float)
            feats.append(feat)
            semantic_labels.append(semantic_label)
            instance_labels.append(instance_label)
            object_labels.append(object_label)
            instance_ids.append(instance_id)
            instance_pointnum.extend(inst_pointnum)
            instance_cls.extend(inst_cls)
            pt_offset_labels.append(pt_offset_label)
            inst_nums.append(inst_num)
            num_points.append(num_point)
            lang_ids.append(lang_id)
            batch_id += 1

        assert batch_id > 0, 'empty batch'
        if batch_id < len(batch):
            print(f'batch is truncated from size {len(batch)} to {batch_id}')

        # merge all the scenes in the batch
        coords = torch.cat(coords, 0)  # long (N, 1 + 3), the batch item idx is put in coords[:, 0]
        batch_idxs = coords[:, 0].int()
        coords_float = torch.cat(coords_float, 0).to(torch.float32)  # float (N, 3)
        feats = torch.cat(feats, 0)  # float (N, C)
        semantic_labels = torch.cat(semantic_labels, 0).long()  # long (N)
        instance_labels = torch.cat(instance_labels, 0).long()  # long (N)
        object_labels = torch.cat(object_labels, 0).long()  # long (B)
        instance_ids = torch.tensor(instance_ids)  # long (B)
        inst_nums = torch.tensor(np.asarray(inst_nums), dtype=torch.int)
        num_points = torch.tensor(np.asarray(num_points), dtype=torch.int)
        instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int)  # int (total_nInst)
        instance_cls = torch.tensor(instance_cls, dtype=torch.long)  # long (total_nInst)
        pt_offset_labels = torch.cat(pt_offset_labels).float()

        object_id = torch.cat([batch[i]['object_id'].unsqueeze(0) for i in range(len(batch))], 0)
        lang_feat = torch.cat([batch[i]['lang_feat'].unsqueeze(0) for i in range(len(batch))], 0)
        lang_len = torch.cat([batch[i]['lang_len'].unsqueeze(0) for i in range(len(batch))], 0)
        lang_ids_tensor = torch.cat([batch[i]['lang_ids_tensor'].unsqueeze(0) for i in range(len(batch))], 0)
        ref_size_label = torch.cat([batch[i]['ref_size_label'].unsqueeze(0) for i in range(len(batch))], 0)
        ref_center_label = torch.cat([batch[i]['ref_center_label'].unsqueeze(0) for i in range(len(batch))], 0)
        center_label = torch.cat([batch[i]['center_label'].unsqueeze(0) for i in range(len(batch))], 0)
        scene_object_ids = torch.cat([batch[i]['scene_object_ids'].unsqueeze(0) for i in range(len(batch))], 0)
        gt_box_corner_label = torch.cat([batch[i]['gt_box_corner_label'].unsqueeze(0) for i in range(len(batch))], 0)
        gt_box_masks = torch.cat([batch[i]['gt_box_masks'].unsqueeze(0) for i in range(len(batch))], 0)
        scene_object_rotations = torch.cat([batch[i]['scene_object_rotations'].unsqueeze(0) for i in range(len(batch))],
                                           0)
        scene_object_rotation_masks = torch.cat(
            [batch[i]['scene_object_rotation_masks'].unsqueeze(0) for i in range(len(batch))], 0)

        spatial_shape = np.clip(coords.max(0)[0][1:].numpy() + 1, self.voxel_cfg.spatial_shape[0], None)

        voxel_coords, v2p_map, p2v_map = voxelization_idx(coords, batch_id)

        return {
            # softgroup need
            'scan_ids': scan_ids,
            'coords': coords,
            'batch_idxs': batch_idxs,
            'voxel_coords': voxel_coords,
            'p2v_map': p2v_map,
            'v2p_map': v2p_map,
            'coords_float': coords_float,
            'feats': feats,
            'semantic_labels': semantic_labels,
            'instance_labels': instance_labels,
            'object_labels': object_labels,
            'instance_pointnum': instance_pointnum,
            'instance_cls': instance_cls,
            'pt_offset_labels': pt_offset_labels,
            'spatial_shape': spatial_shape,
            'inst_nums': inst_nums,
            'batch_size': batch_id,
            'num_points': num_points,

            # proposal module need
            'object_id': object_id,
            'instance_id': instance_ids,
            'ref_size_label': ref_size_label,
            'ref_center_label': ref_center_label,
            'center_label': center_label,
            'gt_box_corner_label': gt_box_corner_label,
            'scene_object_ids': scene_object_ids,
            'gt_box_masks': gt_box_masks,

            # caption module need
            'lang_feat': lang_feat,
            'lang_len': lang_len,
            'lang_ids': lang_ids,
            'lang_ids_tensor': lang_ids_tensor,

            # object rotation:
            'scene_object_rotations': scene_object_rotations,
            'scene_object_rotation_masks': scene_object_rotation_masks,

            # visualization need
            'original_point': batch[0]['original_point']

        }