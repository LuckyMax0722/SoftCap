import os
import sys
from easydict import EasyDict

CONF = EasyDict()

# path
CONF.PATH = EasyDict()
CONF.PATH.BASE = '/home/luk/DenseCap'  # TODO: change this
CONF.PATH.DATA = os.path.join(CONF.PATH.BASE, 'data')
CONF.PATH.SCANNET = os.path.join(CONF.PATH.DATA, "scannet")
CONF.PATH.LIB = os.path.join(CONF.PATH.BASE, "lib")
# CONF.PATH.MODELS = os.path.join(CONF.PATH.BASE, "models")
CONF.PATH.UTILS = os.path.join(CONF.PATH.BASE, "utils")

# scannet data
CONF.PATH.SCANNET_SCANS = os.path.join(CONF.PATH.SCANNET, "scans")
CONF.PATH.SCANNET_META = os.path.join(CONF.PATH.SCANNET, "meta_data")
CONF.PATH.SCANNET_DATA = os.path.join(CONF.PATH.SCANNET, "scannet_data")

# scan2cad data
CONF.PATH.SCAN2CAD = os.path.join(CONF.PATH.DATA, "scan2cad")

# no used
CONF.MULTIVIEW = os.path.join(CONF.PATH.SCANNET_DATA, "enet_feats_maxpool.hdf5")

# train
CONF.TRAIN = EasyDict()
CONF.TRAIN.MAX_DES_LEN = 30
CONF.TRAIN.MIN_IOU_THRESHOLD = 0
CONF.TRAIN.OVERLAID_THRESHOLD = 0.5

# softgroup conofig
CONF.softgroup = EasyDict()
CONF.softgroup.in_channels = 3
CONF.softgroup.channels = 32
CONF.softgroup.num_blocks = 7
CONF.softgroup.semantic_classes = 20
CONF.softgroup.instance_classes = 18
CONF.softgroup.ignore_label = -100

# voxel_cfg
CONF.voxel_cfg = EasyDict()
CONF.voxel_cfg.scale = 50
CONF.voxel_cfg.spatial_shape = [128, 512]
CONF.voxel_cfg.max_npoint = 250000
CONF.voxel_cfg.min_npoint = 5000

# grouping_cfg
CONF.grouping_cfg = EasyDict()
CONF.grouping_cfg.max_num_proposal = 128
CONF.grouping_cfg.score_thr = 0.2
CONF.grouping_cfg.radius = 0.04
CONF.grouping_cfg.mean_active = 300
CONF.grouping_cfg.npoint_thr = 0.04
CONF.grouping_cfg.ignore_classes = [0, 1]
CONF.grouping_cfg.class_numpoint_mean = [-1, -1, 3917., 12056., 2303.,
                                         8331., 3948., 3166., 5629., 11719.,
                                         1003., 3317., 4912., 10221., 3889.,
                                         4136., 2120., 945., 3967., 2589.]
# train_cfg
CONF.train_cfg = EasyDict()
CONF.train_cfg.max_proposal_num = 1024
CONF.train_cfg.pos_iou_thr = 0.5

# instance_voxel_cfg
CONF.instance_voxel_cfg = EasyDict()
CONF.instance_voxel_cfg.scale = 50
CONF.instance_voxel_cfg.spatial_shape = 20

# test_cfg
CONF.test_cfg = EasyDict()
CONF.test_cfg.min_npoint = 100
CONF.test_cfg.x4_split = False
CONF.test_cfg.cls_score_thr = 0.001
CONF.test_cfg.mask_score_thr = -0.5
CONF.test_cfg.eval_tasks = ['semantic', 'instance']

# output
CONF.PATH.OUTPUT = os.path.join(CONF.PATH.BASE, "outputs")

# graph_module
CONF.graph_module = EasyDict()
CONF.graph_module.num_graph_steps = 2  # number of layers
CONF.graph_module.num_proposals = 128  # max proposal
CONF.graph_module.num_locals = 10  # number of neighboring proposal

CONF.graph_module.query_mode = 'corner'  # ['corner', 'center']
CONF.graph_module.graph_mode = 'edge_conv'  # ['graph_conv', 'edge_conv']
CONF.graph_module.return_orientation = True
