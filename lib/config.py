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
CONF.PATH.PRETRAIN = '/home/luk/DenseCap/softgroup.pth'  # TODO: change this
CONF.PATH.EVAL = '/home/luk/DenseCap/eval_model.pth'  # TODO: change this
CONF.PATH.OUTPUT = os.path.join(CONF.PATH.BASE, "outputs")

# model setting
'''
STAGE 1:
pretrain the SoftGroup model (already done)

STAGE 2:
SoftGroup + GRU
use_relation = False, use_attention = False, use_cac = False

SoftGroup + relation + GRU
use_relation = True, use_attention = False, use_cac = False

SoftGroup + relation + Att2GRU
use_relation = True, use_attention = True, use_cac = False

SoftGroup + CAC
use_relation = False, use_attention = False, use_cac = True

SoftGroup + relation + CAC
use_relation = True, use_attention = False, use_cac = True

STAGE 3: 
when using reinforcement learning, setting use_sc = True, others remain the same
and use the pretrained model you got in STAGE 2 with the same structure
'''
CONF.model_setting = EasyDict()
CONF.model_setting.val_tf_on = False
CONF.model_setting.sc = False
CONF.model_setting.use_relation = True
CONF.model_setting.use_attention = False
CONF.model_setting.use_cac = True

# vis setting
CONF.vis_setting = EasyDict()
CONF.vis_setting.eval_detection = True  # eval detection
CONF.vis_setting.eval_caption = True  # eval caption
CONF.vis_setting.visualization = True  # generate vis file
CONF.vis_setting.min_iou = 0.5  # set IoU

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

# softgroup module config
CONF.softgroup = EasyDict()
CONF.softgroup.in_channels = 3
CONF.softgroup.channels = 32
CONF.softgroup.num_blocks = 7
CONF.softgroup.semantic_classes = 20
CONF.softgroup.instance_classes = 18
CONF.softgroup.ignore_label = -100
CONF.softgroup.fixed_modules = ['input_conv', 'unet', 'output_layer']

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

# graph_module
CONF.graph_module = EasyDict()
CONF.graph_module.in_size = 32
CONF.graph_module.out_size = 32
CONF.graph_module.num_graph_steps = 2  # number of layers
CONF.graph_module.num_proposals = 128  # max proposal
CONF.graph_module.feat_size = 32
CONF.graph_module.num_locals = 10  # number of neighboring proposal
CONF.graph_module.query_mode = 'corner'  # ['corner', 'center']
CONF.graph_module.graph_mode = 'edge_conv'  # ['graph_conv', 'edge_conv']
CONF.graph_module.return_orientation = True

# attention_module
CONF.attention_module = EasyDict()
CONF.attention_module.in_size = 32
CONF.attention_module.out_size = 1
CONF.attention_module.hidden_size = 128

# caption module
CONF.caption_module = EasyDict()
CONF.caption_module.emb_size = 300
CONF.caption_module.feat_size = 32
CONF.caption_module.hidden_size = 300

# test_cfg
CONF.test_cfg = EasyDict()
CONF.test_cfg.min_npoint = 100
CONF.test_cfg.x4_split = False
CONF.test_cfg.cls_score_thr = 0.001
CONF.test_cfg.mask_score_thr = -0.5
CONF.test_cfg.eval_tasks = ['semantic', 'instance']
