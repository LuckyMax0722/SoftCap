import torch
import pytorch_lightning as pl
import os.path as osp
import os
import multiprocessing as mp
import numpy as np

from datamodule import ScanReferDataModule
from capnet import CapNet

# prepare dataset and dataloader
data = ScanReferDataModule()
data.prepare_data()
data.setup(stage='test')
test_dataloader = data.test_dataloader()

# from checkpoint load model
checkpoint_path = "/home/luk/DenseCap/scripts/model_checkpoint_epoch1.ckpt" # TODO：改路径
model = CapNet.load_from_checkpoint(checkpoint_path)

# create list
scan_ids, coords, colors, sem_preds, sem_labels = [], [], [], [], []
offset_preds, offset_labels, inst_labels, pred_insts, gt_insts = [], [], [], [], []
bbox_preds, bbox_corners, bbox_corners_all = [], [], []
panoptic_preds = []

# generate results
trainer = pl.Trainer(
    accelerator='gpu',
    devices=1,
)
predictions = trainer.predict(model, dataloaders=test_dataloader)


# save results as .npy file
def save_npy(root, name, scan_ids, arrs):
    root = osp.join(root, name)
    os.makedirs(root, exist_ok=True)
    paths = [osp.join(root, f'{i}.npy') for i in scan_ids]
    pool = mp.Pool()
    pool.starmap(np.save, zip(paths, arrs))
    pool.close()
    pool.join()


for pre in predictions:
    scan_ids.append(pre['scan_id'])
    sem_labels.append(pre['semantic_labels'])
    inst_labels.append(pre['instance_labels'])

    # semantic
    coords.append(pre['coords_float'])
    colors.append(pre['color_feats'])
    sem_preds.append(pre['semantic_preds'])
    offset_preds.append(pre['offset_preds'])
    offset_labels.append(pre['offset_labels'])

    # instance
    pred_insts.append(pre['pred_instances'])
    gt_insts.append(pre['gt_instances'])

    # bbox
    bbox_preds.append(pre["pred_bboxes"])


def get_bbox_corners(bbox_center_length):
    x, y, z = bbox_center_length[0:3]
    l, w, h = bbox_center_length[3:6]

    x_corners = torch.tensor([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2])
    y_corners = torch.tensor([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2])
    z_corners = torch.tensor([h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2])

    x = x.expand(1, 8)
    y = y.expand(1, 8)
    z = z.expand(1, 8)

    x = x + x_corners
    y = y + y_corners
    z = z + z_corners

    bbox_eight_xyz = torch.cat((x, y, z), dim=0)
    bbox_eight_xyz = bbox_eight_xyz.T

    return bbox_eight_xyz


for i, bbox in enumerate(bbox_preds):
    bbox_corners = torch.tensor((8, 3))
    for j, bbox_center_length in enumerate(bbox):
        bbox_eight_xyz = get_bbox_corners(bbox_center_length)
        if j == 0:
            bbox_corners = bbox_eight_xyz.clone()
        else:
            bbox_corners = torch.cat((bbox_corners, bbox_eight_xyz), dim=0)
    bbox_corners_all.append(bbox_corners)

# config for generate .npy file
out = '/home/luk/DenseCap/visualization'

# generate .npy
# semantic
save_npy(out, 'coords', scan_ids, coords)
save_npy(out, 'colors', scan_ids, colors)
save_npy(out, 'semantic_pred', scan_ids, sem_preds)
save_npy(out, 'semantic_label', scan_ids, sem_labels)
save_npy(out, 'offset_pred', scan_ids, offset_preds)
save_npy(out, 'offset_label', scan_ids, offset_labels)
save_npy(out, 'bbox_corners_all', scan_ids, bbox_corners_all)


# instance
'''
nyu_id = data.NYU_ID
save_pred_instances(out, 'pred_instance', scan_ids, pred_insts, nyu_id)
save_gt_instances(out, 'gt_instance', scan_ids, gt_insts, nyu_id)
'''
