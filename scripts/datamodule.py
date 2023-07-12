import os
import sys
import json

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from lib.dataset import ScannetReferenceDataset, ScannetReferenceTestDataset
from lib.dataset import get_scanrefer
from lib.config import CONF

sys.path.append(os.path.join(os.getcwd(), "lib"))  # HACK add the lib folder

SCAN2CAD_ROTATION = json.load(open(os.path.join(CONF.PATH.SCAN2CAD, "scannet_instance_rotations.json")))

class ScanReferDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.dataset_val = None
        self.dataset_test = None
        self.dataset_train = None
        self.all_scene_list = None
        self.test_scene_list = None
        self.Scanrefer_eval_val = None
        self.Scanrefer_eval_train = None
        self.Scanrefer_train = None

    def prepare_data(self):
        self.Scanrefer_train, self.Scanrefer_eval_train, self.Scanrefer_eval_val, self.all_scene_list = get_scanrefer(
            model='')

    def setup(self, stage: str):
        self.dataset_train = ScannetReferenceDataset(
            scanrefer=self.Scanrefer_train,
            scanrefer_all_scene=self.all_scene_list,
            split='train',
            num_points=50000,
            augment=True,
            scan2cad_rotation=SCAN2CAD_ROTATION,
        )
        self.dataset_val = ScannetReferenceDataset(
            scanrefer=self.Scanrefer_eval_val,
            scanrefer_all_scene=self.all_scene_list,
            split='val',
            num_points=40000,
            augment=False,
        )

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=4, shuffle=True, num_workers=4,
                          collate_fn=self.dataset_train.collate_fn, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=1, shuffle=False, num_workers=1,
                          collate_fn=self.dataset_val.collate_fn, drop_last=True)
