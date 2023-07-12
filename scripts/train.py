import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from datamodule import ScanReferDataModule
from capnet import CapNet

# prepare dataset and dataloader
data = ScanReferDataModule()

# create model
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

model = CapNet(val_tf_on=False, sc=False, use_relation=True, use_attention=False, use_cac=True)

# 加载softgroup_pretrain模型
file = '/home/luk/Downloads/epoch_1.pth'
model.softgroup_module.load_state_dict(torch.load(file)['net'])

# 加载训练过的模型
# file = '/home/luk/DenseCap/scripts/model0621_14:06:39_relation_cac_epoch0.pth'
# model.load_state_dict(torch.load(file), strict=True)

# 创建 ModelCheckpoint 回调
checkpoint_callback = ModelCheckpoint(
    dirpath='checkpoints',
    filename='test_checkpoint_{epoch}_{step}',
    save_last=True,  # 保存最后一个检查点
    every_n_train_steps=1000  # 每隔 1000 个迭代保存一次
)

# start training
trainer = pl.Trainer(accelerator='gpu',
                     devices=1,
                     max_epochs=-1,
                     log_every_n_steps=10,
                     num_sanity_val_steps=1,
                     callbacks=[checkpoint_callback])

# 从checkpoint恢复训练
# trainer = pl.Trainer(accelerator='gpu',
#                      devices=1,
#                      max_epochs=-1,
#                      log_every_n_steps=10,
#                      num_sanity_val_steps=1,
#                      callbacks=[checkpoint_callback],
#                      resume_from_checkpoint='/home/luk/DenseCap/scripts/checkpoints/last.ckpt')

trainer.fit(model, data)
