import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from datamodule import ScanReferDataModule
from capnet import CapNet
from lib.config import CONF

# prepare dataset and dataloader
data = ScanReferDataModule()

model = CapNet(CONF.model_setting.val_tf_on, CONF.model_setting.sc, CONF.model_setting.use_relation,
               CONF.model_setting.use_attention, CONF.model_setting.use_cac)

# Load SoftGroup_pretrain model
file = CONF.PATH.PRETRAIN
model.softgroup_module.load_state_dict(torch.load(file)['net'])

# Load pretrained model
# file = '/home/luk/DenseCap/scripts/...pth'
# model.load_state_dict(torch.load(file), strict=True)

# ModelCheckpoint
checkpoint_callback = ModelCheckpoint(
    dirpath='checkpoints',
    filename='test_checkpoint_{epoch}_{step}',
    save_last=True,  # save last ckpt
    every_n_train_steps=1000  # save each 1000 iterations
)

# start training
trainer = pl.Trainer(accelerator='gpu',
                     devices=1,
                     max_epochs=-1,
                     log_every_n_steps=10,
                     num_sanity_val_steps=1,
                     callbacks=[checkpoint_callback])

# resume from ckpt
# trainer = pl.Trainer(accelerator='gpu',
#                      devices=1,
#                      max_epochs=-1,
#                      log_every_n_steps=10,
#                      num_sanity_val_steps=1,
#                      callbacks=[checkpoint_callback],
#                      resume_from_checkpoint='/home/luk/DenseCap/scripts/checkpoints/last.ckpt')

trainer.fit(model, data)
