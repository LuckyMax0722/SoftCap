import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from datamodule import ScanReferDataModule
from capnet import CapNet

# create model
model1 = CapNet(val_tf_on=False)
model2 = CapNet(val_tf_on=False)

file1 = '/home/luk/DenseCap/scripts/test_model.pth'
checkpoint1 = torch.load(file1)
model1.load_state_dict(checkpoint1, strict=True)

# 加载pretrain模型并冻结
file2 = '/home/luk/Downloads/epoch_1.pth'
checkpoint2 = torch.load(file2)
model2.softgroup_module.load_state_dict(checkpoint2['net'], strict=True)

# for name,params in model1.softgroup_module.iou_score_linear.named_parameters():
#     print(name)
#     print(params)
#
# for name,params in model2.softgroup_module.iou_score_linear.named_parameters():
#     print(name)
#     print(params)

# for (name, params) in checkpoint1:
#     print(name)

print(len(checkpoint2['net'].keys()))
print(type(checkpoint2['net']))

print(len(checkpoint1.keys()))
print(type(checkpoint1))

lb1 = []
lb2 = []
for (key, params) in checkpoint1.items():
    lb1.append(params)

for (key, params) in checkpoint2['net'].items():
    lb2.append(params)


# print(lb1)
# print(lb2)


# for i in model2.softgroup_module.output_layer.modules():
#     if isinstance(i, torch.nn.Sequential):
#         for sub_name, sub_module in i.named_modules():
#             print(sub_name, sub_module)
#     break

for i in range(512):
    print(list(checkpoint2['net'].keys())[i])
    print(lb1[i]==lb2[i])
#     # print(lb1[i])
#     # print(lb2[i])
