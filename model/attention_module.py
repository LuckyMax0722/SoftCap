import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class AttentionModule(nn.Module):
    def __init__(self, in_size, out_size, hidden_size, use_relation=False, return_orientation=False):
        super(AttentionModule, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.encoder_attention = nn.Linear(in_size, hidden_size)  # 32, 512
        self.full_att = nn.Linear(hidden_size, out_size)  # 512, 1
        self.use_relation = use_relation
        self.return_orientation = return_orientation

    def forward_train(self, data_dict):
        object_masks = data_dict['object_mask']  # batch_size, num_proposals
        if self.use_relation:
            target_feats = data_dict["enhanced_feats"]  # batch_size, feat_size
            obj_feats = data_dict["bbox_feature"]  # batch_size, num_proposals, feat_size
        else:
            target_feats = data_dict["select_feats"]  # batch_size, feat_size
            obj_feats = data_dict["object_feats"]  # batch_size, num_proposals, feat_size

        # 用target_feats和obj_feats做attention
        combined = self.encoder_attention(obj_feats)  # batch_size, num_proposals, hidden_size
        combined += self.encoder_attention(target_feats).unsqueeze(1)  # batch_size, num_proposals, hidden_size
        combined = torch.tanh(combined)
        scores = self.full_att(combined)  # batch_size, num_proposals, 1
        scores.masked_fill_(object_masks.unsqueeze(-1) == 0, float('-1e30'))

        masks = F.softmax(scores, dim=1)  # batch_size, num_proposals, 1
        attended = obj_feats * masks # batch_size, num_proposals, feat_size
        attended = attended.sum(1)  # batch_size, feat_size

        data_dict["attention_features"] = attended

        return data_dict

    def forward_val(self, data_dict):
        object_masks = data_dict['object_mask']  # 1, num_proposals
        num_proposals = object_masks.shape[1]
        feat_size = self.in_size
        attention_features = torch.zeros(num_proposals, feat_size).cuda()  # num_proposals, feat_size

        if self.use_relation:
            for prop_id in range(num_proposals):
                target_feat = data_dict["bbox_feature"][0][prop_id].unsqueeze(0)
                if self.return_orientation:
                    obj_feats = data_dict["rel_bbox_feature"][0][prop_id].unsqueeze(0)  # 1, num_proposals, feat_size
                else:
                    obj_feats = data_dict["bbox_feature"]  # 1, num_proposals, feat_size

                combined = self.encoder_attention(obj_feats)  # 1, num_proposals, hidden_size
                combined += self.encoder_attention(target_feat).unsqueeze(1)  # 1, num_proposals, hidden_size
                combined = torch.tanh(combined)
                scores = self.full_att(combined)  # batch_size, num_proposals, 1
                scores.masked_fill_(object_masks.unsqueeze(-1) == 0, float('-1e30'))

                masks = F.softmax(scores, dim=1)  # 1, num_proposals, 1
                attended = obj_feats * masks
                attended = attended.sum(1)  # 1, feat_size
                attention_features[prop_id] = attended

        else:
            obj_feats = data_dict["object_feats"]  # 1, num_proposals, feat_size
            for prop_id in range(num_proposals):
                target_feat = obj_feats[0][prop_id].unsqueeze(0)  # 1, feat_size
                combined = self.encoder_attention(obj_feats)  # 1, num_proposals, hidden_size
                combined += self.encoder_attention(target_feat).unsqueeze(1)  # 1, num_proposals, hidden_size
                combined = torch.tanh(combined)
                scores = self.full_att(combined)  # batch_size, num_proposals, 1
                scores.masked_fill_(object_masks.unsqueeze(-1) == 0, float('-1e30'))

                masks = F.softmax(scores, dim=1)  # 1, num_proposals, 1
                attended = obj_feats * masks
                attended = attended.sum(1)  # 1, feat_size
                attention_features[prop_id] = attended

        data_dict["attention_features"] = attention_features  # num_proposals, feat_size

        return data_dict
