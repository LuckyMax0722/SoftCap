import torch
import torch.nn as nn
import numpy as np
import sys
import os
import json
import pickle
import pytorch_lightning as pl

from model.caption_module import CaptionModule
from model.softgroup import SoftGroup
from model.relation_graph_module import GraphModule, bbox_pred_module_train, bbox_pred_module_val, bbox_pred_module_test
from model.attention_module import AttentionModule
from model.cac_caption_module import CACModule

from utils.nn_distance import nn_distance
from utils.box_util import box3d_iou_batch_tensor, box_assignment

from utils.val_helper import decode_caption, check_candidates, organize_candidates, prepare_corpus, collect_results_cpu, \
    save_pred_instances, save_gt_instances

from loss_helper import radian_to_label, compute_node_distance_loss, compute_node_orientation_loss

import lib.capeval.bleu.bleu as capblue
import lib.capeval.cider.cider as capcider
import lib.capeval.rouge.rouge as caprouge
import lib.capeval.meteor.meteor as capmeteor

sys.path.append(os.path.join(os.getcwd(), "lib"))  # HACK add the lib folder
from lib.config import CONF

from plyfile import PlyData, PlyElement
from utils.box_util import get_3d_box_batch
import glob
from collections import OrderedDict
import datetime

vocab_path = os.path.join(CONF.PATH.DATA, "Scanrefer_vocabulary.json")
GLOVE_PICKLE = os.path.join(CONF.PATH.DATA, "glove.p")


class CapNet(pl.LightningModule):
    def __init__(self, val_tf_on=False, sc=False, use_relation=True, use_attention=False, use_cac=False):
        super().__init__()
        self.n_gts = 0
        self.n_preds = 0
        self.results = []  # used for eval map
        self.vocabulary = json.load(open(vocab_path))
        self.embeddings = pickle.load(open(GLOVE_PICKLE, "rb"))
        self.organized = json.load(open(os.path.join(CONF.PATH.DATA, "scanrefer/ScanRefer_filtered_organized.json")))
        self.candidates = {}
        self.val_tf_on = val_tf_on
        self.corpus = {}
        self.CLASSES = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture',
                        'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink',
                        'bathtub', 'otherfurniture')
        self.nyu_id = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)
        self.sc = sc
        self.use_relation = use_relation
        self.use_attention = use_attention
        self.use_cac = use_cac
        self.filename = ''
        # Define the model
        # -------------------------------------------------------------
        # ----------- SoftGroup-based Detection Backbone --------------
        self.softgroup_module = SoftGroup(in_channels=CONF.softgroup.in_channels,
                                          channels=CONF.softgroup.channels,
                                          num_blocks=CONF.softgroup.num_blocks,
                                          semantic_classes=CONF.softgroup.semantic_classes,
                                          instance_classes=CONF.softgroup.instance_classes,
                                          ignore_label=CONF.softgroup.ignore_label,
                                          grouping_cfg=CONF.grouping_cfg,
                                          instance_voxel_cfg=CONF.instance_voxel_cfg,
                                          train_cfg=CONF.train_cfg,
                                          test_cfg=CONF.test_cfg,
                                          fixed_modules=CONF.softgroup.fixed_modules,
                                          )
        # --------------------- Relation Module ---------------------
        self.relation_graph_module = GraphModule(in_size=CONF.graph_module.in_size,
                                                 out_size=CONF.graph_module.out_size,
                                                 num_layers=CONF.graph_module.num_graph_steps,
                                                 num_proposals=CONF.graph_module.num_proposals,
                                                 feat_size=CONF.graph_module.feat_size,
                                                 num_locals=CONF.graph_module.num_locals,
                                                 query_mode=CONF.graph_module.query_mode,
                                                 graph_mode=CONF.graph_module.graph_mode,
                                                 return_orientation=CONF.graph_module.return_orientation,
                                                 )

        # --------------------- Captioning Module ---------------------
        if not self.use_cac:
            if self.use_attention:
                self.attention_module = AttentionModule(in_size=CONF.attention_module.in_size,
                                                        out_size=CONF.attention_module.out_size,
                                                        hidden_size=CONF.attention_module.hidden_size,
                                                        use_relation=self.use_relation,
                                                        return_orientation=CONF.graph_module.return_orientation)

            self.caption_module = CaptionModule(self.vocabulary, self.embeddings,
                                                emb_size=CONF.caption_module.emb_size,
                                                feat_size=CONF.caption_module.feat_size,
                                                hidden_size=CONF.caption_module.hidden_size,
                                                num_proposals=CONF.train_cfg.max_proposal_num,
                                                use_relation=self.use_relation,
                                                use_attention=self.use_attention)
        if self.use_cac:
            self.caption_module = CACModule(self.vocabulary, self.embeddings,
                                            emb_size=CONF.caption_module.emb_size,
                                            feat_size=CONF.caption_module.feat_size,
                                            hidden_size=CONF.caption_module.hidden_size,
                                            num_proposals=CONF.train_cfg.max_proposal_num,
                                            use_relation=self.use_relation)

    def training_step(self, batch):
        batch = self.softgroup_module.forward(batch, mode='train')
        semantic_loss = batch["detection_log_vars"]['semantic_loss']
        offset_loss = batch["detection_log_vars"]['offset_loss']
        cls_loss = batch["detection_log_vars"]['cls_loss']
        mask_loss = batch["detection_log_vars"]['mask_loss']
        iou_score_loss = batch["detection_log_vars"]['iou_score_loss']
        detection_loss = batch["detection_loss"]
        ori_loss = torch.zeros(1)[0].cuda()
        ori_acc = torch.zeros(1)[0].cuda()
        dist_loss = torch.zeros(1)[0].cuda()
        # print('softgroup ok')
        if self.use_relation:
            # graph_module
            batch = bbox_pred_module_train(batch)
            # print('bbox_pred_module ok')
            batch = self.relation_graph_module.forward(batch)
            # print('relation_graph_module ok')
            _, batch['object_assignment'], _, _ = nn_distance(batch['bbox_center'].cuda(),
                                                              batch['center_label'])  # 4, 128

            ori_loss, ori_acc = compute_node_orientation_loss(batch, num_bins=6)
            dist_loss = compute_node_distance_loss(batch)

        if (not self.use_cac) and self.use_attention:
            # attention_module
            batch = self.attention_module.forward_train(batch)
            # print('attention ok')

        if not self.sc:
            # use normal cross entropy loss
            # predict language_features
            batch = self.caption_module.forward(batch, mode='train')
            # print('caption ok')
            cap_loss = batch['cap_loss']
            cap_acc = batch['cap_acc']
            loss = 5 * detection_loss + 0.5 * cap_loss + 1 * (ori_loss + dist_loss)

            self.log("semantic_loss", semantic_loss, on_step=True, prog_bar=True, logger=True)
            self.log("offset_loss", offset_loss, on_step=True, prog_bar=True, logger=True)
            self.log("cls_loss", cls_loss, on_step=True, prog_bar=True, logger=True)
            self.log("mask_loss", mask_loss, on_step=True, prog_bar=True, logger=True)
            self.log("iou_score_loss", iou_score_loss, on_step=True, prog_bar=True, logger=True)
            self.log("cap_loss", cap_loss, on_step=True, prog_bar=True, logger=True)
            self.log("cap_acc", cap_acc, on_step=True, prog_bar=True, logger=True)
            self.log("ori_loss", ori_loss, on_step=True, prog_bar=True, logger=True)
            self.log("ori_acc", ori_acc, on_step=True, prog_bar=True, logger=True)
            self.log("dist_loss", dist_loss, on_step=True, prog_bar=True, logger=True)

        if self.sc:
            # use self-critical training
            out = {}
            # get inference result (using greedy sampling)
            self.caption_module.eval()
            with torch.no_grad():
                greedy_seq, _ = self.caption_module.forward_sample_greedy(batch)  # batch_size, max_len

            # get training result (using monte carlo sampling)
            self.caption_module.train()
            gen_result, sample_logprobs = self.caption_module.forward_sample_mc(data_dict=batch, beam_size=5)

            reward = get_self_critical_reward(greedy_seq, batch['lang_ids'], gen_result, batch)
            reward = torch.from_numpy(reward).cuda()
            out['reward'] = reward[:, 0].mean()

            sample_logprobs = sample_logprobs.reshape(-1)
            reward = reward.reshape(-1)
            mask = (gen_result.data > 0).to(sample_logprobs)
            mask = mask * batch['good_clu_masks'].repeat_interleave(2).unsqueeze(1)
            mask = mask.reshape(-1)
            sc_loss = - sample_logprobs * reward * mask

            sc_loss = torch.sum(sc_loss) / (torch.sum(mask) + 1e-6)

            out['loss'] = sc_loss
            loss = 5 * detection_loss + 0.5 * sc_loss.mean() + 1 * (ori_loss + dist_loss)
            self.log("semantic_loss", semantic_loss, on_step=True, prog_bar=True, logger=True)
            self.log("offset_loss", offset_loss, on_step=True, prog_bar=True, logger=True)
            self.log("cls_loss", cls_loss, on_step=True, prog_bar=True, logger=True)
            self.log("mask_loss", mask_loss, on_step=True, prog_bar=True, logger=True)
            self.log("iou_score_loss", iou_score_loss, on_step=True, prog_bar=True, logger=True)
            self.log("detection_loss", detection_loss, on_step=True, prog_bar=True, logger=True)
            self.log("sc_loss", sc_loss.mean(), on_step=True, prog_bar=True, logger=True)
            self.log("ori_loss", ori_loss, on_step=True, prog_bar=True, logger=True)
            self.log("ori_acc", ori_acc, on_step=True, prog_bar=True, logger=True)
            self.log("dist_loss", dist_loss, on_step=True, prog_bar=True, logger=True)

        return loss

    def on_train_epoch_end(self):
        params = self.state_dict()
        # to cpu
        params_cpu = {k: v.cpu() for k, v in params.items()}
        # save model file
        self.filename = 'model0623_'
        current_time = datetime.datetime.now()
        time_string = current_time.strftime("%H:%M:%S")
        self.filename += time_string
        if self.use_relation:
            self.filename += '_relation'
        if (not self.use_cac) and self.use_attention:
            self.filename += '_attention'
        if self.use_cac:
            self.filename += '_cac'
        if self.sc:
            self.filename += '_sc'
        self.filename += f'_epoch{self.current_epoch}.pth'
        torch.save(params_cpu, self.filename)
        return None

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.candidates = {}
        batch = self.softgroup_module.forward(batch, mode='val')

        # graph_module
        if self.use_relation:
            batch = bbox_pred_module_val(batch)
            batch = self.relation_graph_module.forward_test(batch)
        if (not self.use_cac) and self.use_attention:
            batch = self.attention_module.forward_val(batch)

        # caption_module
        batch = self.caption_module.forward(batch, mode='val')
        captions = batch['final_lang'].argmax(-1)  # M , (max_len-1)

        cls_scores = batch['cls_scores']
        mask_scores = batch['mask_scores']
        proposals_idx = batch['proposals_idx']

        num_points = batch['coords_float'].shape[0]  # num_point
        num_instances = cls_scores.size(0)  # #proposal
        cls_scores = cls_scores.softmax(1)  # softmax score
        max_cls_score, final_cls = cls_scores.max(1)  # M, 1 和 M, 1 best score of each proposal

        mask_pred = torch.zeros((num_instances, num_points), dtype=torch.int, device='cuda')  # M, num_point

        for instance_idx in range(num_instances):
            cur_mask_scores = mask_scores[:, final_cls[instance_idx]]  # N, 1， mask score for point in class i
            mask_inds = cur_mask_scores > -0.5  # threshold assign mask to points
            cur_proposals_idx = proposals_idx[mask_inds].long()

            # M , num_point
            mask_pred[instance_idx, cur_proposals_idx[:, 1]] = 1

        clu_point = torch.zeros((num_instances, num_points), dtype=torch.int, device='cuda')  # M, num_point
        # M , num_point
        clu_point[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1

        final_proposals = clu_point * mask_pred  # M ,num_point final results points belong to the proposal

        final_proposals = final_proposals.cpu().numpy()
        pred_bbox = np.zeros((num_instances, 6))  # M, 6 save each proposal bbox center and size
        for i in range(num_instances):
            idx = (final_proposals[i] == 1)
            object_points = batch['coords_float'][idx].cpu().numpy()
            if object_points.shape[0] != 0:
                max_corner = object_points.max(0)
                min_corner = object_points.min(0)
            else:
                max_corner = np.zeros(3)
                min_corner = np.zeros(3)
            center = (max_corner + min_corner) / 2
            size = abs(max_corner - min_corner)
            pred_bbox[i, 0:3] = center
            pred_bbox[i, 3:6] = size

        pred_box_corner = get_3d_box_batch(pred_bbox[:, 3:6], np.zeros(num_instances), pred_bbox[:, 0:3])  # M, 8, 3
        pred_box_corner = torch.from_numpy(pred_box_corner).unsqueeze(0).cuda()  # 1, M, 8, 3

        _, batch['object_assignment'], _, _ = nn_distance(torch.from_numpy(pred_bbox[:, 0:3]).unsqueeze(0).cuda(),
                                                          batch['center_label'])  # 1, M

        # pick out object ids of detected objects
        detected_object_ids = torch.gather(batch["scene_object_ids"], 1, batch["object_assignment"])  # 1, M

        # bbox corners
        assigned_target_bbox_corners = torch.gather(
            batch["gt_box_corner_label"],
            1,
            batch["object_assignment"].view(1, num_instances, 1, 1).repeat(1, 1, 8, 3)
        )  # 1, M, 8, 3

        detected_bbox_corners = pred_box_corner  # 1, M, 8, 3
        # compute IoU between each detected box and each ground truth box
        ious = box3d_iou_batch_tensor(
            assigned_target_bbox_corners.view(-1, 8, 3),  # 1*M, 8, 3
            detected_bbox_corners.view(-1, 8, 3)  # 1*M, 8, 3
        ).view(1, num_instances)  # 1, M

        # find good boxes (IoU > threshold)
        good_bbox_masks = ious > 0.5  # 1, M
        valid_bbox_masks = final_cls != self.softgroup_module.instance_classes  # M
        iou_cache = {}
        for prop_id in range(num_instances):
            if good_bbox_masks[0, prop_id] == 1:
                scene_id = str(batch['scan_ids'][0])
                object_id = str(detected_object_ids[0, prop_id].item())
                caption_decoded = decode_caption(captions[prop_id], self.vocabulary["idx2word"])

                try:
                    ann_list = list(self.organized[scene_id][object_id].keys())
                    object_name = self.organized[scene_id][object_id][ann_list[0]]["object_name"]
                    # store
                    key = "{}|{}|{}".format(scene_id, object_id, object_name)
                    if key not in self.candidates:
                        self.candidates[key] = [caption_decoded]
                        iou_cache[key] = ious[0, prop_id]
                    else:
                        # update the caption if the iou is higher
                        if ious[0, prop_id] > iou_cache[key]:
                            self.candidates[key] = [caption_decoded]
                            iou_cache[key] = ious[0, prop_id]

                except KeyError:
                    continue

        self.n_preds += num_instances  # #pred_bbox

        for prop_id in range(num_instances):
            scene_id = str(batch['scan_ids'][0])
            object_id = str(detected_object_ids[0, prop_id].item())
            caption_decoded = decode_caption(captions[prop_id], self.vocabulary["idx2word"])
            key = "{}|{}".format(scene_id, str(prop_id))

            if good_bbox_masks[0, prop_id] == 1:
                try:
                    ann_list = list(self.organized[scene_id][object_id].keys())
                    object_name = self.organized[scene_id][object_id][ann_list[0]]["object_name"]
                    self.candidates[key] = [caption_decoded]  # all pred descriptions
                    self.corpus[key] = self.corpus["{}|{}|{}".format(scene_id, object_id, object_name)]

                except KeyError:
                    continue

        return None

    def on_validation_start(self) -> None:
        self.filename = 'model'
        current_time = datetime.datetime.now()
        time_string = current_time.strftime("%H:%M:%S")
        self.filename += time_string
        if self.use_relation:
            self.filename += '_relation'
        if (not self.use_cac) and self.use_attention:
            self.filename += '_attention'
        if self.use_cac:
            self.filename += '_cac'
        if self.sc:
            self.filename += '_sc'
        self.filename += f'_epoch{self.current_epoch}.pth'

        corpus_path = os.path.join(CONF.PATH.OUTPUT, "corpus_val.json")
        if not os.path.exists(corpus_path):
            print("preparing corpus...")
            raw_data = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))
            corpus = prepare_corpus(raw_data, CONF.TRAIN.MAX_DES_LEN)
            with open(corpus_path, "w") as f:
                json.dump(corpus, f, indent=4)
        else:
            print("loading corpus...")
            with open(corpus_path) as f:
                self.corpus = json.load(f)

        self.n_gts = len(self.corpus)
        self.n_preds = 0

    def on_validation_end(self):
        self.candidates = check_candidates(self.corpus, self.candidates)
        self.candidates = organize_candidates(self.corpus, self.candidates)

        pred_path = os.path.join(CONF.PATH.OUTPUT, f"pred_val_{self.filename}.json")

        print("generating descriptions...")
        with open(pred_path, "w") as f:
            json.dump(self.candidates, f, indent=4)

        print("computing scores...")
        bleu = capblue.Bleu(4).compute_score(self.corpus, self.candidates)
        cider = capcider.Cider().compute_score(self.corpus, self.candidates)
        rouge = caprouge.Rouge().compute_score(self.corpus, self.candidates)
        meteor = capmeteor.Meteor().compute_score(self.corpus, self.candidates)

        bleu_recall = np.asarray(bleu[1][3][:self.n_gts]).mean()
        cider_recall = np.asarray(cider[1][:self.n_gts]).mean()
        rouge_recall = np.asarray(rouge[1][:self.n_gts]).mean()
        meteor_recall = np.asarray(meteor[1][:self.n_gts]).mean()

        bleu_precision = np.asarray(bleu[1][3][self.n_gts:]).sum() / self.n_preds
        cider_precision = np.asarray(cider[1][self.n_gts:]).sum() / self.n_preds
        rouge_precision = np.asarray(rouge[1][self.n_gts:]).sum() / self.n_preds
        meteor_precision = np.asarray(meteor[1][self.n_gts:]).sum() / self.n_preds

        cider_f1 = 2 * cider_recall * cider_precision / (cider_recall + cider_precision)
        bleu_f1 = 2 * bleu_recall * bleu_precision / (bleu_recall + bleu_precision)
        meteor_f1 = 2 * meteor_recall * meteor_precision / (meteor_recall + meteor_precision)
        rouge_f1 = 2 * rouge_recall * rouge_precision / (rouge_recall + rouge_precision)

        current_time = datetime.datetime.now()
        time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")
        with open(f"{CONF.PATH.OUTPUT}/captioning_score.txt", "a") as file:
            file.write('Current Time： ' + time_string + '\n')
            file.write(self.filename + '\n')
            file.write('----Recall_scores----\n')
            file.write('CIDEr_recall is: {:.15f}\n'.format(cider_recall))
            file.write('BLEU-4_recall is: {:.15f}\n'.format(bleu_recall))
            file.write('METEOR_recall is: {:.15f}\n'.format(meteor_recall))
            file.write('ROUGE_recall is: {:.15f}\n'.format(rouge_recall))
            file.write('----Precision_scores----\n')
            file.write('CIDEr_precision is: {:.15f}\n'.format(cider_precision))
            file.write('BLEU-4_precision is: {:.15f}\n'.format(bleu_precision))
            file.write('METEOR_precision is: {:.15f}\n'.format(meteor_precision))
            file.write('ROUGE_precision is: {:.15f}\n'.format(rouge_precision))
            file.write('----F1_scores----\n')
            file.write('CIDEr_F1 is: {:.15f}\n'.format(cider_f1))
            file.write('BLEU-4_F1 is: {:.15f}\n'.format(bleu_f1))
            file.write('METEOR_F1 is: {:.15f}\n'.format(meteor_f1))
            file.write('ROUGE_F1 is: {:.15f}\n'.format(rouge_f1))
            file.write('\n')

        print('CIDEr_recall is:', cider_recall)
        print('BLEU-4_recall is:', bleu_recall)
        print('METEOR_recall is:', meteor_recall)
        print('ROUGE_recall is:', rouge_recall)

        print('CIDEr_precision is:', cider_precision)
        print('BLEU-4_precision is:', bleu_precision)
        print('METEOR_precision is:', meteor_precision)
        print('ROUGE_precision is:', rouge_precision)

        print('CIDEr_F1 is:', cider_f1)
        print('BLEU-4_F1 is:', bleu_f1)
        print('METEOR_F1 is:', meteor_f1)
        print('ROUGE_F1 is:', rouge_f1)

        return None

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        # scheduler = StepLR(optimizer, step_size=1, gamma=0.9)  # 每个epoch后，学习率乘0.9
        # return [optimizer], [scheduler]

        opt_params = []
        opt_params.append({'params': self.softgroup_module.parameters(), 'weight_decay': 0})

        if self.use_relation:
            opt_params.append({'params': self.relation_graph_module.parameters(), 'weight_decay': CONF.model_setting.weight_decay})

        if (not self.use_cac) and self.use_attention:
            opt_params.append({'params': self.attention_module.parameters(), 'weight_decay': CONF.model_setting.weight_decay})

        opt_params.append({'params': self.caption_module.parameters(), 'weight_decay': CONF.model_setting.weight_decay})

        optimizer = torch.optim.Adam(opt_params, lr=CONF.model_setting.lr)

        return optimizer


def get_self_critical_reward(greedy_seq, data_gts, gen_result, data_dict):
    batch_size = greedy_seq.shape[0]
    gen_result_size = gen_result.shape[0]
    seq_per_obj = gen_result_size // batch_size  # gen_result_size  = batch_size * seq_per_obj

    gen_result = gen_result.data.cpu().numpy()
    greedy_seq = greedy_seq.data.cpu().numpy()

    # caption to staring for reward
    def array_to_str(arr):
        out = ''
        if str(arr[0]) != '2':
            out += str(2) + ' '  # sos
        for i in range(len(arr)):
            out += str(arr[i]) + ' '
            if arr[i] == 3:
                break
        if out[-2] != '3':
            out += '3'
        return out.strip()

    res = OrderedDict()
    gts = OrderedDict()

    for i in range(gen_result_size):  # sample results
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):  # greedy results
        res[gen_result_size + i] = [array_to_str(greedy_seq[i])]

    for i in range(batch_size):
        gts[i] = [array_to_str(data_gts[i][j]) for j in range(len(data_gts[i]))]

    res_ = [{'id': i, 'caption': res[i]} for i in range(len(res))]
    res__ = {i: res[i] for i in range(len(res_))}
    gts_ = {i: gts[i // seq_per_obj] for i in range(gen_result_size)}
    gts_.update({i + gen_result_size: gts[i] for i in range(batch_size)})

    _, cider_scores = capcider.Cider().compute_score(gts_, res__)

    scores = cider_scores

    scores = scores[:gen_result_size].reshape(batch_size, seq_per_obj) - scores[-batch_size:][:, np.newaxis]

    scores = scores.reshape(gen_result_size)
    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    return rewards


def rl_crit(input, seq, reward, reduction='mean'):
    N, L = input.shape[:2]

    input = input.reshape(-1)
    reward = reward.reshape(-1)
    mask = (seq > 0).to(input)
    mask = mask.reshape(-1)
    output = - input * reward * mask

    if reduction == 'none':
        output = output.view(N, L).sum(1) / mask.view(N, L).sum(1)
    elif reduction == 'mean':
        output = torch.sum(output) / torch.sum(mask)

    return output
