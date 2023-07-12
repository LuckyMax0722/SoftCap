import torch
import torch.nn as nn
import numpy as np
import sys
import os
import json
import pickle
import pytorch_lightning as pl
import argparse
import datetime
from plyfile import PlyData, PlyElement

sys.path.append(os.getcwd())  # HACK add the root folder
from model.caption_module import CaptionModule
from model.softgroup import SoftGroup
from model.relation_graph_module import GraphModule, bbox_pred_module_train, bbox_pred_module_val
from model.attention_module import AttentionModule
from model.cac_caption_module import CACModule

from utils.nn_distance import nn_distance
from utils.box_util import box3d_iou_batch_tensor, get_3d_box_batch
from utils.val_helper import decode_caption, check_candidates, organize_candidates, prepare_corpus, collect_results_cpu, \
    save_pred_instances, save_gt_instances

sys.path.append(os.path.join(os.getcwd(), "lib"))  # HACK add the lib folder
import lib.capeval.bleu.bleu as capblue
import lib.capeval.cider.cider as capcider
import lib.capeval.rouge.rouge as caprouge
import lib.capeval.meteor.meteor as capmeteor
from lib.dataset import ScannetReferenceDataset
from lib.dataset import get_scanrefer
from lib.config import CONF

import glob
from multiprocessing import Pool
from eval_det import eval_det
from visualization import write_bbox
from datamodule import ScanReferDataModule

from torch.utils.data import DataLoader

vocab_path = os.path.join(CONF.PATH.DATA, "Scanrefer_vocabulary.json")
GLOVE_PICKLE = os.path.join(CONF.PATH.DATA, "glove.p")


class CapNetEval(pl.LightningModule):
    def __init__(self, use_relation=False, use_attention=False, use_cac=False, eval_detection=False, eval_caption=False,
                 visualization=False, min_iou=0.5):
        super().__init__()
        self.n_gts = 0
        self.n_preds = 0
        self.results = []  # 用来验证map
        self.vocabulary = json.load(open(vocab_path))
        self.embeddings = pickle.load(open(GLOVE_PICKLE, "rb"))
        self.organized = json.load(open(os.path.join(CONF.PATH.DATA, "scanrefer/ScanRefer_filtered_organized.json")))
        self.candidates = {}
        self.corpus = {}
        self.CLASSES = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture',
                        'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink',
                        'bathtub', 'otherfurniture')
        self.nyu_id = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)
        self.eval_detection = eval_detection
        self.eval_caption = eval_caption
        self.visualization = visualization
        self.min_iou = min_iou
        self.use_relation = use_relation
        self.use_attention = use_attention
        self.use_cac = use_cac

        # Define the model
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
                                          fixed_modules=CONF.softgroup.fixed_modules)

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

    def test_step(self, batch, batch_idx):  # batch size must be 1
        if self.eval_detection:
            ret = self.softgroup_module.forward(batch, mode='eval')
            self.results.append(ret)
        if self.eval_caption:
            if batch_idx == 0:
                self.candidates = {}
            scan_id = batch['scan_ids'][0]
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
            cls_scores = cls_scores.softmax(1)  # softmax scores
            max_cls_score, final_cls = cls_scores.max(1)  # M, 1 and M, 1 Best score of Each proposal

            mask_pred = torch.zeros((num_instances, num_points), dtype=torch.int, device='cuda')  # M, num_point
            for instance_idx in range(num_instances):
                cur_mask_scores = mask_scores[:, final_cls[instance_idx]]  # N, 1， mask score
                mask_inds = cur_mask_scores > -0.5  # threshold assign mask to points
                cur_proposals_idx = proposals_idx[mask_inds].long()
                # M , num_point
                mask_pred[instance_idx, cur_proposals_idx[:, 1]] = 1

            clu_point = torch.zeros((num_instances, num_points), dtype=torch.int, device='cuda')  # M, num_point
            # M , num_point points belong to proposal
            clu_point[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1

            final_proposals = clu_point * mask_pred  # M ,num_point points belong to proposal

            final_proposals = final_proposals.cpu().numpy()
            pred_bbox = np.zeros((num_instances, 6))  # M, 6 proposal bbox center and size
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
            good_bbox_masks = ious > self.min_iou  # 1, M
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
                        self.candidates[key] = [caption_decoded]  # all pred captions
                        self.corpus[key] = self.corpus["{}|{}|{}".format(scene_id, object_id, object_name)]

                    except KeyError:
                        continue

            # vis bbox
            final_proposals = final_proposals[valid_bbox_masks.cpu().numpy()]
            detected_object_ids = detected_object_ids.flatten()[valid_bbox_masks.cpu().numpy()]

            if self.visualization:
                visual(final_proposals, detected_object_ids, scan_id)

        return None

    def on_test_start(self) -> None:
        if self.eval_caption:
            corpus_path = os.path.join(CONF.PATH.OUTPUT, "corpus_val.json")
            if not os.path.exists(corpus_path):
                print("preparing corpus...")
                raw_data = json.load(open(os.path.join(CONF.PATH.DATA, "scanrefer/ScanRefer_filtered_val.json")))
                corpus = prepare_corpus(raw_data, CONF.TRAIN.MAX_DES_LEN)
                with open(corpus_path, "w") as f:
                    json.dump(corpus, f, indent=4)
            else:
                print("loading corpus...")
                with open(corpus_path) as f:
                    self.corpus = json.load(f)

            self.n_gts = len(self.corpus)
            self.n_preds = 0

    def on_test_end(self):
        if self.eval_detection:
            self.results = collect_results_cpu(self.results, len(self.results))
            scan_ids = []
            pred_insts, gt_insts = [], []
            for res in self.results:
                scan_ids.append(res['scan_id'])
                pred_insts.append(res['pred_instances'])
                gt_insts.append(res['gt_instances'])
            root = CONF.PATH.OUTPUT
            save_pred_instances(root, 'pred_instance', scan_ids, pred_insts, self.nyu_id)
            save_gt_instances(root, 'gt_instance', scan_ids, gt_insts, self.nyu_id)

            # calculate MAP
            data_path = CONF.PATH.SCANNET
            results_path = CONF.PATH.OUTPUT
            iou_threshold = self.min_iou
            instance_paths = glob.glob(os.path.join(results_path, 'pred_instance', '*.txt'))
            instance_paths.sort()

            CLASS_LABELS = [
                'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture',
                'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub',
                'otherfurniture'
            ]
            VALID_CLASS_IDS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]

            def single_process(instance_path):
                img_id = os.path.basename(instance_path)[:-4]
                print('Processing', img_id)
                gt = os.path.join(data_path, 'val', img_id + '_inst_nostuff.pth')  # 0-based index
                assert os.path.isfile(gt)
                coords, rgb, semantic_label, instance_label, _, _, _, _ = torch.load(gt)
                pred_infos = open(instance_path, 'r').readlines()
                pred_infos = [x.rstrip().split() for x in pred_infos]  # nyu_id index
                mask_path, labels, scores = list(zip(*pred_infos))
                pred = []
                for mask_path, label, score in pred_infos:
                    mask_full_path = os.path.join(results_path, 'pred_instance', mask_path)
                    mask = np.array(open(mask_full_path).read().splitlines(), dtype=int).astype(bool)
                    instance = coords[mask]
                    box_min = instance.min(0)
                    box_max = instance.max(0)
                    box = np.concatenate([box_min, box_max])
                    class_name = CLASS_LABELS[VALID_CLASS_IDS.index(int(label))]
                    pred.append((class_name, box, float(score)))

                instance_num = int(instance_label.max()) + 1
                gt = []

                for i in range(instance_num):
                    inds = instance_label == i
                    gt_label_loc = np.nonzero(inds)[0][0]
                    cls_id = int(semantic_label[gt_label_loc])
                    if cls_id >= 2:
                        instance = coords[inds]
                        box_min = instance.min(0)
                        box_max = instance.max(0)
                        box = np.concatenate([box_min, box_max])
                        class_name = CLASS_LABELS[cls_id - 2]
                        gt.append((class_name, box))
                return img_id, pred, gt

            pred_gt_results = []
            for i in instance_paths:
                pred_gt_results.append(single_process(i))

            pred_all = {}
            gt_all = {}
            for img_id, pred, gt in pred_gt_results:
                pred_all[img_id] = pred
                gt_all[img_id] = gt

            print('Evaluating...')
            eval_res = eval_det(pred_all, gt_all, ovthresh=iou_threshold)
            aps = list(eval_res[-1].values())
            mAP = np.mean(aps)
            print(f'mAP@{iou_threshold}:', mAP)

        if self.eval_caption:
            self.candidates = check_candidates(self.corpus, self.candidates)
            self.candidates = organize_candidates(self.corpus, self.candidates)

            pred_path = os.path.join(CONF.PATH.OUTPUT, "pred_val.json")

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
            with open("/home/jiachen/DenseCap/outputs/captioning_score_eval.txt", "a") as file:
                file.write('Current Time： ' + time_string + '\n')
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

                if self.eval_detection:
                    file.write('----mAP_scores----\n')
                    file.write('mAP@{}:{}\n'.format(iou_threshold, mAP))

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


def visual(final_proposals, detected_object_ids, scan_id):
    # write aligned scene ply
    plydata = PlyData.read(os.path.join(CONF.PATH.SCANNET, 'val', scan_id + '_vh_clean_2.ply'))
    num_verts = plydata['vertex'].count

    lines = open(os.path.join(CONF.PATH.SCANNET, 'val', scan_id + '.txt')).readlines()

    axis_align_matrix = None
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]
    axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))

    vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
    vertices[:, 0] = plydata['vertex'].data['x']
    vertices[:, 1] = plydata['vertex'].data['y']
    vertices[:, 2] = plydata['vertex'].data['z']

    pts = np.ones((vertices.shape[0], 4))
    pts[:, 0:3] = vertices[:, 0:3]  # using homogeneous coordinates
    pts = np.dot(pts, axis_align_matrix.transpose())  # Nx4
    aligned_vertices = np.copy(vertices)
    aligned_vertices[:, 0:3] = pts[:, 0:3]

    aligned_vertices[:, 0:3] = aligned_vertices[:, 0:3] - aligned_vertices[:, 0:3].mean(0)
    # vertices[:, 0:3] = vertices[:, 0:3] - vertices[:, 0:3].mean(0)

    plydata['vertex'].data['x'] = aligned_vertices[:, 0]
    plydata['vertex'].data['y'] = aligned_vertices[:, 1]
    plydata['vertex'].data['z'] = aligned_vertices[:, 2]

    if os.path.exists(os.path.join(CONF.PATH.OUTPUT, scan_id)):
        plydata.write(os.path.join(CONF.PATH.OUTPUT, scan_id, f'{scan_id}.ply'))
    else:
        os.mkdir(os.path.join(CONF.PATH.OUTPUT, scan_id))
        plydata.write(os.path.join(CONF.PATH.OUTPUT, scan_id, f'{scan_id}.ply'))

    # write objects ply
    for i in range(len(final_proposals)):
        idx = np.where(final_proposals[i] == 1)
        object_points = aligned_vertices[idx[0]]
        if not os.path.exists(os.path.join(CONF.PATH.OUTPUT, scan_id)):
            print('Creating new data folder: {}'.format(scan_id))
            os.mkdir(os.path.join(CONF.PATH.OUTPUT, scan_id))
        if idx[0].shape[0] != 0:
            box_min = np.min(object_points, axis=0)
            box_max = np.max(object_points, axis=0)
            color = np.asarray([220, 220, 60])
            output_path = os.path.join(CONF.PATH.OUTPUT, scan_id, 'object_id{}_bbox.ply'.format(detected_object_ids[i]))
            write_bbox(box_min, box_max, color, output_path)
    return None


class ScanReferEvalModule(pl.LightningDataModule):
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
        self.dataset_test = ScannetReferenceDataset(
            scanrefer=self.Scanrefer_eval_val,
            scanrefer_all_scene=self.all_scene_list,
            split='val',
            num_points=40000,
            augment=False,
        )

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=1, shuffle=False, num_workers=4,
                          collate_fn=self.dataset_test.collate_fn)


if __name__ == "__main__":
    # prepare dataset and dataloader
    data = ScanReferEvalModule()

    # create model
    model = CapNetEval(use_relation=CONF.model_setting.use_relation,
                       use_attention=CONF.model_setting.use_attention,
                       use_cac=CONF.model_setting.use_cac,
                       eval_detection=CONF.vis_setting.eval_detection,
                       eval_caption=CONF.vis_setting.eval_caption,
                       visualization=CONF.vis_setting.visualization,
                       min_iou=CONF.vis_setting.min_iou)

    # load model
    file = '/home/jiachen/DenseCap/scripts/model0625_12:16:49_relation_cac_sc_epoch12.pth'  # TODO: change this for model eval
    model.load_state_dict(torch.load(file), strict=False)

    # create trainer
    trainer = pl.Trainer(accelerator='gpu', devices=1)

    # performance on test set
    trainer.test(model, data)
