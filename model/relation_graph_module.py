import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as GUtils

from torch_geometric.data import Data as GData
from torch_geometric.data import DataLoader as GDataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.typing import Adj, Size

from utils.box_util import box3d_iou_batch_tensor, get_3d_box_batch

from scipy.sparse import coo_matrix

from lib.config import CONF

torch.autograd.set_detect_anomaly(True)


def write_ply(points, colour, output_file):
    file = open(output_file, 'w')
    file.write('ply \n')
    file.write('format ascii 1.0\n')
    file.write('element vertex {:d}\n'.format(len(points)))
    file.write('property float x\n')
    file.write('property float y\n')
    file.write('property float z\n')
    file.write('property uchar red\n')
    file.write('property uchar green\n')
    file.write('property uchar blue\n')
    file.write('element face {:d}\n'.format(0))
    file.write('property list uchar uint vertex_indices\n')
    file.write('end_header\n')
    for point in points:
        file.write(
            '{:f} {:f} {:f} {:d} {:d} {:d}\n'.format(point[0], point[1], point[2], colour[0], colour[1], colour[2]))
    file.close()


def only_for_vis_test(data_dict, pred_box_corner_each_scene_pad):
    scan_id = data_dict['scan_ids'][0]
    num_points = data_dict["num_points"][0]
    coords = data_dict['coords_float'][0: num_points]

    output = '/home/luk/DenseCap/outputs/'
    output_for_all_points = output + scan_id + '.ply'
    colour = [0, 0, 0]
    write_ply(coords, colour, output_for_all_points)
    colour = [255, 0, 0]
    for i in range(128):
        output_for_bbox = output + scan_id + '_object' + str(i) + '.ply'
        write_ply(pred_box_corner_each_scene_pad[0][i], colour, output_for_bbox)


def bbox_pred_module_train(data_dict):
    cls_scores = data_dict["cls_scores"].cuda()
    mask_scores = data_dict["mask_scores"].cuda()
    proposals_idx = data_dict["proposals_idx"].cuda()
    proposal_each_scene = data_dict["proposal_each_scene"].cuda()
    point_each_scene = data_dict["num_points"].cuda()
    all_points = data_dict['coords_float'].cuda()  # 获取场景内的所有的point坐标

    for batch_idx in range(data_dict["batch_size"]):
        num_points = point_each_scene[batch_idx]  # 场景内point的数量
        num_instances = proposal_each_scene[batch_idx]  # 场景内proposal的数量

        # 每个场景内proposal_idx开始索引和结束索引
        proposal_each_scene_idx_start = sum(proposal_each_scene[:batch_idx])
        proposal_each_scene_idx_end = sum(proposal_each_scene[:batch_idx + 1])

        # cls_scores是B个场景内的所有proposal的 --> M, 19
        # 获取每个batch下场景的softmax分类得分
        cls_scores_each_scene = cls_scores[proposal_each_scene_idx_start:proposal_each_scene_idx_end].softmax(1)

        max_cls_score, final_cls = cls_scores_each_scene.max(1)  # 1, M 和 1, M 每个proposal得到的分类最高分，以及属于哪个类

        # 注意 proposal使用的点比一个场景内的all_points点少
        # 获取当前场景内第一个proposal的开始点idx
        proposal_each_scene_points_idx_start = (proposals_idx[:, 0] == proposal_each_scene_idx_start).nonzero()
        # 获取当前场景内最后一个proposal的结束点idx
        proposal_each_scene_points_idx_end = (proposals_idx[:, 0] == proposal_each_scene_idx_end - 1).nonzero()

        # 注意 proposal使用的点比一个场景内的all_points点少
        # 获取当前场景内proposal的每个点的mask_scored， proposals_idx
        mask_scores_each_scene = mask_scores[
                                 proposal_each_scene_points_idx_start[0]:proposal_each_scene_points_idx_end[-1] + 1]
        proposals_idx_each_scene = proposals_idx[
                                   proposal_each_scene_points_idx_start[0]:proposal_each_scene_points_idx_end[-1] + 1]

        mask_pred = torch.zeros((num_instances, num_points), dtype=torch.int, device='cuda')  # M, num_point
        for instance_idx in range(num_instances):
            cur_mask_scores = mask_scores_each_scene[:, final_cls[instance_idx]]  # N, 1， N个点对第i个proposal的类的mask score
            mask_inds = cur_mask_scores > -0.5  # threshold 取mask高于阈值的点
            cur_proposals_idx = proposals_idx_each_scene[mask_inds].long()

            # 下面两行用于调整proposal和点的索引, proposal的类别从0开始
            cur_proposals_idx[:, 0] = cur_proposals_idx[:, 0] - cur_proposals_idx[0, 0]
            cur_proposals_idx[:, 1] = cur_proposals_idx[:, 1] - sum(point_each_scene[:batch_idx])

            # M , num_point 表示有哪些点可能属于这个proposal对应的cls
            mask_pred[instance_idx, cur_proposals_idx[:, 1]] = 1

        clu_point = torch.zeros((num_instances, num_points), dtype=torch.int, device='cuda')  # M, num_point
        cur_proposals_idx_2 = torch.zeros_like(proposals_idx_each_scene, device='cuda')

        # 下面两行用于调整proposal和点的索引, proposal的类别从0开始
        cur_proposals_idx_2[:, 0] = proposals_idx_each_scene[:, 0] - proposals_idx_each_scene[0, 0]
        cur_proposals_idx_2[:, 1] = proposals_idx_each_scene[:, 1] - sum(point_each_scene[:batch_idx])

        # M , num_point 表示有哪些点被group到这个proposal
        clu_point[cur_proposals_idx_2[:, 0].long(), cur_proposals_idx_2[:, 1].long()] = 1

        final_proposals = clu_point * mask_pred  # M ,num_point 最终结果，最终的proposal中有哪些点

        final_proposals = final_proposals.cpu().numpy()
        pred_bbox = np.zeros((num_instances, 6))  # M, 6 存储每个proposal的bbox的center和size

        # 获取当前场景内所有点的位置
        all_points_each_scene = all_points[sum(point_each_scene[:batch_idx]):sum(point_each_scene[:batch_idx + 1])]
        for instance_idx in range(num_instances):
            idx = (final_proposals[instance_idx] == 1)
            object_points = all_points_each_scene[idx].cpu().numpy()

            if object_points.shape[0] != 0:
                max_corner = object_points.max(0)
                min_corner = object_points.min(0)
            else:
                max_corner = np.zeros(3)
                min_corner = np.zeros(3)
            center = (max_corner + min_corner) / 2
            size = abs(max_corner - min_corner)
            pred_bbox[instance_idx, 0:3] = center
            pred_bbox[instance_idx, 3:6] = size

        pred_box_corner_each_scene_pad = np.zeros((128, 8, 3))
        pred_box_corner_each_scene = get_3d_box_batch(pred_bbox[:, 3:6], np.zeros(num_instances),
                                                      pred_bbox[:, 0:3])  # M, 8, 3
        pred_box_corner_each_scene_pad[0:num_instances] = pred_box_corner_each_scene  # 补全 128, 8, 3
        pred_box_corner_each_scene_pad = torch.from_numpy(pred_box_corner_each_scene_pad).unsqueeze(
            0).cuda()  # 1, 128, 8, 3

        pred_center_each_scene_pad = np.zeros((128, 3))

        pred_center_each_scene_pad[0:pred_box_corner_each_scene.shape[0]] = pred_bbox[:, 0:3]

        pred_center_each_scene_pad = torch.from_numpy(pred_center_each_scene_pad).unsqueeze(
            0).cuda()

        if batch_idx == 0:
            pred_box_corner = pred_box_corner_each_scene_pad.clone()
            pred_box_center = pred_center_each_scene_pad.clone()
        else:
            pred_box_corner = torch.cat((pred_box_corner, pred_box_corner_each_scene_pad), dim=0)
            pred_box_center = torch.cat((pred_box_center, pred_center_each_scene_pad), dim=0)

    # only_for_vis_test(data_dict, pred_box_corner)

    data_dict['bbox_corner'] = pred_box_corner
    data_dict['bbox_center'] = pred_box_center
    return data_dict


def bbox_pred_module_val(data_dict):
    cls_scores = data_dict["cls_scores"].cuda()
    mask_scores = data_dict["mask_scores"].cuda()
    proposals_idx = data_dict["proposals_idx"].cuda()
    proposal_each_scene = data_dict["proposal_each_scene"].cuda()
    point_each_scene = data_dict["num_points"].cuda()
    all_points = data_dict['coords_float'].cuda()  # 获取场景内的所有的point坐标
    num_points = data_dict["num_points"].cuda()  # 场景内point的数量

    num_instances = proposal_each_scene[0]  # proposal的数量
    cls_scores = cls_scores.softmax(1)  # softmax分类得分
    max_cls_score, final_cls = cls_scores.max(1)  # M, 1 和 M, 1 每个proposal得到的分类最高分，以及属于哪个类

    mask_pred = torch.zeros((num_instances, num_points), dtype=torch.int, device='cuda')  # M, num_point
    for instance_idx in range(num_instances):
        cur_mask_scores = mask_scores[:, final_cls[instance_idx]]  # N, 1， N个点对第i个proposal的类的mask score
        mask_inds = cur_mask_scores > -0.5  # threshold 取mask高于阈值的点
        cur_proposals_idx = proposals_idx[mask_inds].long()

        # M , num_point 表示有哪些点可能属于这个proposal对应的cls
        mask_pred[instance_idx, cur_proposals_idx[:, 1]] = 1

    clu_point = torch.zeros((num_instances, num_points), dtype=torch.int, device='cuda')  # M, num_point
    # M , num_point 表示有哪些点被group到这个proposal
    clu_point[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1

    final_proposals = clu_point * mask_pred  # M ,num_point 最终结果，最终的proposal中有哪些点

    final_proposals = final_proposals.cpu().numpy()
    pred_bbox = np.zeros((num_instances, 6))  # M, 6 存储每个proposal的bbox的center和size

    for instance_idx in range(num_instances):
        idx = (final_proposals[instance_idx] == 1)
        object_points = all_points[idx].cpu().numpy()

        if object_points.shape[0] != 0:
            max_corner = object_points.max(0)
            min_corner = object_points.min(0)
        else:
            max_corner = np.zeros(3)
            min_corner = np.zeros(3)
        center = (max_corner + min_corner) / 2
        size = abs(max_corner - min_corner)
        pred_bbox[instance_idx, 0:3] = center
        pred_bbox[instance_idx, 3:6] = size

    pred_box_corner_pad = np.zeros((128, 8, 3))
    pred_box_corner = get_3d_box_batch(pred_bbox[:, 3:6], np.zeros(num_instances),
                                       pred_bbox[:, 0:3])  # M, 8, 3
    pred_box_corner_pad[0:num_instances] = pred_box_corner  # 补全 128, 8, 3
    pred_box_corner_pad = torch.from_numpy(pred_box_corner_pad).unsqueeze(0).cuda()  # 1, 128, 8, 3

    data_dict['bbox_corner'] = pred_box_corner_pad

    # only_for_vis_test(data_dict, pred_box_corner_pad)

    return data_dict


# def bbox_pred_module(data_dict):
#     cls_scores = data_dict["cls_scores"]
#     mask_scores = data_dict["mask_scores"]
#     proposals_idx = data_dict["proposals_idx"]
#     proposal_each_scene = data_dict["proposal_each_scene"]
#     all_points = data_dict['coords_float']  # 获取场景内的所有的point坐标
#
#     instance_idx_start = 0
#     instance_idx_end = proposal_each_scene[0]
#
#     proposals_index = proposal_each_scene[0]
#     proposals_idx = proposals_idx.cuda()
#
#     point_idx_start = 0
#
#     all_points_idx_start = 0
#     all_points_idx_end = 0
#
#     for i in range(data_dict["batch_size"]):
#         num_points = data_dict["num_points"][i]  # 场景内point的数量
#         num_instances = proposal_each_scene[i]  # 场景内proposal的数量
#
#         # cls_scores是B个场景内的所有proposal的 --> sum(proposal_each_scene) * 19
#         cls_scores_each_scene = cls_scores[instance_idx_start:instance_idx_end].softmax(1)  # 获取每个batch下场景的softmax分类得分
#         # 下方对每个场景的proposal_each_scene进行索引的修改
#         if i < data_dict["batch_size"] - 1:
#             instance_idx_start = instance_idx_start + num_instances
#             instance_idx_end = instance_idx_end + proposal_each_scene[i + 1]
#
#         max_cls_score, final_cls = cls_scores_each_scene.max(1)  # M, 1 和 M, 1 每个proposal得到的分类最高分，以及属于哪个类
#
#         point_last = (proposals_idx[:, 0] == proposals_index - 1).nonzero()  # 获取当前场景内最后一个proposal的点
#         point_idx_end = point_last[-1]  # 获取当前场景内最后一个proposal的最后一个点
#         mask_scores_each_scene = mask_scores[point_idx_start:point_idx_end]
#         proposals_idx_each_scene = proposals_idx[point_idx_start:point_idx_end]
#         # 下方对每个场景的mask_scores_each_scene， proposals_idx_each_scene进行索引的修改
#         if i < data_dict["batch_size"] - 1:
#             point_idx_start = point_idx_end + 1
#             proposals_index = proposals_index + proposal_each_scene[i + 1]
#
#         mask_pred = torch.zeros((num_instances, num_points), dtype=torch.int, device='cuda')  # M, num_point
#         for j in range(num_instances):
#             cur_mask_scores = mask_scores_each_scene[:, final_cls[j]]  # N, 1， N个点对第i个proposal的类的mask score
#             mask_inds = cur_mask_scores > -0.5  # threshold 取mask高于阈值的点
#             cur_proposals_idx = proposals_idx_each_scene[mask_inds].long()
#             # 下面两行用于调整proposal和点的索引
#             cur_proposals_idx[:, 0] = cur_proposals_idx[:, 0] - cur_proposals_idx[0, 0]
#             cur_proposals_idx[:, 1] = cur_proposals_idx[:, 1] - proposals_idx_each_scene[0][1]
#             # M , num_point 表示有哪些点可能属于这个proposal对应的cls
#             mask_pred[cur_proposals_idx[:, 0], cur_proposals_idx[:, 1]] = 1
#
#         clu_point = torch.zeros((num_instances, num_points), dtype=torch.int, device='cuda')  # M, num_point
#         for j in range(num_instances):
#             # 调整索引
#             proposals_idx_each_scene[:, 0] = proposals_idx_each_scene[:, 0] - proposals_idx_each_scene[0, 0]
#             proposals_idx_each_scene[:, 1] = proposals_idx_each_scene[:, 1] - proposals_idx_each_scene[0, 1]
#             # M , num_point 表示有哪些点被group到这个proposal
#             clu_point[proposals_idx_each_scene[:, 0].long(), proposals_idx_each_scene[:, 1].long()] = 1
#
#         final_proposals = clu_point * mask_pred  # M ,num_point 最终结果，最终的proposal中有哪些点
#
#         final_proposals = final_proposals.cpu().numpy()
#         pred_bbox = np.zeros((num_instances, 6))  # M, 6 存储每个proposal的bbox的center和size
#
#         all_points_idx_end = all_points_idx_end + num_points
#         all_points_each_scene = all_points[all_points_idx_start:all_points_idx_end]
#         all_points_idx_start = all_points_idx_start + num_points
#
#         for j in range(num_instances):
#             idx = (final_proposals[j] == 1)
#             object_points = all_points_each_scene[idx].cpu().numpy()
#
#             if object_points.shape[0] != 0:
#                 max_corner = object_points.max(0)
#                 min_corner = object_points.min(0)
#             else:
#                 max_corner = np.zeros(3)
#                 min_corner = np.zeros(3)
#             center = (max_corner + min_corner) / 2
#             size = abs(max_corner - min_corner)
#             pred_bbox[j, 0:3] = center
#             pred_bbox[j, 3:6] = size
#
#         pred_box_corner_each_scene = get_3d_box_batch(pred_bbox[:, 3:6], np.zeros(num_instances),
#                                                       pred_bbox[:, 0:3])  # M, 8, 3
#         pred_box_corner_each_scene_pad = np.zeros((128, 8, 3))
#
#         pred_box_corner_each_scene_pad[0:pred_box_corner_each_scene.shape[0]] = pred_box_corner_each_scene  # 补全
#
#         pred_box_corner_each_scene_pad = torch.from_numpy(pred_box_corner_each_scene_pad).unsqueeze(
#             0).cuda()  # 1, 128, 8, 3
#
#         pred_center_each_scene_pad = np.zeros((128, 3))
#
#         pred_center_each_scene_pad[0:pred_box_corner_each_scene.shape[0]] = pred_bbox[:, 0:3]
#
#         pred_center_each_scene_pad = torch.from_numpy(pred_center_each_scene_pad).unsqueeze(
#             0).cuda()
#
#         if i == 0:
#             pred_box_corner = pred_box_corner_each_scene_pad.clone()
#             pred_box_center = pred_center_each_scene_pad.clone()
#         else:
#             pred_box_corner = torch.cat((pred_box_corner, pred_box_corner_each_scene_pad), dim=0)
#             pred_box_center = torch.cat((pred_box_center, pred_center_each_scene_pad), dim=0)
#
#     # only_for_vis_test(data_dict, pred_box_corner)
#
#     data_dict['bbox_corner'] = pred_box_corner
#     data_dict['bbox_center'] = pred_box_center
#     return data_dict


def bbox_pred_module_test(data_dict):
    cls_scores = data_dict["cls_scores"]
    mask_scores = data_dict["mask_scores"]
    proposals_idx = data_dict["proposals_idx"]
    proposal_each_scene = data_dict["proposal_each_scene"]
    all_points = data_dict['coords_float']  # 获取场景内的所有的point坐标

    num_points = data_dict["num_points"]  # 场景内point的数量
    num_instances = cls_scores.size(0)  # proposal的数量
    cls_scores = cls_scores.softmax(1)  # softmax分类得分
    max_cls_score, final_cls = cls_scores.max(1)  # M, 1 和 M, 1 每个proposal得到的分类最高分，以及属于哪个类

    mask_pred = torch.zeros((num_instances, num_points), dtype=torch.int, device='cuda')  # M, num_point
    for i in range(num_instances):
        cur_mask_scores = mask_scores[:, final_cls[i]]  # N, 1， N个点对第i个proposal的类的mask score
        mask_inds = cur_mask_scores > -0.5  # threshold 取mask高于阈值的点
        cur_proposals_idx = proposals_idx[mask_inds].long()
        mask_pred[
            cur_proposals_idx[:, 0], cur_proposals_idx[:, 1]] = 1  # M , num_point 表示有哪些点可能属于这个proposal对应的cls

    clu_point = torch.zeros((num_instances, num_points), dtype=torch.int, device='cuda')  # M, num_point
    for i in range(num_instances):
        clu_point[
            proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1  # M , num_point 表示有哪些点被group到这个proposal
    final_proposals = clu_point * mask_pred  # M ,num_point 最终结果，最终的proposal中有哪些点

    final_proposals = final_proposals.cpu().numpy()
    pred_bbox = np.zeros((num_instances, 6))  # M, 6 存储每个proposal的bbox的center和size
    for i in range(num_instances):
        idx = (final_proposals[i] == 1)
        object_points = data_dict['coords_float'][idx].cpu().numpy()
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
    pred_box_corner_pad = np.zeros((128, 8, 3))
    pred_box_corner_pad[0:pred_box_corner.shape[0]] = pred_box_corner  # 补全
    pred_box_corner_pad = torch.from_numpy(pred_box_corner_pad).unsqueeze(0).cuda()  # 1, 128, 8, 3

    data_dict['bbox_corner'] = pred_box_corner_pad

    return data_dict


class EdgeConv(MessagePassing):
    def __init__(self, in_size, out_size, aggregation="add"):
        super().__init__(aggr=aggregation)
        self.in_size = in_size
        self.out_size = out_size

        self.map_edge = nn.Sequential(
            nn.Linear(2 * in_size, out_size),
            nn.ReLU(),
            nn.Linear(out_size, out_size)
        )
        # self.map_node = nn.Sequential(
        #     nn.Linear(out_size, out_size),
        #     nn.ReLU()
        # )
        self.__explain__ = False  # 解释说明标志，默认为 False
        self.__edge_mask__ = None  # 边掩码，默认为 None

    def forward(self, x, edge_index):
        # x has shape [N, in_size]
        # edge_index has shape [2, E]

        self.__explain__ = True
        self.__edge_mask__ = torch.ones(edge_index.shape[1]).cuda()

        return self.propagate(edge_index, x=x)

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        r"""The initial call to start propagating messages.

        Args:
            adj (Tensor or SparseTensor): A :obj:`torch.LongTensor` or a
                :obj:`torch_sparse.SparseTensor` that defines the underlying
                graph connectivity/message passing flow.
                :obj:`edge_index` holds the indices of a general (sparse)
                assignment matrix of shape :obj:`[N, M]`.
                If :obj:`edge_index` is of type :obj:`torch.LongTensor`, its
                shape must be defined as :obj:`[2, num_messages]`, where
                messages from nodes in :obj:`edge_index[0]` are sent to
                nodes in :obj:`edge_index[1]`
                (in case :obj:`flow="source_to_target"`).
                If :obj:`edge_index` is of type
                :obj:`torch_sparse.SparseTensor`, its sparse indices
                :obj:`(row, col)` should relate to :obj:`row = edge_index[1]`
                and :obj:`col = edge_index[0]`.
                The major difference between both formats is that we need to
                input the *transposed* sparse adjacency matrix into
                :func:`propagate`.
            size (tuple, optional): The size :obj:`(N, M)` of the assignment
                matrix in case :obj:`edge_index` is a :obj:`LongTensor`.
                If set to :obj:`None`, the size will be automatically inferred
                and assumed to be quadratic.
                This argument is ignored in case :obj:`edge_index` is a
                :obj:`torch_sparse.SparseTensor`. (default: :obj:`None`)
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """
        size = self.__check_input__(edge_index, size)

        coll_dict = self.__collect__(self.__user_args__, edge_index, size,
                                     kwargs)

        msg_kwargs = self.inspector.distribute('message', coll_dict)
        message = self.message(**msg_kwargs)

        # For `GNNExplainer`, we require a separate message and aggregate
        # procedure since this allows us to inject the `edge_mask` into the
        # message passing computation scheme.
        if self.__explain__:
            edge_mask = self.__edge_mask__.sigmoid()

            # Some ops add self-loops to `edge_index`. We need to do the
            # same for `edge_mask` (but do not train those).
            if message.size(self.node_dim) != edge_mask.size(0):
                loop = edge_mask.new_ones(size[0])
                edge_mask = torch.cat([edge_mask, loop], dim=0)
            assert message.size(self.node_dim) == edge_mask.size(0)
            message = message * edge_mask.view([-1] + [1] * (message.dim() - 1))

        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        out = self.aggregate(message, **aggr_kwargs)

        update_kwargs = self.inspector.distribute('update', coll_dict)

        return self.update(out, **update_kwargs), message

    def message(self, x_i, x_j):
        # x_i has shape [E, in_size]
        # x_j has shape [E, in_size]
        edge = torch.cat([x_i, x_j - x_i], dim=1)  # edge has shape [E, 2 * in_size]
        # edge = torch.cat([x_i, x_j], dim=1)  # edge has shape [E, 2 * in_size]

        return self.map_edge(edge)

    def update(self, x_i):
        # x has shape [N, out_size]

        # return self.map_node(x_i)
        return x_i


class GraphModule(nn.Module):
    def __init__(self, in_size, out_size, num_layers, num_proposals, feat_size, num_locals,
                 query_mode="corner", graph_mode="graph_conv", return_edge=False, graph_aggr="add",
                 return_orientation=False, num_bins=6, return_distance=False):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size

        self.num_proposals = num_proposals
        self.feat_size = feat_size

        self.num_locals = num_locals
        self.query_mode = query_mode

        # graph layers
        self.graph_mode = graph_mode
        self.gc_layers = nn.ModuleList()

        for _ in range(num_layers):
            if graph_mode == "graph_conv":
                self.gc_layers.append(GCNConv(in_size, out_size))
            elif graph_mode == "edge_conv":
                self.gc_layers.append(EdgeConv(in_size, out_size, graph_aggr))
            else:
                raise ValueError("invalid graph mode, choices: [\"graph_conv\", \"edge_conv\"]")

        # graph edges
        self.return_edge = return_edge
        self.return_orientation = return_orientation
        self.return_distance = return_distance
        self.num_bins = num_bins

        # output final edges
        if self.return_orientation:
            assert self.graph_mode == "edge_conv"
            self.edge_layer = EdgeConv(in_size, out_size, graph_aggr)
            self.edge_predict = nn.Linear(out_size, num_bins + 1)

    def _nn_distance(self, pc1, pc2):
        """
        Input:
            pc1: (B,N,C) torch tensor
            pc2: (B,M,C) torch tensor

        Output:
            dist1: (B,N) torch float32 tensor
            idx1: (B,N) torch int64 tensor
            dist2: (B,M) torch float32 tensor
            idx2: (B,M) torch int64 tensor
        """

        N = pc1.shape[1]
        M = pc2.shape[1]
        pc1_expand_tile = pc1.unsqueeze(2).repeat(1, 1, M, 1)
        pc2_expand_tile = pc2.unsqueeze(1).repeat(1, N, 1, 1)
        pc_diff = pc1_expand_tile - pc2_expand_tile
        pc_dist = torch.sqrt(torch.sum(pc_diff ** 2, dim=-1) + 1e-8)  # (B,N,M)

        return pc_dist

    def _get_bbox_centers(self, corners):
        coord_min = torch.min(corners, dim=2)[0]  # batch_size, num_proposals, 3
        coord_max = torch.max(corners, dim=2)[0]  # batch_size, num_proposals, 3

        return (coord_min + coord_max) / 2

    def _query_locals(self, data_dict, target_ids, object_masks, include_self=True,
                      overlay_threshold=CONF.TRAIN.OVERLAID_THRESHOLD):
        corners = data_dict["bbox_corner"]  # batch_size, num_proposals, 8, 3

        centers = self._get_bbox_centers(corners)  # batch_size, num_proposals, 3
        batch_size, _, _ = centers.shape

        # decode target box info
        target_centers = torch.gather(centers, 1, target_ids.view(-1, 1, 1).repeat(1, 1, 3))  # batch_size, 1, 3
        target_corners = torch.gather(corners, 1,
                                      target_ids.view(-1, 1, 1, 1).repeat(1, 1, 8, 3))  # batch_size, 1, 8, 3

        # get the distance
        if self.query_mode == "center":
            pc_dist = self._nn_distance(target_centers, centers).squeeze(1)  # batch_size, num_proposals
        elif self.query_mode == "corner":
            pc_dist = self._nn_distance(target_corners.squeeze(1), centers)  # batch_size, 8, num_proposals
            pc_dist, _ = torch.min(pc_dist, dim=1)  # batch_size, num_proposals
        else:
            raise ValueError("invalid distance mode, choice: [\"center\", \"corner\"]")

        # mask out invalid objects
        pc_dist.masked_fill_(object_masks == 0, float('1e30'))  # distance to invalid objects: infinity

        # exclude overlaid boxes
        tar2neigbor_iou = box3d_iou_batch_tensor(
            target_corners.repeat(1, self.num_proposals, 1, 1).view(-1, 8, 3), corners.view(-1, 8, 3)).view(batch_size,
                                                                                                            self.num_proposals)  # batch_size, num_proposals
        overlaid_masks = tar2neigbor_iou >= overlay_threshold
        pc_dist.masked_fill_(overlaid_masks, float('1e30'))  # distance to overlaid objects: infinity

        # include the target objects themselves
        self_dist = 0 if include_self else float('1e30')
        self_masks = torch.zeros(batch_size, self.num_proposals).cuda()
        self_masks.scatter_(1, target_ids.view(-1, 1), 1)
        pc_dist.masked_fill_(self_masks == 1, self_dist)  # distance to themselves: 0 or infinity

        # get the top-k object ids
        _, topk_ids = torch.topk(pc_dist, self.num_locals, largest=False, dim=1)  # batch_size, num_locals

        # construct masks for the local context
        local_masks = torch.zeros(batch_size, self.num_proposals).cuda()
        local_masks.scatter_(1, topk_ids, 1)

        return local_masks

    def _create_adjacent_mat(self, data_dict, object_masks):
        batch_size, num_objects = object_masks.shape  # batch_size, num_proposals
        adjacent_mat = torch.zeros(batch_size, num_objects, num_objects).cuda()

        for obj_id in range(num_objects):
            target_ids = torch.LongTensor([obj_id for _ in range(batch_size)]).cuda()
            adjacent_entry = self._query_locals(data_dict, target_ids, object_masks,
                                                include_self=False)  # batch_size, num_objects
            adjacent_mat[:, obj_id] = adjacent_entry

        return adjacent_mat

    def _feed(self, graph):
        feat, edge = graph.x, graph.edge_index

        for layer in self.gc_layers:
            if self.graph_mode == "graph_conv":
                feat = layer(feat, edge)
                message = None
            elif self.graph_mode == "edge_conv":
                feat, message = layer(feat, edge)

        return feat, message

    def _add_relation_feat(self, data_dict, obj_feats, target_ids):
        rel_feats = data_dict["edge_feature"]  # batch_size, num_proposals, num_locals, feat_size
        batch_size = rel_feats.shape[0]

        rel_feats = torch.gather(rel_feats, 1,
                                 target_ids.view(batch_size, 1, 1, 1).repeat(1, 1, self.num_locals,
                                                                             self.feat_size)).squeeze(
            1)  # batch_size, num_locals, feat_size

        # new_obj_feats = torch.cat([obj_feats, rel_feats], dim=1) # batch_size, num_proposals + num_locals, feat_size

        # scatter the relation features to objects
        adjacent_mat = data_dict["adjacent_mat"]  # batch_size, num_proposals, num_proposals
        rel_indices = torch.gather(adjacent_mat, 1,
                                   target_ids.view(batch_size, 1, 1).repeat(1, 1, self.num_proposals)).squeeze(
            1)  # batch_size, num_proposals
        rel_masks = rel_indices.unsqueeze(-1).repeat(1, 1, self.feat_size) == 1  # batch_size, num_proposals, feat_size
        scattered_rel_feats = torch.zeros(obj_feats.shape).cuda().masked_scatter(rel_masks,
                                                                                 rel_feats)  # batch_size, num_proposals, feat_size
        rel_obj_feats = obj_feats + scattered_rel_feats
        # new_obj_feats = torch.cat([obj_feats, scattered_rel_feats], dim=-1)
        # new_obj_feats = self.map_rel(new_obj_feats)

        return rel_obj_feats

    def forward(self, data_dict):
        object_feats = data_dict["object_feats"].cuda()  # batch_size, num_proposals, feat_size
        object_masks = data_dict["object_mask"].cuda()  # batch_size, num_proposals
        select_feat_idx = data_dict["select_feat_idx"].cuda()

        batch_size, num_objects, _ = object_feats.shape
        adjacent_mat = self._create_adjacent_mat(data_dict, object_masks)  # batch_size, num_proposals, num_proposals

        new_obj_feats = torch.zeros(batch_size, num_objects, self.feat_size).cuda()
        enhanced_feats = torch.zeros(batch_size, self.feat_size).cuda()
        edge_indices = torch.zeros(batch_size, 2, num_objects * self.num_locals).cuda()
        edge_feats = torch.zeros(batch_size, num_objects, self.num_locals, self.out_size).cuda()
        edge_preds = torch.zeros(batch_size, num_objects * self.num_locals, self.num_bins + 1).cuda()
        num_sources = torch.zeros(batch_size).long().cuda()
        num_targets = torch.zeros(batch_size).long().cuda()

        for batch_id in range(batch_size):
            cur_select_feats_idx = data_dict['select_feat_idx'][batch_id]

            # valid object masks
            batch_object_masks = object_masks[batch_id]  # num_objects

            # create adjacent matric for this scene
            batch_adjacent_mat = adjacent_mat[batch_id]  # num_objects, num_objects
            batch_adjacent_mat = batch_adjacent_mat[batch_object_masks == 1, :][:,
                                 batch_object_masks == 1]  # num_valid_objects, num_valid_objects

            # initialize graph for this scene
            sparse_mat = coo_matrix(batch_adjacent_mat.detach().cpu().numpy())
            batch_edge_index, edge_attr = GUtils.from_scipy_sparse_matrix(sparse_mat)
            batch_obj_feats = object_feats[batch_id, batch_object_masks == 1].clone()  # num_valid_objects, in_size
            batch_graph = GData(x=batch_obj_feats, edge_index=batch_edge_index.cuda())

            # graph conv
            node_feat, edge_feat = self._feed(batch_graph)

            # output last edge
            if self.return_orientation:
                # output edge
                try:
                    num_src_objects = len(set(batch_edge_index[0].cpu().numpy()))
                    num_tar_objects = int(edge_feat.shape[0] / num_src_objects)

                    num_sources[batch_id] = num_src_objects
                    num_targets[batch_id] = num_tar_objects

                    edge_feat = edge_feat[
                                :num_src_objects * num_tar_objects]  # in case there are less than 10 neighbors
                    edge_feats[batch_id, :num_src_objects, :num_tar_objects] = edge_feat.view(num_src_objects,
                                                                                              num_tar_objects,
                                                                                              self.out_size)
                    edge_indices[batch_id, :, :num_src_objects * num_tar_objects] = batch_edge_index[:,
                                                                                    :num_src_objects * num_tar_objects]

                    _, edge_feat = self.edge_layer(node_feat, batch_edge_index.cuda())
                    edge_pred = self.edge_predict(edge_feat)
                    edge_preds[batch_id, :num_src_objects * num_tar_objects] = edge_pred

                except Exception:
                    print("error occurs when dealing with graph, skipping...")

            # skip connection
            output = batch_obj_feats + node_feat
            new_obj_feats[batch_id, batch_object_masks == 1] = output

            enhanced_feats[batch_id] = new_obj_feats[batch_id][cur_select_feats_idx]

        valid_mask = torch.zeros(batch_size, num_objects).bool().cuda()  # batch_size, num_proposals
        for batch_id in range(batch_size):
            cur_select_feats_idx = data_dict['select_feat_idx'][batch_id]
            valid_mask[batch_id] = adjacent_mat[batch_id][cur_select_feats_idx]

        data_dict["bbox_feature"] = new_obj_feats
        data_dict["adjacent_mat"] = adjacent_mat
        data_dict["edge_index"] = edge_indices
        data_dict["edge_feature"] = edge_feats
        data_dict["num_edge_source"] = num_sources
        data_dict["num_edge_target"] = num_targets
        data_dict["edge_orientations"] = edge_preds[:, :, :-1]
        data_dict["edge_distances"] = edge_preds[:, :, -1]
        data_dict["enhanced_feats"] = enhanced_feats
        data_dict["valid_mask"] = valid_mask
        # print(valid_mask)
        # print(object_masks)
        if self.return_orientation:
            rel_obj_feats = self._add_relation_feat(data_dict, new_obj_feats,
                                                    select_feat_idx)  # batch_size, num_proposals, feat_size
            data_dict["bbox_feature"] = rel_obj_feats

        return data_dict

    def forward_test(self, data_dict):
        object_feats = data_dict["object_feats"].cuda()  # 1, num_proposals, feat_size
        object_masks = data_dict["object_mask"].cuda()  # 1, num_proposals

        batch_size, num_objects, _ = object_feats.shape
        adjacent_mat = self._create_adjacent_mat(data_dict, object_masks)  # batch_size, num_proposals, num_proposals

        new_obj_feats = torch.zeros(batch_size, num_objects, self.feat_size).cuda()
        edge_indices = torch.zeros(batch_size, 2, num_objects * self.num_locals).cuda()
        edge_feats = torch.zeros(batch_size, num_objects, self.num_locals, self.out_size).cuda()
        edge_preds = torch.zeros(batch_size, num_objects * self.num_locals, self.num_bins + 1).cuda()
        num_sources = torch.zeros(batch_size).long().cuda()
        num_targets = torch.zeros(batch_size).long().cuda()

        for batch_id in range(batch_size):
            # valid object masks
            batch_object_masks = object_masks[batch_id]  # num_objects

            # create adjacent matric for this scene
            batch_adjacent_mat = adjacent_mat[batch_id]  # num_objects, num_objects
            batch_adjacent_mat = batch_adjacent_mat[batch_object_masks == 1, :][:,
                                 batch_object_masks == 1]  # num_valid_objects, num_valid_objects

            # initialize graph for this scene
            sparse_mat = coo_matrix(batch_adjacent_mat.detach().cpu().numpy())
            batch_edge_index, edge_attr = GUtils.from_scipy_sparse_matrix(sparse_mat)
            batch_obj_feats = object_feats[batch_id, batch_object_masks == 1].clone()  # num_valid_objects, in_size
            batch_graph = GData(x=batch_obj_feats, edge_index=batch_edge_index.cuda())

            # graph conv
            node_feat, edge_feat = self._feed(batch_graph)

            # output last edge
            if self.return_orientation:
                # output edge
                try:
                    num_src_objects = len(set(batch_edge_index[0].cpu().numpy()))
                    num_tar_objects = int(edge_feat.shape[0] / num_src_objects)

                    num_sources[batch_id] = num_src_objects
                    num_targets[batch_id] = num_tar_objects

                    edge_feat = edge_feat[
                                :num_src_objects * num_tar_objects]  # in case there are less than 10 neighbors
                    edge_feats[batch_id, :num_src_objects, :num_tar_objects] = edge_feat.view(num_src_objects,
                                                                                              num_tar_objects,
                                                                                              self.out_size)
                    edge_indices[batch_id, :, :num_src_objects * num_tar_objects] = batch_edge_index[:,
                                                                                    :num_src_objects * num_tar_objects]

                    _, edge_feat = self.edge_layer(node_feat, batch_edge_index.cuda())
                    edge_pred = self.edge_predict(edge_feat)
                    edge_preds[batch_id, :num_src_objects * num_tar_objects] = edge_pred

                except Exception:
                    print("error occurs when dealing with graph, skipping...")

            # skip connection
            output = batch_obj_feats + node_feat
            new_obj_feats[batch_id, batch_object_masks == 1] = output

        data_dict["bbox_feature"] = new_obj_feats  # 1, num_proposals, feat_size
        data_dict["adjacent_mat"] = adjacent_mat  # 1, num_proposals, num_proposals
        data_dict["edge_index"] = edge_indices
        data_dict["edge_feature"] = edge_feats
        data_dict["num_edge_source"] = num_sources
        data_dict["num_edge_target"] = num_targets
        data_dict["edge_orientations"] = edge_preds[:, :, :-1]
        data_dict["edge_distances"] = edge_preds[:, :, -1]

        if self.return_orientation:
            rel_obj_feats = torch.zeros(batch_size, num_objects, num_objects, self.feat_size).cuda()
            for i in range(num_objects):
                select_feat_idx = torch.tensor(i).long().cuda()
                rel_obj_feats[0][i] = self._add_relation_feat(data_dict, new_obj_feats, select_feat_idx)
            data_dict["rel_bbox_feature"] = rel_obj_feats  # 1, num_proposals, num_proposals, feat_size

        return data_dict
