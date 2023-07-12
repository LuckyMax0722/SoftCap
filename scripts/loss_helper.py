import torch
import torch.nn as nn
import numpy as np
import sys
import os


def radian_to_label(radians, num_bins=6):
    """
        convert radians to labels

        Arguments:
            radians: a tensor representing the rotation radians, (batch_size)
            radians: a binary tensor representing the valid masks, (batch_size)
            num_bins: number of bins for discretizing the rotation degrees

        Return:
            labels: a long tensor representing the discretized rotation degree classes, (batch_size)
    """

    boundaries = torch.arange(np.pi / num_bins, np.pi - 1e-8, np.pi / num_bins).cuda()
    labels = torch.bucketize(radians, boundaries)

    return labels


def compute_node_orientation_loss(data_dict, num_bins=6):
    object_assignment = data_dict["object_assignment"]
    edge_indices = data_dict["edge_index"]
    edge_preds = data_dict["edge_orientations"]
    num_sources = data_dict["num_edge_source"]
    num_targets = data_dict["num_edge_target"]
    batch_size, num_proposals = object_assignment.shape

    object_rotation_matrices = torch.gather(
        data_dict["scene_object_rotations"],
        1,
        object_assignment.view(batch_size, num_proposals, 1, 1).repeat(1, 1, 3, 3)
    )  # batch_size, num_proposals, 3, 3
    object_rotation_masks = torch.gather(
        data_dict["scene_object_rotation_masks"],
        1,
        object_assignment
    )  # batch_size, num_proposals

    preds = []
    labels = []
    masks = []
    for batch_id in range(batch_size):
        batch_rotations = object_rotation_matrices[batch_id]  # num_proposals, 3, 3
        batch_rotation_masks = object_rotation_masks[batch_id]  # num_proposals

        batch_num_sources = num_sources[batch_id]
        batch_num_targets = num_targets[batch_id]
        batch_edge_indices = edge_indices[batch_id, :batch_num_sources * batch_num_targets]

        source_indices = edge_indices[batch_id, 0, :batch_num_sources * batch_num_targets].long()
        target_indices = edge_indices[batch_id, 1, :batch_num_sources * batch_num_targets].long()

        source_rot = torch.index_select(batch_rotations, 0, source_indices)
        target_rot = torch.index_select(batch_rotations, 0, target_indices)

        relative_rot = torch.matmul(source_rot, target_rot.transpose(2, 1))
        relative_rot = torch.acos(
            torch.clamp(0.5 * (torch.diagonal(relative_rot, dim1=-2, dim2=-1).sum(-1) - 1), -1, 1))
        assert torch.isfinite(relative_rot).sum() == source_indices.shape[0]

        source_masks = torch.index_select(batch_rotation_masks, 0, source_indices)
        target_masks = torch.index_select(batch_rotation_masks, 0, target_indices)
        batch_edge_masks = source_masks * target_masks

        batch_edge_labels = radian_to_label(relative_rot, num_bins)
        batch_edge_preds = edge_preds[batch_id, :batch_num_sources * batch_num_targets]

        preds.append(batch_edge_preds)
        labels.append(batch_edge_labels)
        masks.append(batch_edge_masks)

    # aggregate
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    masks = torch.cat(masks, dim=0)

    criterion = nn.CrossEntropyLoss(reduction="none")
    loss = criterion(preds, labels)
    loss = (loss * masks).sum() / (masks.sum() + 1e-8)

    preds = preds.argmax(-1)
    acc = (preds[masks == 1] == labels[masks == 1]).sum().float() / (masks.sum().float() + 1e-8)

    return loss, acc


def compute_node_distance_loss(data_dict):
    gt_center = data_dict["center_label"][:, :, 0:3]
    object_assignment = data_dict["object_assignment"]

    gt_center = torch.gather(gt_center, 1, object_assignment.unsqueeze(-1).repeat(1, 1, 3))
    batch_size, _, _ = gt_center.shape

    edge_indices = data_dict["edge_index"]
    edge_preds = data_dict["edge_distances"]
    num_sources = data_dict["num_edge_source"]
    num_targets = data_dict["num_edge_target"]

    preds = []
    labels = []
    for batch_id in range(batch_size):
        batch_gt_center = gt_center[batch_id]

        batch_num_sources = num_sources[batch_id]
        batch_num_targets = num_targets[batch_id]
        batch_edge_indices = edge_indices[batch_id, :batch_num_sources * batch_num_targets]

        source_indices = edge_indices[batch_id, 0, :batch_num_sources * batch_num_targets].long()
        target_indices = edge_indices[batch_id, 1, :batch_num_sources * batch_num_targets].long()

        source_centers = torch.index_select(batch_gt_center, 0, source_indices)
        target_centers = torch.index_select(batch_gt_center, 0, target_indices)

        batch_edge_labels = torch.norm(source_centers - target_centers, dim=1)
        batch_edge_preds = edge_preds[batch_id, :batch_num_sources * batch_num_targets]

        preds.append(batch_edge_preds)
        labels.append(batch_edge_labels)

    # aggregate
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)

    criterion = nn.MSELoss()
    loss = criterion(preds, labels)

    return loss