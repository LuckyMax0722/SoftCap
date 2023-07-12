from lib.config import CONF
import functools
import os
import os.path as osp
import pickle
import shutil
import tempfile

import torch
from torch import distributed as dist

import numpy as np
import multiprocessing as mp

def decode_caption(raw_caption, idx2word):
    decoded = ["sos"]
    for token_idx in raw_caption:
        token_idx = token_idx.item()
        token = idx2word[str(token_idx)]
        decoded.append(token)
        if token == "eos": break

    if "eos" not in decoded: decoded.append("eos")
    decoded = " ".join(decoded)

    return decoded


def check_candidates(corpus, candidates):
    placeholder = "sos eos"
    corpus_keys = list(corpus.keys())
    candidate_keys = list(candidates.keys())
    missing_keys = [key for key in corpus_keys if key not in candidate_keys]

    if len(missing_keys) != 0:
        for key in missing_keys:
            candidates[key] = [placeholder]

    return candidates


def organize_candidates(corpus, candidates):
    new_candidates = {}
    for key in corpus.keys():
        new_candidates[key] = candidates[key]

    return new_candidates


def prepare_corpus(raw_data, max_len=CONF.TRAIN.MAX_DES_LEN):
    corpus = {}
    for data in raw_data:
        scene_id = data["scene_id"]
        object_id = data["object_id"]
        object_name = data["object_name"]
        token = data["token"][:max_len]
        description = " ".join(token)

        # add start and end token
        description = "sos " + description
        description += " eos"

        key = "{}|{}|{}".format(scene_id, object_id, object_name)

        if key not in corpus:
            corpus[key] = []

        corpus[key].append(description)

    return corpus


def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN,), 32, dtype=torch.uint8, device='cuda')
        if rank == 0:
            os.makedirs('.dist_test', exist_ok=True)
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        # dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        os.makedirs(tmpdir, exist_ok=True)
    # dump the part result to the dir
    pickle.dump(result_part, open(osp.join(tmpdir, f'part_{rank}.pkl'), 'wb'))
    # dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(pickle.load(open(part_file, 'rb')))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def save_single_instance(root, scan_id, insts, nyu_id=None):
    f = open(osp.join(root, f'{scan_id}.txt'), 'w')
    os.makedirs(osp.join(root, 'predicted_masks'), exist_ok=True)
    for i, inst in enumerate(insts):
        assert scan_id == inst['scan_id']
        label_id = inst['label_id']
        # scannet dataset use nyu_id for evaluation
        if nyu_id is not None:
            label_id = nyu_id[label_id - 1]
        conf = inst['conf']
        f.write(f'predicted_masks/{scan_id}_{i:03d}.txt {label_id} {conf:.4f}\n')
        mask_path = osp.join(root, 'predicted_masks', f'{scan_id}_{i:03d}.txt')
        mask = rle_decode(inst['pred_mask'])
        np.savetxt(mask_path, mask, fmt='%d')
    print('save pred_instance for', scan_id)
    f.close()


def save_pred_instances(root, name, scan_ids, pred_insts, nyu_id=None):
    root = osp.join(root, name)
    os.makedirs(root, exist_ok=True)
    roots = [root] * len(scan_ids)
    nyu_ids = [nyu_id] * len(scan_ids)
    pool = mp.Pool()
    pool.starmap(save_single_instance, zip(roots, scan_ids, pred_insts, nyu_ids))
    pool.close()
    pool.join()


def save_gt_instance(path, gt_inst, nyu_id=None):
    if nyu_id is not None:
        sem = gt_inst // 1000
        ignore = sem == 0
        ins = gt_inst % 1000
        nyu_id = np.array(nyu_id)
        sem = nyu_id[sem - 1]
        sem[ignore] = 0
        gt_inst = sem * 1000 + ins
    np.savetxt(path, gt_inst, fmt='%d')


def save_gt_instances(root, name, scan_ids, gt_insts, nyu_id=None):
    root = osp.join(root, name)
    os.makedirs(root, exist_ok=True)
    paths = [osp.join(root, f'{i}.txt') for i in scan_ids]
    pool = mp.Pool()
    nyu_ids = [nyu_id] * len(scan_ids)
    pool.starmap(save_gt_instance, zip(paths, gt_insts, nyu_ids))
    pool.close()
    pool.join()

def rle_encode(mask):
    """Encode RLE (Run-length-encode) from 1D binary mask.

    Args:
        mask (np.ndarray): 1D binary mask
    Returns:
        rle (dict): encoded RLE
    """
    length = mask.shape[0]
    mask = np.concatenate([[0], mask, [0]])
    runs = np.where(mask[1:] != mask[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    counts = ' '.join(str(x) for x in runs)
    rle = dict(length=length, counts=counts)
    return rle


def rle_decode(rle):
    """Decode rle to get binary mask.

    Args:
        rle (dict): rle of encoded mask
    Returns:
        mask (np.ndarray): decoded mask
    """
    length = rle['length']
    counts = rle['counts']
    s = counts.split()
    starts, nums = [np.asarray(x, dtype=np.int32) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + nums
    mask = np.zeros(length, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask

