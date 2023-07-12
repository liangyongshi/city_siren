"""
@Author: Pengfei Li
@File: utils_eval.py
@Description: 
@Date: 2021/08/06
"""

import os
import sys
import numpy as np
import torch
import yaml
from sklearn.neighbors import NearestNeighbors

import modules


class iouEval:
  def __init__(self, n_classes, ignore=None):
    # classes
    self.n_classes = n_classes

    # What to include and ignore from the means
    self.ignore = np.array(ignore, dtype=np.int64)
    self.include = np.array(
        [n for n in range(self.n_classes) if n not in self.ignore], dtype=np.int64)
    # print("[IOU EVAL] IGNORE: ", self.ignore)
    # print("[IOU EVAL] INCLUDE: ", self.include)

    # reset the class counters
    self.reset()

  def num_classes(self):
    return self.n_classes

  def reset(self):
    self.conf_matrix = np.zeros((self.n_classes,
                                 self.n_classes),
                                dtype=np.int64)

  def addBatch(self, x, y):  # x=preds, y=targets
    # sizes should be matching
    x_row = x.reshape(-1)  # de-batchify
    y_row = y.reshape(-1)  # de-batchify

    # check
    assert(x_row.shape == x_row.shape)

    # create indexes
    idxs = tuple(np.stack((x_row, y_row), axis=0))

    # make confusion matrix (cols = gt, rows = pred)
    np.add.at(self.conf_matrix, idxs, 1)

  def getStats(self):
    # remove fp from confusion on the ignore classes cols
    conf = self.conf_matrix.copy()
    conf[:, self.ignore] = 0

    # get the clean stats
    tp = np.diag(conf)
    fp = conf.sum(axis=1) - tp
    fn = conf.sum(axis=0) - tp
    return tp, fp, fn

  def getIoU(self):
    tp, fp, fn = self.getStats()
    intersection = tp
    union = tp + fp + fn + 1e-15
    iou = intersection / union
    iou_mean = (intersection[self.include] / union[self.include]).mean()
    return iou_mean, iou  # returns "iou mean", "iou per class" ALL CLASSES

  def getacc(self):
    tp, fp, fn = self.getStats()
    total_tp = tp.sum()
    total = tp[self.include].sum() + fp[self.include].sum() + 1e-15
    acc_mean = total_tp / total
    return acc_mean  # returns "acc mean"
    
  def get_confusion(self):
    return self.conf_matrix.copy()


def unpack(compressed):
    ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
    uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
    uncompressed[::8] = compressed[:] >> 7 & 1
    uncompressed[1::8] = compressed[:] >> 6 & 1
    uncompressed[2::8] = compressed[:] >> 5 & 1
    uncompressed[3::8] = compressed[:] >> 4 & 1
    uncompressed[4::8] = compressed[:] >> 3 & 1
    uncompressed[5::8] = compressed[:] >> 2 & 1
    uncompressed[6::8] = compressed[:] >> 1 & 1
    uncompressed[7::8] = compressed[:] & 1

    return uncompressed


def get_eval_mask(labels, invalid_voxels):
  """
  Ignore labels set to 255 and invalid voxels (the ones never hit by a laser ray, probed using ray tracing)
  :param labels: input ground truth voxels
  :param invalid_voxels: voxels ignored during evaluation since the lie beyond the scene that was captured by the laser
  :return: boolean mask to subsample the voxels to evaluate
  """
  masks = np.ones_like(labels, dtype=np.bool)
  masks[labels == 255] = False
  masks[invalid_voxels == 1] = False

  return masks


def eval_cd(pred, gt, masks):
    '''pred gt masks are all of size 256,256,32'''
    pred[masks == False] = 0
    gt[masks == False] = 0

    pred_xyz = np.transpose(pred.nonzero())
    gt_xyz = np.transpose(gt.nonzero())

    cd1 = 0
    neigh = NearestNeighbors(n_neighbors=1, radius=100.0)
    neigh.fit(gt_xyz)
    dist, indexes = neigh.kneighbors(pred_xyz, return_distance=True)
    cd1 = dist.mean()

    cd2 = 0
    neigh = NearestNeighbors(n_neighbors=1, radius=100.0)
    neigh.fit(pred_xyz)
    dist, indexes = neigh.kneighbors(gt_xyz, return_distance=True)
    cd2 = dist.mean()

    return (cd1 + cd2)*0.2


def get_discrete_sdf(model, N=256, max_batch=64 ** 3):
    '''get discrete sdf from model, store it in the file'''
    model.eval()
    
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() // N) % N
    samples[:, 0] = ((overall_index.long() // N) // N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()
        model_in = {'coords': sample_subset}

        samples[head : min(head + max_batch, num_samples), 3] = (
            model(model_in)['model_out']
            .squeeze()#.squeeze(1)
            .detach()
            .cpu()
        )
        head += max_batch

    sdf_values = samples[:, 3].numpy()

    return sdf_values


def eval_model(model, gt_label_path, N=256):
    
    SCALE = [256,256,32]
    NUM_CLASS_COMPLET = 20

    complet_evaluator = iouEval(NUM_CLASS_COMPLET, [])

    config_file = os.path.join('semantic-kitti.yaml')
    kitti_config = yaml.safe_load(open(config_file, 'r'))
    remapdict = kitti_config["learning_map"]
    maxkey = max(remapdict.keys())
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut[list(remapdict.keys())] = list(remapdict.values())
    remap_lut[remap_lut == 0] = 255  # map 0 to 'invalid'
    remap_lut[0] = 0  # only 'empty' stays 'empty'.

    # ground truth
    label_path = gt_label_path
    invalid_path = gt_label_path.replace('.label','.invalid')
    label = np.fromfile(label_path, dtype=np.uint16).reshape(SCALE)
    label = remap_lut[label]
    invalid = unpack(np.fromfile(invalid_path, dtype=np.uint8)).reshape(SCALE)

    masks = get_eval_mask(label, invalid)

    label_iou = label[masks]

    # predicted results
    sdf_values = get_discrete_sdf(model, N).reshape(N, N, N)[:,:,:32]

    zero_array = np.zeros(SCALE)
    one_array = np.ones(SCALE)

    threshold_list = [0.03,0.02,0.015,0.01,0.008,0.006]
    eval_result = {}
    eval_result['threshold'] = threshold_list
    eval_result['iou'] = []
    eval_result['cd'] = []
    for threshold in threshold_list:
        pred = np.where(abs(sdf_values) < threshold, one_array, zero_array)
        pred_iou = pred[masks]

        # calculate IoU and CD
        complet_evaluator.reset()
        complet_evaluator.addBatch(pred_iou.astype(int), label_iou.astype(int))
        conf = complet_evaluator.get_confusion()
        acc_cmpltn = (np.sum(conf[1:, 1:])) / (np.sum(conf) - conf[0, 0])
        cd_value = eval_cd(pred, label, masks)

        eval_result['iou'].append(acc_cmpltn * 100)
        eval_result['cd'].append(cd_value)

    return eval_result