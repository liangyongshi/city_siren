import os
import logging
import numpy as np
from sklearn.neighbors import NearestNeighbors
from plyfile import PlyData, PlyElement
import skimage.measure
import yaml
import configargparse
import time
import torch
import modules
import torch.nn.functional as F
import eval_dataio as ed
import sdf_meshing as sdf
import dataio
from torch.utils.data import DataLoader
import pandas as pd


class BaseOptions():
    def __init__(self):
        self.p = configargparse.ArgumentParser()
        self.initialized = False
    def initialize(self):
        self.p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')
        self.p.add_argument('--experiment_name', type=str, default='kitti_label_rec_fourier')

        # General training options
        self.p.add_argument('--checkpoint_path',
                    default='/home/shiyl/siren_gln10_sample/logs/gn5_sample_backup')

        self.p.add_argument('--model_type', type=str, default='sine',
                    help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')
        self.p.add_argument('--mode', type=str, default='mlp',
                    help='Options are "mlp" or "mlp"')
        self.p.add_argument('--resolution', type=int, default=256)
        self.p.add_argument('--num_encoding_functions', type=int, default=10)
        self.p.add_argument('--dataset_path', type=str, default='/home/shiyl/pyProject/dataset/odometry_velodyne/05/newfile/')
        self.p.add_argument('--log_path', type=str, default='logs/')
        self.p.add_argument('--expr_name', type=str, default='test')
        self.p.add_argument('--integtxt', type=str,
                    default='/data/shiyl/kitti/05/integtxt_4.txt')


        self.p.add_argument('--backup', type=str, default='kitti_label_backup_fourier')


        ####################################################################
        self.p.add_argument('--batch_size', type=int, default=150000)  # the minimum count of points is 4995 (10, 000195)
        self.p.add_argument('--fix_coordiante', type=bool, default=True)
        self.p.add_argument('--pemode', type=str, default='fourier')  ## nerf mlp fourier leafour
        ####################################################################
        self.p.add_argument('--incremental_sampling', type=bool, default=True)
        self.p.add_argument('--pre_sample_rate', type=float, default=0.75)
        ####################################################################
        self.p.add_argument('--ply_path', type=str)
        self.p.add_argument('--eval', type=str, default='iou') #iou cd
        self.p.add_argument('--pthpath', type=str,
                    default='/home/user/pyProject/experiment_results/fourier_5/g.pth')       
        self.initialized = True
        
    def parser(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.p.parse_args()
        return self.opt
        
        


config_file = os.path.join('semantic-kitti.yaml')
kitti_config = yaml.safe_load(open(config_file, 'r'))

# reconstruct inv_map and color_map, and the augmentation part is filled with 0
inv_map = kitti_config['learning_map_inv']
maxkey = max(inv_map.keys())
inv_map_lut = np.zeros((maxkey + 100), dtype=np.int32)
inv_map_lut[list(inv_map.keys())] = list(inv_map.values())

rev_map = kitti_config['learning_map_rev']
revkey = max(rev_map.keys())
rev_map_lut = np.zeros((revkey + 100), dtype=np.int32)
rev_map_lut[list(rev_map.keys())] = list(rev_map.values())

color_map = kitti_config['color_map']
maxkey = max(color_map.keys())
color_map_lut = np.zeros((maxkey + 100, 3), dtype=np.int32)
color_map_lut[list(color_map.keys())] = list(color_map.values())


class iouEval():
    def __init__(self, n_classes, ignore=None):
        # input classes number
        self.n_classes = n_classes

        # what to include and ignore from the means
        self.ignore = np.array(ignore, dtype=np.int64)
        self.include = np.array([
            n for n in range(self.n_classes) if n not in self.ignore
        ], dtype=np.int64)

        # reset the class counters
        self.reset()

    def num_classes(self):
        return self.n_classes

    def reset(self):
        self.conf_matrix = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)

    def addBatach(self, x, y):  # x = preds, y = targets
        # sizes should be matching
        x_row = x.reshape(-1)
        y_row = y.reshape(-1)

        # check if the shape is the same
        assert (x_row.shape == x_row.shape)

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
        return iou_mean, iou  # returns 'iou mean', 'iou per class' ALL CLASSES

    def getacc(self):
        tp, fp, fn = self.getStats()
        total_tp = tp.sum()
        total = tp[self.include].sum() + fp[self.include].sum() + 1e-15
        acc_mean = total_tp / total
        return acc_mean

    def get_confusion(self):
        conf = self.conf_matrix.copy()

        return conf


def get_test(datasets):
    index = 910
    class SDFDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = modules.semanticSIREN(type=opt.model_type, in_features=3, pemode=opt.pemode,
                                               num_encoding_functions=opt.num_encoding_functions)
            self.model.load_state_dict(torch.load(os.path.join(opt.checkpoint_path, str(index) + '.pth')))
            # self.model.load_state_dict(torch.load(os.path.join(opt.checkpoint_path, 'gl.pth')))
            self.model.cuda()

        def forward(self, coords):
            model_in = {'coords': coords}
            return self.model(model_in)

    sdf_decoder = SDFDecoder()


    sdf_value, labels = sdf.save_labels(sdf_decoder, datasets, )
    print(sdf_value.shape, labels.shape)
    return sdf_value, labels


def iou_out():
    model = modules.semanticSIREN(type=opt.model_type, in_features=3, pemode=opt.pemode,
                                  num_encoding_functions=opt.num_encoding_functions)

    first_frame = True

    pre_size = 0
    sdf_out = []
    labels_out = []
    labels_in = []
    index = 0
    
    train_dataset = ed.PointCloud(opt.integtxt, on_surface_points=opt.batch_size,
                                      fix_coordiante=opt.fix_coordiante
                                      )
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=16)
    for step, (model_input, gt) in enumerate(train_dataloader):
        label_in = gt['labels'].cpu().squeeze().detach().numpy()
        sdf_value, label_out = get_test(model_input)
        labels_in.append(label_in)
        # sdf_out.append(sdf_value)
        labels_out.append(label_out)    


    labels_in = list(map(lambda x: x + 1, labels_in))
    labels_out = list(map(lambda x: x + 1, labels_out))
    labels_in = np.array(labels_in).reshape(1, -1).squeeze().astype(int)
    labels_out = np.array(labels_out).reshape(1, -1).squeeze().astype(int)

    for i in range(len(labels_in)):
        if labels_in[i] == 256:
            labels_in[i] = 0


    print("==============Statics===================")
    print("labels_in_shape: ", labels_in.shape, "labels_out_shape: ", labels_out.shape)

    NUM_CLASS_COMPLET = 20
    test = iouEval(NUM_CLASS_COMPLET, [])
    test.reset()

    test.addBatach(labels_in, labels_out)

    confusion_map = test.get_confusion()
    conf = confusion_map.copy()
    conf = np.delete(conf, 8, axis=1)
    conf = np.delete(conf, 8, axis=0)
    conf = np.delete(conf, 7, axis=1)
    conf = np.delete(conf, 7, axis=0)
    conf = np.delete(conf, 0, axis=1)
    conf = np.delete(conf, 0, axis=0)
    tp = np.diag(conf)
    fp = conf.sum(axis=1) - tp
    fn = conf.sum(axis=0) - tp
    intersection = tp
    union = tp + fp + fn + 1e-15
    iou = intersection / union
    n_classes = 17
    include = np.array([
        n for n in range(n_classes)
    ], dtype=np.int64)
    iou_mean = (intersection[include] / union[include]).mean()
    total_tp = tp.sum()
    total = tp[include].sum() + fp[include].sum() + 1e-15
    acc_mean = total_tp / total
    acc_mean
    print("=============Evaluations================")
    print("mIOU: ", iou_mean, "IOU: ", iou)
    print("Accuracy = ", acc_mean)
    # print("============Confusion_Map===============")
    # print(confusion_map)


if __name__ == '__main__':
    opt = BaseOptions().parser()
    if opt.eval == 'iou':
        iou_out()
    else:
        print('No such evaluate option!')


