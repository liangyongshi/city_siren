'''From the DeepSDF repository https://github.com/facebookresearch/DeepSDF
'''
#!/usr/bin/env python3

import logging
import numpy as np
import time
import torch

from IPython import embed

LABELS_TO_CLASS = {
  10: 1,     # "car"
  11: 2,     # "bicycle"
  13: 5,     # "bus" mapped to "other-vehicle" --------------------------mapped
  15: 3,     # "motorcycle"
  16: 5,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
  18: 4,     # "truck"
  20: 5,     # "other-vehicle"
  30: 6,     # "person"
  31: 7,     # "bicyclist"
  32: 8,     # "motorcyclist"
  40: 9,     # "road"
  44: 10,    # "parking"
  48: 11,    # "sidewalk"
  49: 12,    # "other-ground"
  50: 13,    # "building"
  51: 14,    # "fence"
  52: 0,     # "other-structure" mapped to "unlabeled" ------------------mapped
  60: 9,     # "lane-marking" to "road" ---------------------------------mapped
  70: 15,    # "vegetation"
  71: 16,    # "trunk"
  72: 17,    # "terrain"
  80: 18,    # "pole"
  81: 19,    # "traffic-sign"
  99: 0,     # "other-object" to "unlabeled" ----------------------------mapped
  255: 255   # "noise"
}

COLOR_MAP = {
  10: [245, 150, 100],
  11: [245, 230, 100],
  13: [250, 80, 100],
  15: [150, 60, 30],
  16: [255, 0, 0],
  18: [180, 30, 80],
  20: [255, 0, 0],
  30: [30, 30, 255],
  31: [200, 40, 255],
  32: [90, 30, 150],
  40: [255, 0, 255],
  44: [255, 150, 255],
  48: [75, 0, 75],
  49: [75, 0, 175],
  50: [0, 200, 255],
  51: [50, 120, 255],
  52: [0, 150, 255],
  60: [170, 255, 150],
  70: [0, 175, 0],
  71: [0, 60, 135],
  72: [80, 240, 150],
  80: [150, 240, 255],
  81: [0, 0, 255],
  99: [255, 255, 50],
  255: [0, 0, 0]
}   #其中两个值是随便设的

def create_picture(
    coords, decoder, filename, N=256, max_batch=64 ** 3, offset=None, scale=None
):
    start = time.time()

    decoder.eval()

    coords = torch.tensor(coords)
    num_samples = coords.shape[0]
    labels = torch.zeros(coords.shape[0])
    labels_v = torch.zeros(coords.shape[0])

    head = 0

    while head < num_samples:
        print(head)
        sample_subset = coords[head : min(head + max_batch, num_samples)].cuda()
        model_out = decoder(sample_subset)

        labels[head : min(head + max_batch, num_samples)] = (
            model_out['label_out'].squeeze().detach().data.max(1)[1].cpu()
        ) 
        
        head += max_batch

    end = time.time()
    print("sampling takes: %f" % (end - start))
    labels = np.array(labels).astype(np.int)
    labels_v = np.array(labels_v).astype(np.int)
    # np.savetxt('/data/hdd01/yuhang/semantic_siren/experiment_scripts/logs/try.txt', labels)
    
    colors = np.zeros([labels.shape[0], 3])
    for i in range(len(labels)):
        labels_v[i] = int(LABELS_TO_CLASS[labels[i]])
        colors[i] = COLOR_MAP[labels_v[i]]
    points = np.concatenate([coords[:,:3], colors], axis=-1)

    np.savetxt(filename + '.txt', points)

