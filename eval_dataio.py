import os
import numpy as np
import torch
from torch.utils.data import Dataset
import open3d as o3d

VALID_CLASS_LABELS = {
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
  255: 0,   # "off-surface" 
}

class PointCloud(Dataset):
    def __init__(self, pointcloud_path, on_surface_points,fix_coordiante=False):
        super().__init__()        
        self.fix_coordiante=fix_coordiante
        print("Loading point cloud")
        point_cloud = np.genfromtxt(pointcloud_path)
        print("Finished loading point cloud")
        coords = point_cloud[:, :3]  ##默认是深拷贝
        coords_temp = (point_cloud[:, :3]).copy() ##必须是浅拷贝，否则后面求最大最小值有误
        self.labels=point_cloud[:, 3]
    
        
        if(self.fix_coordiante==True):
            print("fix global coordiante")
            mean=np.array([[132.59720398,-10.72106049,1.8637712]])  ### for kitti_05_label
            coords -= mean
            coord_max=311.13929249
            coord_min=-241.77887551
                     
        self.coords = (coords - coord_min) / (coord_max - coord_min)
        self.coords -= 0.5
        self.coords *= 2.             ###normalize the point cloud data into [-1,1]

        self.on_surface_points = on_surface_points       
        maxkey = max(VALID_CLASS_LABELS.keys())
        remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
        remap_lut[list(VALID_CLASS_LABELS.keys())] = list(VALID_CLASS_LABELS.values())
        remap_lut=remap_lut-1
        remap_lut[remap_lut==-1]=255
        self.remap_lut = remap_lut
        print("dataio init finished")

    def __len__(self):
        return self.coords.shape[0] // self.on_surface_points

    def __getitem__(self, idx):
        
        point_cloud_size = self.coords.shape[0]
        total_samples = self.on_surface_points 
        rand_idcs = np.random.choice(point_cloud_size, size=self.on_surface_points)
        on_surface_coords = self.coords[rand_idcs, :]     
        sdf = np.zeros((total_samples, 1))  # on-surface = 0
        on_surface_labels = self.labels[rand_idcs]


        coords = on_surface_coords
        labels = on_surface_labels.astype(np.int)
        labels = (self.remap_lut[labels]).astype(np.long)
        
        return {'coords': torch.from_numpy(coords).float()}, {'sdf': torch.from_numpy(sdf).float(),                                                           
                                                              'labels': torch.from_numpy(labels)}
