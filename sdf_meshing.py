'''From the DeepSDF repository https://github.com/facebookresearch/DeepSDF
'''
#!/usr/bin/env python3

import logging
import numpy as np
import plyfile
import skimage.measure
import time
import torch
import yaml
import os
from sklearn.neighbors import NearestNeighbors
from IPython import embed

config_file = os.path.join('semantic-kitti.yaml')
kitti_config = yaml.safe_load(open(config_file, 'r'))

inv_map = kitti_config['learning_map_inv']
maxkey = max(inv_map.keys())
inv_map_lut = np.zeros((maxkey + 100), dtype=np.int32)
inv_map_lut[list(inv_map.keys())] = list(inv_map.values())

color_map = kitti_config['color_map']
maxkey = max(color_map.keys())
color_map_lut = np.zeros((maxkey + 100, 3), dtype=np.int32)
color_map_lut[list(color_map.keys())] = list(color_map.values())

#########kitti的数据集归一化后z轴占数据很少，会有很多无效计算，增加分辨率，而z轴只用很小一部分，同样的计算资源能够支持地图更高的分辨率#######
high_resolution=True  ##用于Kitti数据集，x、y方向的尺度远大于z方向的尺度，所以
high_resolution_voxels=2000 ##2000 3000
z_voxels=140  ##180 180
center_offset=830### 830 1260   z方 向尺度（-2.0～0.3）相对于x、y方向来说基本就在0附近，所以在中心1500的附近取值,1340
#################

LABELS_TO_CLASS = {
  0: 0,      # "unlabeled", and others ignored
  1: 10,     # "car"
  2: 11,     # "bicycle"
  3: 15,     # "motorcycle"
  4: 18,     # "truck"
  5: 20,     # "other-vehicle"
  6: 30,     # "person"
  7: 31,     # "bicyclist"
  8: 32,     # "motorcyclist"
  9: 40,     # "road"
  10: 44,    # "parking"
  11: 48,    # "sidewalk"
  12: 49,    # "other-ground"
  13: 50,    # "building"
  14: 51,    # "fence"
  15: 70,    # "vegetation"
  16: 71,    # "trunk"
  17: 72,    # "terrain"
  18: 80,    # "pole"
  19: 81,    # "traffic-sign"
  20: 0,     #'off-surface'
}

COLOR_MAP = {
  0 : (255, 255, 255),
  1 : (0, 0, 255),
  10: (245, 150, 100),
  11: (245, 230, 100),
  13: (250, 80, 100),
  15: (150, 60, 30),
  16: (255, 0, 0),
  18: (180, 30, 80),
  20: (255, 0, 0),
  30: (30, 30, 255),
  31: (200, 40, 255),
  32: (90, 30, 150),
  40: (255, 0, 255),
  44: (255, 150, 255),
  48: (75, 0, 75),
  49: (75, 0, 175),
  50: (0, 200, 255),
  51: (50, 120, 255),
  52: (0, 150, 255),
  60: (170, 255, 150),
  70: (0, 175, 0),
  71: (0, 60, 135),
  72: (80, 240, 150),
  80: (150, 240, 255),
  81: (0, 0, 255),
  99: (255, 255, 50),
}
def create_mesh(
    decoder, filename, N=256, max_batch=64 ** 3, offset=None,scale=None
):
    start = time.time()
    ply_filename = filename
    
    
    decoder.eval()
    print("decoder.eval time:",time.time()-start)

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    
    
    if (high_resolution):
        voxel_size = 2.0 / (high_resolution_voxels - 1)
        overall_index = torch.arange(0, high_resolution_voxels*high_resolution_voxels*z_voxels, 1, out=torch.LongTensor())
        samples = torch.zeros(high_resolution_voxels*high_resolution_voxels*z_voxels, 4)
        labels=torch.zeros(high_resolution_voxels*high_resolution_voxels*z_voxels)
        samples[:, 2] = overall_index % (z_voxels)+center_offset
        samples[:, 1] = (overall_index.long() // z_voxels) % (high_resolution_voxels)
        samples[:, 0] = ((overall_index.long() // z_voxels) // (high_resolution_voxels)) % (high_resolution_voxels)
        samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]  
        samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
        samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]  
        num_samples = high_resolution_voxels*high_resolution_voxels*z_voxels  
    else:
        voxel_size = 2.0 / (N - 1)
        overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
        samples = torch.zeros(N ** 3, 4)
        labels = torch.zeros(N ** 3)
        samples[:, 2] = overall_index % N
        samples[:, 1] = (overall_index.long() // N) % N
        samples[:, 0] = ((overall_index.long() // N) // N) % N
        samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]  
        samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
        samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]
        num_samples = N ** 3
     
    samples.requires_grad = False      

    head = 0

    while head < num_samples:
        print(head)
        dim=3
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:dim].cuda()
        model_out= decoder(sample_subset)
        
        samples[head : min(head + max_batch, num_samples), dim] = (
            model_out['sdf_out'].squeeze().detach().cpu())      
        
        
        labels[head : min(head + max_batch, num_samples)]=(
            model_out['label_out'].squeeze().detach().data.max(1)[1].cpu()) 
        
        head += max_batch

    sdf_values = samples[:, dim]
    
    if (high_resolution):
        sdf_values = sdf_values.reshape(high_resolution_voxels, high_resolution_voxels, z_voxels)
        np.save("field_slice20",sdf_values[:,:,20:21])
        np.save("field_slice30",sdf_values[:,:,30:31])    
        np.save("field_slice70",sdf_values[:,:,20:21])
        np.save("field_slice80",sdf_values[:,:,30:31])     
        labels = np.array(labels.reshape(high_resolution_voxels, high_resolution_voxels, z_voxels)).astype(np.int)
    else:       
        sdf_values = sdf_values.reshape(N, N, N)
        labels = np.array(labels.reshape(N, N, N)).astype(np.int)

    end = time.time()
    print("sampling takes: %f" % (end - start))
    

    # convert_sdf_samples_to_ply(
    #     sdf_values.data.cpu(),
    #     voxel_origin,
    #     voxel_size,
    #     ply_filename + ".ply",
    #     offset,
    #     scale,
    # )
    convert_sdf_label_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        labels,
        ply_filename + ".ply",
        offset,
        scale,
    )
def convert_sdf_label_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    labels,
    ply_filename_out,   
    offset=None,
    scale=None,
    ):
    '''
    Convert sdf samples to .ply with semantic infomation

    :param sdf_values: a numpy array of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    '''
    start_time = time.time()
    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()
    
    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    try:
        verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
            numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
        )
    except:
        pass
    

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset
    

    colors = np.zeros_like(verts)   
    num_verts = verts.shape[0]
    num_faces = faces.shape[0]
    

    if labels is not None:
        verts_tuple = np.zeros((num_verts,), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])        
        ######比较简单meshpoints label方法#######
        index = tuple(np.transpose((verts // voxel_size).astype(np.int)))
        v_label = labels[index]
        #####################       
        #####给每个verts找空间位置最近的label#######################
        # label_coords = np.transpose(labels.nonzero())*voxel_size-1.0    
        # k_neigh = NearestNeighbors(n_neighbors=1)
        # # embed()
        # k_neigh.fit(label_coords)   
        # # embed() 
        # index = k_neigh.kneighbors(mesh_points, return_distance=False).squeeze() 
        # coords=(label_coords[index]+1.0)/voxel_size        
        # v_label = labels[tuple(np.transpose(coords).astype(np.int))] 
        
        #########################################################             
        for i in range(0, num_verts):   
            label=LABELS_TO_CLASS[v_label[i]+1]
            # embed()  
            colors[i,:]=COLOR_MAP[label]
            verts_tuple[i] = tuple(np.concatenate((mesh_points[i, :], colors[i, :]), axis=-1))
    else:
        verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
        for i in range(0, num_verts):
            verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[('vertex_indices', 'i4', (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, 'vertex')
    el_faces = plyfile.PlyElement.describe(faces_tuple, 'face')

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug('saving mesh to %s' % (ply_filename_out))
    ply_data.write(ply_filename_out)
    logging.debug(
        'converting to ply format and writing to file took {} s'.format(
            time.time() - start_time
        )
    )

def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()
    

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    try:
        verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
            numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
        )
    except:
        pass    

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

    logging.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )
def save_labels(decoder, dataset, max_batch=64 ** 3):
    decoder.eval()
    x ,y, z = np.shape(dataset['coords'])[0],np.shape(dataset['coords'])[1],np.shape(dataset['coords'])[2]
    num_samples = x * y * z
    samples = torch.zeros(num_samples, 4)
    labels = torch.zeros(num_samples)
    samples.requires_grad = False
    head = 0

    while head < num_samples:
        dim = 3
        sample_subset = torch.tensor(dataset['coords']).cuda()
        model_out = decoder(sample_subset)

        samples = model_out['sdf_out']

        labels = model_out['label_out'].squeeze().detach().data.max(1)[1]

        head += max_batch

    sdf_values = samples[:, dim].cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    return sdf_values, labels