"""
@Author: Pengfei Li
@File: modify_points.py
@Description: 
@Date: 2021/08/06
"""
import torch
import math
import open3d as o3d
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch.nn.functional as F
import os

import modules, diff_operators

from IPython import embed


def threshold(v,t):
    if len(v.shape) == 1:
        norm_v = np.linalg.norm(v)
        if norm_v < t:
            return v
        else:
            return v * t / norm_v
    else:
        norm_v = np.linalg.norm(v,axis=1)
        out = np.array([v[i] if norm_v[i] < t else v[i] * t / norm_v[i] for i in range(v.shape[0])])
        return out


def upsample(point_cloud):
    '''adding attraction and repulsion terms, then upsampling'''

    epsilon = 0.0001

    k_neigh = NearestNeighbors(n_neighbors=9)
    k_neigh.fit(point_cloud)

    modify_list = []
    for p in point_cloud:
        # dis,order are the distances and their orders of the 9 nearest points,respectively
        dis,order = k_neigh.kneighbors([p])
        # the first term of dis or order refers to q itself,needed to be eliminated   
        dis = np.delete(dis,0)
        order = np.delete(order,0)
        ave_dis = dis.sum()/8
        D = 3.5 * ave_dis
        t = 0.1 * ave_dis

        attr_xyz = [0,0,0]
        repul_xyz = [0,0,0]

        attr  = 0	# attraction and replusion weights needs to be divided 
        repul = 0

        # modify the q's coordinate through the nearest 8 points
        for index,r in enumerate(order):
            delta = point_cloud[r,:3] - p[:3]
            normal = point_cloud[r,3:]
            w  = math.exp(0 - np.power(dis[index],2) / (D + epsilon)) 
            v  = math.exp(0 - np.power((delta*normal).sum(),2) / (D + epsilon))

            attr_xyz -= v * delta
            repul_xyz += w * delta
	    
            attr  += v
            repul += w

        modify_list.append(threshold(attr_xyz/attr, t) + threshold(0.5*repul_xyz/repul, t))

    point_cloud[:, :3] -= modify_list

    k_neigh.fit(point_cloud)
    new_points = np.array([[]]*6).T
    for p in point_cloud:
        dis, order = k_neigh.kneighbors([p])
        dis = np.delete(dis,0)
        order = np.delete(order,0)
        ave_dis = np.mean(dis)

        for index, r in enumerate(order):
            if dis[index] > ave_dis and dis[index] > 9:
                new_p = (2*p + point_cloud[r])/3
                new_points = np.append(new_points,[new_p],axis=0)

    point_cloud = np.concatenate((point_cloud, new_points),axis=0)

    return point_cloud


def uniform(point_cloud):
    epsilon = 0.0001

    k_neigh = NearestNeighbors(n_neighbors=9)
    k_neigh.fit(point_cloud)
    
    modify_list = []
    for q in point_cloud:
        dis, order = k_neigh.kneighbors([q])
       
        dis = np.delete(dis, 0)
        order = np.delete(order, 0)

        ave_dis = dis.sum()/8
        delta = [0,0,0]
        for index, r in enumerate(order):
            single_delta = point_cloud[r, :3] - q[:3]
            w = math.exp(0 - np.power(dis[index],2)/(ave_dis + epsilon))

            delta += w * single_delta / (dis[index] + epsilon)
        
        modify_list.append(delta)

    point_cloud[:, :3] -= modify_list

    return point_cloud


def project(point_cloud, model):
    model.eval()

    point_count = point_cloud.shape[0]
    t_threshold = (2*math.sqrt(3)) / (2*point_count)

    max_batch = 1000

    iteration_count = 10
    for i in range(iteration_count):
        samples = torch.zeros(point_count, 4)
        samples[:,:3] = torch.tensor(point_cloud[:,:3])
        samples.requires_grad = True

        gradient = []
        head = 0
        while head < point_count:
            sample_subset = samples[head : min(head + max_batch, point_count), 0:3].cuda()
            temp_out = model({'coords': sample_subset})
            samples[head : min(head + max_batch, point_count), 3] = (
                temp_out['model_out']
                .squeeze()
            )
            gradient.append(diff_operators.gradient(temp_out['model_out'], temp_out['model_in']).cpu())
            head += max_batch

        gradient = torch.cat(gradient, 0).cpu()

        # point_cloud[:,:3] = point_cloud[:,:3] - \
        #     threshold((F.normalize(gradient, p=2, dim=1)*samples[:, 3].unsqueeze(1)).detach().numpy(), t_threshold)
        point_cloud[:,:3] = point_cloud[:,:3] - \
            (F.normalize(gradient, p=2, dim=1)*samples[:, 3].unsqueeze(1)).detach().numpy()

    return point_cloud


def updata_points(model, point_cloud, cycle, model_dir):
    # '''coordinates and normals'''
    
    # reshape the coordinates to original scale
    point_cloud[:,:3] = (point_cloud[:,:3] / 2 + 0.5) * 255

    print('upsample begin')
    point_cloud = upsample(point_cloud)
    print('upsample end')

    np.savetxt(os.path.join(model_dir, 'cycle_'+str(cycle)+'_upsample.xyz'), point_cloud)

    print('uniform begin')
    point_cloud = uniform(point_cloud)
    print('uniform end')

    np.savetxt(os.path.join(model_dir, 'cycle_'+str(cycle)+'_uniform.xyz'), point_cloud)

    print('project begin')
    point_cloud = project(point_cloud, model)
    print('project end')
    
    np.savetxt(os.path.join(model_dir, 'cycle_'+str(cycle)+'_project.xyz'), point_cloud)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
    pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0.,0.,1.]))
    point_cloud = np.concatenate((np.asarray(pcd.points),np.asarray(pcd.normals)),axis=1)

    point_cloud[:,:3] = (point_cloud[:,:3] / 255 - 0.5) * 2

    return point_cloud


def updata_normals(point_cloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
    pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0.,0.,1.]))
    # pcd.normalize_normals()
    pcd.remove_non_finite_points(remove_nan=True,remove_infinite=True)
    point_cloud = np.concatenate((np.asarray(pcd.points),np.asarray(pcd.normals)),axis=1)
    # if point_cloud.shape[0]<370000:
    #     pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    #     pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    #     pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0.,0.,1.]))
    #     # pcd.normalize_normals()
    #     pcd.remove_non_finite_points(remove_nan=True,remove_infinite=True)
    #     point_cloud = np.concatenate((np.asarray(pcd.points),np.asarray(pcd.normals)),axis=1)
    # else:
    #     pcd.points = o3d.utility.Vector3dVector(point_cloud[-350000:, :3])
    #     pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    #     pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0.,0.,1.]))
    #     # pcd.normalize_normals()
    #     pcd.remove_non_finite_points(remove_nan=True,remove_infinite=True)
    #     point_cloud_temp = np.concatenate((np.asarray(pcd.points),np.asarray(pcd.normals)),axis=1)
    #     point_cloud=np.concatenate((np.asarray(point_cloud_temp),np.asarray(point_cloud[:350000,:])),axis=0)
    print("update normals finished")
    return point_cloud


