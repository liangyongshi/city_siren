import numpy as np
pointcloud_path="/home/shiyl/kitti/05/newfile/000005.txt"
point_cloud = np.genfromtxt(pointcloud_path)
cars_coords = np.where(point_cloud[:,-1]==10)
cars = np.squeeze(point_cloud[cars_coords,:])
print(cars.shape)
np.savetxt("/home/shiyl/kitti/cars/car05.txt",cars)