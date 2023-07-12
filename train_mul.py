import sys
import os

from torch.utils.data import DataLoader
import configargparse
import time
import numpy as np
import shutil

from IPython import embed

import training, modules, loss_functions

import torch
import utils
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import dataset, modify_points, utils_eval
import dataio
# import wandb
p = configargparse.ArgumentParser()

# General training options
p.add_argument('--dataset_path', type=str, default='/home/user/pyProject/dataset/odometry_velodyne/05/newfile/')
p.add_argument('--log_path', type=str, default='logs/')
p.add_argument('--expr_name', type=str, default='test')
p.add_argument('--integtxt', type=str, default='/home/user/pyProject/siren_mlp_en/data/integtxt.txt')
p.add_argument('--logging_root', type=str, default='/home/user/pyProject/siren_mlp_en/logs/', help='root for logging')
p.add_argument('--experiment_name', type=str, default='experiment05_fourier')
# p.add_argument('--experiment_name', type=str, default='kitti_label_experiment_fourier')
p.add_argument('--backup', type=str, default='kitti05_backup_fourier')
# p.add_argument('--backup', type=str, default='kitti_label_backup_fourier')

p.add_argument('--num_epochs', type=int, default=100,
               help='Number of epochs to train for.')
p.add_argument('--epochs_til_ckpt', type=int, default=100,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=100,
               help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--clip_grad', type=bool, default=True)

p.add_argument('--model_type', type=str, default='sine',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')

p.add_argument('--eval_resolution', type=int, default=256)

p.add_argument('--data_class', type=str, default='all',
               help='Options are "all"(train & valid & test) and "valid"')

####################################################################
p.add_argument('--batch_size', type=int, default=160000)  # the minimum count of points is 4995 (10, 000195)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate.')
p.add_argument('--fix_coordiante', type=bool, default=True)
p.add_argument('--pemode', type=str, default='fourier') ## nerf mlp fourier
p.add_argument('--num_encoding_functions', type=int, default=10) 
####################################################################
p.add_argument('--incremental_sampling', type=bool, default=True)
p.add_argument('--pre_sample_rate', type=float, default=0.75)
####################################################################
p.add_argument('--frames', type=int, default=920)
p.add_argument('--interval', type=int, default=10)
p.add_argument('--frame_accumulate_num', type=int, default=10)
#################################################################
p.add_argument('--use_pre_model', type=bool, default=False)
p.add_argument('--pthpath', type=str, default='/home/user/pyProject/siren_mlp_en/logs/kitti05_backup_fourier/440.pth')

opt = p.parse_args()
# wandb.init(project="siren")

model = modules.semanticSIREN(type=opt.model_type,in_features=3,pemode=opt.pemode,num_encoding_functions=opt.num_encoding_functions)  
if opt.use_pre_model:
    model.load_state_dict(torch.load(opt.pthpath))
model.cuda()

loss_fn = loss_functions.sdf_label


summary_fn = utils.write_sdf_summary

root_path = os.path.join(opt.logging_root, opt.experiment_name)

if not os.path.exists(root_path):
    os.makedirs(root_path)
    
if not opt.use_pre_model:                
    if os.path.exists(opt.integtxt):
        os.remove(opt.integtxt) 


if not opt.use_pre_model:     
    first_frame=True  
else:
    first_frame=False

pre_size=0 

for index in range(450,opt.frames,opt.interval):
    # time1 = time.time() 
    input_data_path=open(opt.integtxt,'a+') 
    if ((opt.incremental_sampling==True) and (first_frame == False)):
        pc = np.genfromtxt(opt.integtxt)
        pre_size=pc.shape[0]
        print("pre_size",pre_size)
    num=index
    for i in range(0,opt.frame_accumulate_num,1):
        filepath=os.path.join(opt.dataset_path,str((num+i)).zfill(6)+".txt") ## for kitti dataset,1 frame of every 5 frames   
        # filepath=os.path.join(opt.dataset_path,str((num+i)*10)+".xyz")  ##for ICL dataset    
        f=open(filepath)
        input_data_path.write(f.read()+'\n') 
        print(filepath)             
    input_data_path.close()     
           
    train_dataset = dataio.PointCloud(opt.integtxt,on_surface_points=opt.batch_size,fix_coordiante=opt.fix_coordiante,incremental_sampling=opt.incremental_sampling,pre_size=pre_size,first_frame=first_frame,pre_sample_rate=opt.pre_sample_rate)   
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=16)
    

    changingepoch=10
    
    if not os.path.exists(os.path.join(opt.logging_root,opt.backup)):
        os.makedirs(os.path.join(opt.logging_root,opt.backup))
    backuppth= os.path.join(opt.logging_root,opt.backup) 
    
    training.train(index,model=model, train_dataloader=train_dataloader, epochs=changingepoch, lr=opt.lr,
               steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
               model_dir=root_path, backuppth=backuppth, loss_fn=loss_fn, summary_fn=summary_fn, double_precision=False,
               clip_grad=True)
    
    first_frame=False

    # time2 = time.time()
    # print('time elapse:', (time2-time1)//60, 'min', (time2-time1)%60, 's')
    torch.cuda.empty_cache()

os.remove(opt.integtxt)    ##文件一直追加数据，所以最后删除
