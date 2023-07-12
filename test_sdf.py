'''Test script for experiments in paper Sec. 4.2, Supplement Sec. 3, reconstruction from laplacian.
'''

# Enable import from parent package
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import torch
import modules, utils
import sdf_meshing
import configargparse
import sdf_plot_right
import dataio

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='logs', help='root for logging')
p.add_argument('--experiment_name', type=str, default='gln30')

# General training options
p.add_argument('--batch_size', type=int, default=20000)
p.add_argument('--checkpoint_path', default='/home/user/pyProject/experiment_results/gln30')

p.add_argument('--model_type', type=str, default='sine',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')
p.add_argument('--mode', type=str, default='mlp',
               help='Options are "mlp" or "mlp"')
p.add_argument('--resolution', type=int, default=800)
p.add_argument('--frames', type=int, default=1100)
p.add_argument('--interval', type=int, default=60)
p.add_argument('--pemode', type=str, default='fourier')# nerf fourier mlp xyz
p.add_argument('--num_encoding_functions', type=int, default=10)

opt = p.parse_args()


for index in range(910,opt.frames,opt.interval):

    class SDFDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = modules.semanticSIREN(type=opt.model_type,in_features=3,pemode=opt.pemode,num_encoding_functions=opt.num_encoding_functions)
            self.model.load_state_dict(torch.load(os.path.join(opt.checkpoint_path,str(index)+'.pth')))
            self.model.cuda()


        def forward(self, coords):
            model_in = {'coords': coords}
            return self.model(model_in)

    time1=time.time()
    
    sdf_decoder = SDFDecoder()
    
    time2=time.time()
    
    print("initial time cost",time2-time1)
    root_path = os.path.join(opt.logging_root, opt.experiment_name)
    
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    
    sdf_meshing.create_mesh(sdf_decoder, os.path.join(root_path, str(index)), N=opt.resolution)
    time3=time.time()
    print("initial time cost",time3-time2)
    
    
    
    
    
    
    
