#Import libs
from __future__ import print_function
#matplotlib
import matplotlib.pyplot as plt
#matplotlib inline
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
from models import *
import torch
import torch.optim
from torch.autograd import Variable
from utils.denoising_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

imsize = -1
sigma = 25
sigma_ = sigma/255.
SAVE = True

#Load Image
fname = 'data/images/noise_image.png'
# Add synthetic noise
img_pil = crop_image(get_image(fname, imsize)[0], d=32)
img_np = pil_to_np(img_pil)     
if SAVE:
    #plot_image_grid([img_np, img_noisy_np], 4, 6)
    saveImage("Results/Denoising/Denoising_Original.png", img_np, 4, 6)

#Setup
INPUT = 'noise' # 'meshgrid'
pad = 'reflection'
OPT_OVER = 'net' # 'net,input'

reg_noise_std = 1./30. # set to 1./20. for sigma=50
LR = 0.01

OPTIMIZER='adam' # 'LBFGS'
show_every = 300

num_iter = 1800
input_depth = 32 
figsize = 4 
    
net = get_net(input_depth, 'skip', pad,
                skip_n33d=128, 
                skip_n33u=128, 
                skip_n11=4, 
                num_scales=5,
                upsample_mode='bilinear').type(dtype)
    
net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()

# Compute number of parameters
s  = sum([np.prod(list(p.size())) for p in net.parameters()]); 
print ('Number of params: %d' % s)

# Loss
mse = torch.nn.MSELoss().type(dtype)

img_noisy_var = np_to_var(img_np).type(dtype)

#Optimize
net_input_saved = net_input.data.clone()
noise = net_input.data.clone()

i = 0
def closure():
    
    global i
    
    if reg_noise_std > 0:
        net_input.data = net_input_saved + (noise.normal_() * reg_noise_std)
    
    out = net(net_input)
   
    total_loss = mse(out, img_noisy_var)
    total_loss.backward()
        
    print ('Iteration %05d    Loss %f' % (i, total_loss.data[0]), '\r', end='')
    if  SAVE and i % show_every == 0:
        out_np = var_to_np(out)
        #plot_image_grid([np.clip(out_np, 0, 1)], factor=figsize, nrow=1)
        saveImage("Results/Denoising/Denoising_Itr" + str(i) + ".png", out_np, nrow = 1, factor = figsize)
        
    i += 1

    return total_loss

p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)

out_np = var_to_np(net(net_input))
if SAVE:
    #q = plot_image_grid([np.clip(out_np, 0, 1), img_np], factor=13)
    saveImage("Results/Denoising/Denoising_FinalOutput.png", out_np, factor=13)