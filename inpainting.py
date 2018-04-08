#Import libs
from __future__ import print_function
import matplotlib.pyplot as plt
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
from models.resnet import ResNet
from models.unet import UNet
from models.skip import skip
import torch
import torch.optim
from torch.autograd import Variable
from utils.inpainting_utils import *
#Setup libray parameters
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

imsize=-1
dim_div_by = 64
dtype = torch.cuda.FloatTensor
#Choose Figure
## Fig 7 (top)
img_path  = 'data/images/bonus/2.png'
mask_path = 'data/images/bonus/2_mask.png'
#Load Mask
img_pil, img_np = get_image(img_path, imsize)
img_mask_pil, img_mask_np = get_image(mask_path, imsize)
#Center Crop
img_mask_pil = crop_image(img_mask_pil, dim_div_by)
img_pil      = crop_image(img_pil,      dim_div_by)
img_np      = pil_to_np(img_pil)
img_mask_np = pil_to_np(img_mask_pil)
#Visualize
img_mask_var = np_to_var(img_mask_np).type(dtype)
saveImage("Results/Inpainting/OriginalImage1.png", img_np, 3, 11)
saveImage("Results/Inpainting/OriginalImage2.png", img_mask_np, 3, 11)
saveImage("Results/Inpainting/OriginalImage3.png", img_mask_np*img_np, 3, 11)
#plot_image_grid([img_np, img_mask_np, img_mask_np*img_np], 3,11)

#Setup
NET_TYPE = 'skip_depth6' # one of skip_depth4|skip_depth2|UNET|ResNet
pad = 'reflection' # 'zero'
OPT_OVER = 'net'
OPTIMIZER = 'adam'
INPUT = 'meshgrid'
input_depth = 2
LR = 0.01 
num_iter = 5000
param_noise = False
show_every = 500
figsize = 5
reg_noise_std = 0.03
net = skip(input_depth, img_np.shape[0], 
            num_channels_down = [128] * 5,
            num_channels_up   = [128] * 5,
            num_channels_skip = [0] * 5,  
            upsample_mode='nearest', filter_skip_size=1, filter_size_up=3, filter_size_down=3,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)
net = net.type(dtype)
net_input = get_noise(input_depth, INPUT, img_np.shape[1:]).type(dtype)
# Compute number of parameters
s  = sum(np.prod(list(p.size())) for p in net.parameters())
print ('Number of params: %d' % s)
# Loss
mse = torch.nn.MSELoss().type(dtype)
img_var = np_to_var(img_np).type(dtype)
mask_var = np_to_var(img_mask_np).type(dtype)
#Main Loop
i = 0
def closure():
    
    global i
    
    if param_noise:
        for n in [x for x in net.parameters() if len(x.size()) == 4]:
            n.data += n.data.clone().normal_()*n.data.std()/50
    
    if reg_noise_std > 0:
        net_input.data = net_input_saved + (noise.normal_() * reg_noise_std)
        
        
    out = net(net_input)
   
    total_loss = mse(out * mask_var, img_var * mask_var)
    total_loss.backward()
        
    print ('Iteration %05d    Loss %f' % (i, total_loss.data[0]), '\r', end='')
    if  i % show_every == 0:
        out_np = var_to_np(out)
        saveImage("Results/Inpainting/Inpainting_Itr" + str(i) + ".png", out_np, nrow = 1, factor = figsize)  
    i += 1
    return total_loss

net_input_saved = net_input.data.clone()
noise = net_input.data.clone()
p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)

out_np = var_to_np(net(net_input))
saveImage("Results/Inpainting/Inpainting_Final.png", out_np, factor = 5)