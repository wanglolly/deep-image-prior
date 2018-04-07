#Import libs
from __future__ import print_function
import matplotlib.pyplot as plt
import argparse
import os
import csv
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
from models import *
import torch
import torch.optim
from skimage.measure import compare_psnr
from models.downsampler import Downsampler
from utils.sr_utils import *

#Setup libray parameters
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

#Setup file parameters
imsize = -1 
factor = 4 # 8
enforse_div32 = 'CROP' # we usually need the dimensions to be divisible by a power of two (32 in this case)
SAVE = True
# To produce images from the paper we took *_GT.png images from LapSRN viewer for corresponding factor,
# e.g. x4/zebra_GT.png for factor=4, and x8/zebra_GT.png for factor=8 
path_to_image = 'data/images/SR_GT.png'

#Load Image
img_pil = crop_image(get_image(path_to_image, imsize)[0], d=32)
img_np = pil_to_np(img_pil)     
if SAVE:
    saveImage("Results/SuperResolution/SR_Original.png", img_np, 4, 12)

#Load GT image
GTfilename = "data/images/SR_GT.png"
GTimg_pil = crop_image(get_image(GTfilename, imsize)[0], d=32)
GTimg_np = pil_to_np(GTimg_pil) 

#Set up parameters and net
input_depth = 32
 
INPUT =     'noise'
pad   =     'reflection'
OPT_OVER =  'net'
KERNEL_TYPE='lanczos2'

LR = 0.01
tv_weight = 0.0

OPTIMIZER = 'adam'

if factor == 4: 
    num_iter = 2000
    reg_noise_std = 1./30.
elif factor == 8:
    num_iter = 4000
    reg_noise_std = 1./20.
else:
    assert False, 'We did not experiment with other factors'

net_input = get_noise(input_depth, INPUT, (int(GTimg_np.size[1]), int(GTimg_np.size[0])).type(dtype).detach()

NET_TYPE = 'skip' # UNet, ResNet
net = get_net(input_depth, 'skip', pad,
              skip_n33d=128, 
              skip_n33u=128, 
              skip_n11=4, 
              num_scales=5,
              upsample_mode='bilinear').type(dtype)

# Losses
mse = torch.nn.MSELoss().type(dtype)

img_LR_var = np_to_var(img_np).type(dtype)

downsampler = Downsampler(n_planes=3, factor=factor, kernel_type=KERNEL_TYPE, phase=0.5, preserve_size=True).type(dtype)

#Prepare PSNR File
PSNRFilename = 'Results/SuperResolution/PSNR.csv'
PSNRFile = open(PSNRFilename, 'w')
PSNRCursor = csv.writer(PSNRFile)

#Initial PSNR
print ('Input PSNR %.3f' % (compare_psnr(downsampler(GTimg_np), img_np)), '\n', end='') 
PSNRCursor.writerow(['Input', compare_psnr(downsampler(GTimg_np), img_np)])

#Define closure and optimize
def closure():
    global i
    
    if reg_noise_std > 0:
        net_input.data = net_input_saved + (noise.normal_() * reg_noise_std)
        
    out_HR = net(net_input)
    out_LR = downsampler(out_HR)

    total_loss = mse(out_LR, img_LR_var) 
    
    if tv_weight > 0:
        total_loss += tv_weight * tv_loss(out_HR)
        
    total_loss.backward()

    # Log
    psnr_LR = compare_psnr(img_np, var_to_np(out_LR))
    psnr_HR = compare_psnr(GTimg_np, var_to_np(out_HR))
    print ('Iteration %05d    PSNR_LR %.3f   PSNR_HR %.3f' % (i, psnr_LR, psnr_HR), '\r', end='')
    PSNRCursor.writerow([i, psnr_HR, psnr_LR])                  
    # History
    psnr_history.append([psnr_LR, psnr_HR])
    
    if SAVE and i % 500 == 0:
        out_HR_np = var_to_np(out_HR)
        saveImage("Results/SuperResolution/SR_Itr" + str(i) + ".png", np.clip(out_HR_np, 0, 1), nrow = 3, factor = 13)
 
    i += 1
    
    return total_loss

psnr_history = [] 
net_input_saved = net_input.data.clone()
noise = net_input.data.clone()

i = 0
p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)

out_HR_np = np.clip(var_to_np(net(net_input)), 0, 1)
#result_deep_prior = put_in_center(out_HR_np, imgs['orig_np'].shape[1:])

# For the paper we acually took `_bicubic.png` files from LapSRN viewer and used `result_deep_prior` as our result
saveImage("Results/SuperResolution/SR_Final.png", out_HR_np, nrow = 1, factor = 4)
PSNRCursor.writerow(['Final', compare_psnr(GTimg_np, out_HR_np)]) 
PSNRFile.close()