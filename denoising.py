#Import libs
from __future__ import print_function
#matplotlib
import matplotlib.pyplot as plt
#matplotlib inline
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import csv
import numpy as np
from models import *
import torch
import torch.optim
from torch.autograd import Variable
from utils.denoising_utils import *
from skimage.measure import compare_psnr

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

imsize = -1
sigma = 25
sigma_ = sigma/255.
SAVE = True

#Load Image
fname = 'data/images/noise_image.png'
img_pil = crop_image(get_image(fname, imsize)[0], d=32)
img_np = pil_to_np(img_pil)     
if SAVE:
    saveImage("Results/Denoising/Denoising_Original.png", img_np, 4, 6)

#Load GT image
GTfilename = "data/images/noise_GT.png"
GTimg_pil = crop_image(get_image(GTfilename, imsize)[0], d=32)
GTimg_np = pil_to_np(GTimg_pil) 

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
figsize = 1 
    
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

#Prepare PSNR File
PSNRFilename = 'Results/Denoising/PSNR.csv'
PSNRFile = open(PSNRFilename, 'w')
PSNRCursor = csv.writer(PSNRFile)

#Initial PSNR
print ('Input PSNR %.3f' % (compare_psnr(GTimg_np, img_np)), '\n', end='') 
PSNRCursor.writerow(['Input', compare_psnr(GTimg_np, img_np)])

i = 0
def closure():
    
    global i
    
    if reg_noise_std > 0:
        net_input.data = net_input_saved + (noise.normal_() * reg_noise_std)
    
    out = net(net_input)
   
    total_loss = mse(out, img_noisy_var)
    total_loss.backward()

    out_np = var_to_np(out)    
    print ('Iteration %05d    Loss %f   PSNR %.3f' % (i, total_loss.data[0], compare_psnr(GTimg_np, out_np)), '\r', end='')
    PSNRCursor.writerow([str(i), compare_psnr(GTimg_np, out_np), total_loss.data[0]])
    if  SAVE and i % show_every == 0:
        saveImage("Results/Denoising/Denoising_Itr" + str(i) + ".png", out_np, nrow = 1)  
    i += 1

    return total_loss

p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)

out_np = var_to_np(net(net_input))
if SAVE:
    saveImage("Results/Denoising/Denoising_FinalOutput.png", out_np)
    print ('\n' +'Final PSNR %.3f' % (compare_psnr(GTimg_np, out_np)), '\n', end='') 
    PSNRCursor.writerow(['Final', compare_psnr(GTimg_np, out_np)])
PSNRFile.close()
