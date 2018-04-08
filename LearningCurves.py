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

#Setup
INPUT = 'noise' # 'meshgrid'
pad = 'reflection'
OPT_OVER = 'net' # 'net,input'
OPTIMIZER='adam' # 'LBFGS'

reg_noise_std = 1./30. # set to 1./20. for sigma=50
LR = 0.01
imsize = -1
SAVE = False
show_every = 500
num_iter = 2400
input_depth = 32 
figsize = 5 

#Load Image (Target1)
imageFname = 'data/images/natureImage.jpeg'
img_pil = crop_image(get_image(imageFname, imsize)[0], d=32)
img_np = pil_to_np(img_pil)
saveImage("Results/LearningCurves/Image.png", img_np, nrow = 4, factor = 1)
#Target2(Image noise)
sigma = 25
sigma_ = sigma/255.
img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)
saveImage("Results/LearningCurves/ImageNoise.png", img_noisy_np, nrow = 4, factor = 1)
#Target3(Image pixel shuffle)
img_shuffle_np = np.copy(img_np)
np.random.shuffle(img_shuffle_np.flat)
saveImage("Results/LearningCurves/ImageShuffle.png", img_shuffle_np, nrow = 4, factor = 1)
#Target4(White noise)
random_np = np.random.random_sample((3, img_pil.size[1], img_pil.size[0]))
saveImage("Results/LearningCurves/Noise.png", random_np, nrow = 4, factor = 1)

TargetImage = {'image' : img_np,
                'imageNoise' : img_noisy_np,
                'imageShuffle' : img_shuffle_np,
                'Noise' : random_np}
Target = 'Noise'

#Prepare Loss File
LossFilename = 'Results/LearningCurves/LearningCurve_' + Target + '.csv'
LossFile = open(LossFilename, 'w')
LossCursor = csv.writer(LossFile)

#Setup net    
net = skip(input_depth, 3, 
            num_channels_down = [8, 16, 32, 64, 128], 
            num_channels_up   = [8, 16, 32, 64, 128],
            num_channels_skip = [0, 0, 0, 4, 4], 
            upsample_mode='bilinear',
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)
   
net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()

# Compute number of parameters
s  = sum([np.prod(list(p.size())) for p in net.parameters()]); 
print ('Number of params: %d' % s)

# Loss
mse = torch.nn.MSELoss().type(dtype)
img_noisy_var = np_to_var(TargetImage[Target]).type(dtype)

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

    out_np = var_to_np(out)    
    print ('Iteration %05d    Loss %f' % (i, total_loss.data[0]), '\r', end='')
    LossCursor.writerow([str(i), total_loss.data[0]])
    i += 1

    return total_loss

p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)
LossFile.close()
