import random
import sys
import time
import math
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from scipy.misc import imread, imshow, imsave
import models
import torch.backends.cudnn as cudnn
import datasets
from utils import *

lvl = 2
bins = 36
b_size = 64

[train_data,valid_data] = datasets.__dict__['NotreDame'](bins = bins)
train_loader = torch.utils.data.DataLoader(train_data, batch_size = b_size, num_workers = 6, shuffle = True, drop_last = True)
train_length = len(train_loader)

# choose models
model = models.__dict__['STN_v1'](lvl = lvl,bins = bins)
model = model.cuda()
oldweights = torch.load('/home/jdai/Documents/TransNet/TransNet/experiments/STN_v1/checkpoint.pth.tar')
if 'state_dict' in oldweights.keys():
	model.load_state_dict(oldweights['state_dict'])
else:
	model.load_state_dict(oldweights)
model.eval()
# enable cudnn for static size data
cudnn.benchmark = True

Ang_mo = data_monitor()

for batch_id, [patch_ts,T_affine, bin_id, scale, theta] in enumerate(train_loader):
	error_Ang_agg = 0

	# copy to GPU
	patch_cuda = patch_ts.cuda()
	T_affine_cuda = T_affine.cuda()
	bin_id_cuda = bin_id.cuda()
	scale_cuda = scale.cuda().float()
	
	# make differentiable
	patches_var = torch.autograd.Variable(patch_cuda,volatile = True)
	T_affine_var = torch.autograd.Variable(T_affine_cuda,volatile = True)
	bin_id_var = torch.autograd.Variable(bin_id_cuda,volatile = True)
	scale_var = torch.autograd.Variable(scale_cuda,volatile = True)

	inputs = [patches_var,T_affine_var]

	# forward
	[pred_angle_var,pred_scale_var] = model(inputs)

	pred_bin_id = F.log_softmax(pred_angle_var, 1)
	m_value,m_indice = torch.max(pred_bin_id,1)
	m_indice = m_indice.data.cpu().numpy()

	for i in range(len(m_indice)):
		pred_id = m_indice[i]
		pred_id_l = (pred_id-1) % bins
		pred_id_r = (pred_id+1) % bins
		y0 = pred_angle_var[i,pred_id_l].data.cpu()
		y1 = pred_angle_var[i,pred_id].data.cpu()
		y2 = pred_angle_var[i,pred_id_r].data.cpu()
		A = float(y0 / (-2))
		B = float(y1 / (-1))
		C = float(y2 / (2))
		x_hat = (A*1+C*(-1))/(2*(A+B+C))
		pred_Ang = (x_hat+pred_id)*(360/bins)
		gt_Ang = (theta[i]*(180/math.pi))
		error_Ang = abs(gt_Ang - pred_Ang) % 360
		error_Ang = (error_Ang if error_Ang < 180 else (360 - error_Ang))
		error_Ang_agg += error_Ang

	error_Ang_agg /= len(m_indice)


	Ang_mo.update(error_Ang_agg,len(m_indice))
	print('batch:%i/%i, errors:%.3f' % (batch_id,train_length,Ang_mo.avg))
