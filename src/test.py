import sys
import numpy as np
import torch
import models
import datasets
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from datasets.NotreDame import NotreDame
from scipy.misc import imshow
from models.stacknet import StackNet

train_data = NotreDame()
train_loader = torch.utils.data.DataLoader(train_data, batch_size = 1, num_workers = 1, shuffle = True)
train_length = len(train_loader)

model1 = models.__dict__['StackNet']()
model1 = model1.cuda()
model2 = models.__dict__['SiameseNet']()
model2 = model2.cuda()
model3 = models.__dict__['Siamese2StreamNet']()
model3 = model3.cuda()

ckpt1 = '/home/jdai/Documents/OrientationNet/experiments/stacknet/checkpoint.pth.tar'
ckpt2 = '/home/jdai/Documents/OrientationNet/experiments/siamesenet/checkpoint.pth.tar'
ckpt3 = '/home/jdai/Documents/OrientationNet/experiments/Siamese2StreamNet/checkpoint.pth.tar'

oldweights = torch.load(ckpt1)
if 'state_dict' in oldweights.keys():
	model1.load_state_dict(oldweights['state_dict'])
else:
	model1.load_state_dict(oldweights)

oldweights = torch.load(ckpt2)
if 'state_dict' in oldweights.keys():
	model2.load_state_dict(oldweights['state_dict'])
else:
	model2.load_state_dict(oldweights)

oldweights = torch.load(ckpt3)
if 'state_dict' in oldweights.keys():
	model3.load_state_dict(oldweights['state_dict'])
else:
	model3.load_state_dict(oldweights)

T_affine = np.matrix([[1,0,0],[0,1,0]]).astype(np.float32)
T_affine = torch.from_numpy(T_affine).unsqueeze(0)

model1.eval()
model2.eval()
model3.eval()

for batch_id, patches in enumerate(train_loader):
	patches_cuda = [patch[:,:,32:96,32:96].cuda() for patch in patches]
	T_affine_cuda = [T_affine.cuda(),T_affine.cuda()]
	patches_var = torch.autograd.Variable(torch.cat(patches_cuda,dim = 1),volatile = True)
	T_affine_var = torch.autograd.Variable(torch.cat(T_affine_cuda,dim = 1),volatile = True)
	input_var = [patches_var,T_affine_var]
	# forward
	output_var1 = model1(input_var)
	output_var2 = model2(input_var)
	output_var3 = model3(input_var)

	T_pred1 = output_var1.data.cpu()
	T_pred1 = F.pad(T_pred1,(0,1))
	T_pred2 = output_var2.data.cpu()
	T_pred2 = F.pad(T_pred2,(0,1))
	T_pred3 = output_var3.data.cpu()
	T_pred3 = F.pad(T_pred3,(0,1))

	p1_ts = patches[0][:,:,32:96,32:96].contiguous()
	p2_ts = patches[1][:,:,32:96,32:96].contiguous()
	p2_ori_ts = patches[1].contiguous()

	p1 = p1_ts.view(64,64).numpy()
	p2 = p2_ts.view(64,64).numpy()

	grid = F.affine_grid(T_pred1,p2_ts.size())/2
	p2_m1_ts = F.grid_sample(p2_ori_ts,grid).data

	grid = F.affine_grid(T_pred2,p2_ts.size())/2
	p2_m2_ts = F.grid_sample(p2_ori_ts,grid).data

	grid = F.affine_grid(T_pred3,p2_ts.size())/2
	p2_m3_ts = F.grid_sample(p2_ori_ts,grid).data

	p2_m1 = p2_m1_ts.view(64,64).numpy()
	p2_m2 = p2_m2_ts.view(64,64).numpy()
	p2_m3 = p2_m3_ts.view(64,64).numpy()

	imshow(np.concatenate([p1,p2,p2_m1,p2_m2,p2_m3],axis=1))
