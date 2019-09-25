import sys
import math
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from scipy.misc import imshow
import models
import datasets
from utils import *
from metrics import *
import torch.nn.functional as F

system_check()

iterN = 0
best_score = sys.float_info.max
b_size = 1
bins = 36
lvl = 2

print('========================Dataset=========================')
# choose training dataset
[train_data,valid_data] = datasets.__dict__['NotreDame'](bins = bins,t_f = False, v_f=False)
train_loader = torch.utils.data.DataLoader(train_data, batch_size = b_size, num_workers = 6, shuffle = True)
train_length = len(train_loader)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size = b_size, num_workers = 6, shuffle = False)
valid_length = len(valid_loader)
print("Training:\t%i" % (train_length))
print("Valid:\t\t%i" % (valid_length))

# choose models
model = models.__dict__['STN_test'](lvl = lvl,bins = bins)
model = model.cuda()
oldweights = torch.load('/home/jdai/Documents/TransNet/RotNet/experiments/STN_v1/checkpoint.pth.tar')
if 'state_dict' in oldweights.keys():
	model.load_state_dict(oldweights['state_dict'])
else:
	model.load_state_dict(oldweights)


# enable cudnn for static size data
cudnn.benchmark = True

model.eval()

for batch_id, [patch_ts,T_affine, bin_id, scale] in enumerate(valid_loader):

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
	[patch0,patch1,pred_angle] = model(inputs)

	pred_bin_id = F.log_softmax(pred_angle, 1)
	m_value,m_indice = torch.max(pred_bin_id,1)
	m_indice = m_indice.data.cpu().numpy()
	

	pred_id = m_indice[0]
	pred_id_l = (pred_id-1) % bins
	pred_id_r = (pred_id+1) % bins

	y0 = pred_angle[0,pred_id_l].data.cpu()
	y1 = pred_angle[0,pred_id].data.cpu()
	y2 = pred_angle[0,pred_id_r].data.cpu()
	A = float(y0/2-y1+y2/2)
	B = float(y2/2-y0/2)
	C = float(y1)
	x_hat = - B / (2*A)
	pred_Ang = (x_hat+pred_id)*(2*math.pi/bins)

	p0 = patch0.data.cpu().squeeze(0).numpy()
	p0 = np.transpose(p0,[1,2,0])
	p1 = patch1.data.cpu().squeeze(0).numpy()
	p1 = np.transpose(p1,[1,2,0])

	T_rot = np.matrix([[math.cos(pred_Ang),math.sin(pred_Ang),0],[-math.sin(pred_Ang),math.cos(pred_Ang),0]])
	T_affine = T_rot.astype(np.float32)
	T_affine = torch.from_numpy(T_affine)
	T_affine = torch.autograd.Variable(T_affine.unsqueeze(0).cuda(),volatile=True)

	grid = F.affine_grid(T_affine,torch.Size([1,3,64,64])) / 2.0
	patch_af = F.grid_sample(patches_var, grid)
	p0af = patch_af.data.cpu().squeeze(0).numpy()
	p0af = np.transpose(p0af,[1,2,0])

	imshow(np.concatenate([p0,p1,p0af],axis=1))




		
