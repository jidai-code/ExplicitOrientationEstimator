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
model = models.__dict__['STN_empty'](lvl = lvl,bins = bins)
model = model.cuda()
oldweights = torch.load('/home/jdai/Documents/TransNet/RotNet/experiments/STN_v1/checkpoint.pth.tar')
if 'state_dict' in oldweights.keys():
	model.load_state_dict(oldweights['state_dict'])
else:
	model.load_state_dict(oldweights)


# enable cudnn for static size data
cudnn.benchmark = True

model.eval()
empty_np = np.random.rand(1,3,64,64).astype(np.float32)
empty_patch_var = torch.autograd.Variable(torch.from_numpy(empty_np).cuda(),volatile=True)

for batch_id, [patch_ts,T_affine,bin_id,theta] in enumerate(valid_loader):

	# copy to GPU
	patch_cuda = patch_ts.cuda()
	T_affine_cuda = T_affine.cuda()
	
	# make differentiable
	patches_var = torch.autograd.Variable(patch_cuda,volatile = True)
	T_affine_var = torch.autograd.Variable(T_affine_cuda,volatile = True)

	grid = F.affine_grid(T_affine_var,torch.Size([1,3,64,64])) / 2.0
	patch1_var = F.grid_sample(patches_var, grid)
	patch0_var = patches_var[:,:,32:96,32:96]

	torch.cuda.FloatTensor(torch.Size([1,3,64,64]))

	pred_bin0 = model([empty_patch_var,patch0_var])
	pred_bin1 = model([empty_patch_var,patch1_var])

	pred_angle0 = get_angle(pred_bin0,bins)
	pred_angle1 = get_angle(pred_bin1,bins)

	pred_theta = pred_angle1-pred_angle0
	if pred_theta < 0:
		pred_theta += 2*math.pi
	print('angle_pred:%.3f,angle_gt:%.3f: error:%.3f' % (pred_theta,theta,torch.min(2*math.pi-torch.abs(pred_theta-theta),torch.abs(pred_theta-theta))))





		
