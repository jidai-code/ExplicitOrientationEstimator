import sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from NotreDame import NotreDame
from scipy.misc import imshow
from models.stacknet import StackNet

train_data = NotreDame()
train_loader = torch.utils.data.DataLoader(train_data, batch_size = 1, num_workers = 1, shuffle = True)
train_length = len(train_loader)

model = StackNet()
model = model.cuda()

ckpt = '/home/jdai/Documents/OrientationNet/experiments/stacknet/checkpoint.pth.tar'
oldweights = torch.load(ckpt)
if 'state_dict' in oldweights.keys():
	model.load_state_dict(oldweights['state_dict'])
else:
	model.load_state_dict(oldweights)

T_affine = np.matrix([[1,0,0],[0,1,0]]).astype(np.float32)
T_affine = torch.from_numpy(T_affine).unsqueeze(0)

model.eval()

for batch_id, patches in enumerate(train_loader):
	patches_cuda = [patch.cuda() for patch in patches]
	T_affine_cuda = T_affine.cuda()
	patches_var = torch.autograd.Variable(torch.cat(patches_cuda,dim = 1),volatile = True)
	T_affine_var = torch.autograd.Variable(T_affine_cuda,volatile = True)
	input_var = [patches_var,T_affine_var]
	# forward
	output_var = model(input_var)
	T_pred = output_var.data.cpu()
	T_pred = F.pad(T_pred,(0,1))
	p1_ts = patches[0]
	p2_ts = patches[1]

	p1 = p1_ts.view(64,64).numpy()
	p2 = p2_ts.view(64,64).numpy()

	grid = F.affine_grid(T_pred,p2_ts.size())
	p2_af_ts = F.grid_sample(p2_ts,grid).data
	p2_af = p2_af_ts.view(64,64).numpy()

	imshow(np.concatenate([p1,p2,p2_af],axis=1))
