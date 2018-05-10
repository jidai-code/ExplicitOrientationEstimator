import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from .net_utils import *

class STN_test(base_net):
	def __init__(self):
		base_net.__init__(self)

	def setup(self):
		print('STN:\t\ttestNet')

		self.conv1 = nn.Sequential(
			conv(in_planes = 3, out_planes = 32, kernel_size = 3, padding = 1),
			conv(in_planes = 32, out_planes = 32, kernel_size = 3, padding = 1),
			nn.MaxPool2d(kernel_size = 2, stride = 2),
			)
		self.conv2 = nn.Sequential(
			conv(in_planes = 32, out_planes = 32, kernel_size = 3, padding = 1),
			conv(in_planes = 32, out_planes = 32, kernel_size = 3, padding = 1),
			nn.MaxPool2d(kernel_size = 2, stride = 2),
			)
		self.conv3 = nn.Sequential(
			conv(in_planes = 32, out_planes = 32, kernel_size = 3, padding = 1),
			conv(in_planes = 32, out_planes = 32, kernel_size = 3, padding = 1),
			nn.MaxPool2d(kernel_size = 2, stride = 2),
			)
		self.fc = nn.Sequential(
			nn.Linear(in_features = 4096, out_features = 256),
			nn.ReLU(inplace = True),
			nn.Linear(in_features = 256, out_features = 4),
			)

		self.initialize_weights()

		self.fc[2].weight.data.fill_(0)
		self.fc[2].bias.data = torch.FloatTensor([1, 0, 0, 1])



	def STN(self,patch,T_affine):
		grid = F.affine_grid(T_affine,torch.Size([patch.size(0),3,64,64])) / 2.0
		return F.grid_sample(patch, grid)

	def forward(self,x):
		
		patch = x[0]
		
		T_affine = x[1]

		patch0 = patch[:,:,32:96,32:96]
		patch1 = self.STN(patch,T_affine)

		patch0_conv = self.conv3(self.conv2(self.conv1(patch0)))
		patch1_conv = self.conv3(self.conv2(self.conv1(patch1)))

		patch0_conv = patch0_conv.view(-1,2048)
		patch1_conv = patch1_conv.view(-1,2048)

		pred_affine = self.fc(torch.cat((patch0_conv,patch1_conv),dim=1))
		pred_affine = pred_affine.view(-1,2,2)

		return pred_affine

