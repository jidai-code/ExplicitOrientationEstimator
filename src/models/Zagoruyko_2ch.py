import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from .net_utils import *

class Zagoruyko_2ch(base_net):
	def __init__(self):
		base_net.__init__(self)

	def setup(self):
		print('STN:\t\tZagoruyko_TwoChannels')

		self.conv1 = nn.Sequential(
			conv(in_planes = 6, out_planes = 96, kernel_size = 7, stride = 3),
			nn.MaxPool2d(kernel_size = 2, stride = 2),
			)
		self.conv2 = nn.Sequential(
			conv(in_planes = 96, out_planes = 192, kernel_size = 5),
			nn.MaxPool2d(kernel_size = 2, stride = 2),
			)
		self.conv3 = nn.Sequential(
			conv(in_planes = 192, out_planes = 256, kernel_size = 3),
			)
		self.fc = nn.Sequential(
			nn.Linear(in_features = 256, out_features = 256),
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

		inputs = torch.cat((patch0,patch1),dim=1)

		conv1_out = self.conv1(inputs)
		conv2_out = self.conv2(conv1_out)
		conv3_out = self.conv3(conv2_out)
		conv3_out = conv3_out.view(-1,256)
		pred_affine = self.fc(conv3_out)
		pred_affine = pred_affine.view(-1,2,2)

		return pred_affine
