import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class StackNet(nn.Module):
	def __init__(self):
		super(StackNet,self).__init__()
		self.setup()

	def setup(self):
		print('Model:\t\tStackNet')

		self.conv1 = nn.Sequential(
			nn.Conv2d(in_channels = 6, out_channels = 96, kernel_size = 7, stride = 3),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(kernel_size = 2, stride = 2),
			)
		self.conv2 = nn.Sequential(
			nn.Conv2d(in_channels = 96, out_channels = 192, kernel_size = 5),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(kernel_size = 2, stride = 2),
			)
		self.conv3 = nn.Sequential(
			nn.Conv2d(in_channels = 192, out_channels = 256, kernel_size = 3),
			nn.ReLU(inplace = True),
			)
		self.fc = nn.Sequential(
			nn.Linear(in_features = 256, out_features = 256),
			nn.ReLU(inplace = True),
			nn.Linear(in_features = 256, out_features = 4),
			)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				if m.bias is not None:
					init.uniform(m.bias)
				init.xavier_uniform(m.weight)

			if isinstance(m, nn.ConvTranspose2d):
				if m.bias is not None:
					init.uniform(m.bias)
				init.xavier_uniform(m.weight)

		self.fc[2].weight.data.fill_(0)
		self.fc[2].bias.data = torch.FloatTensor([1, 0, 0, 1])

	def STN(self,patch,T_affine):
		grid = F.affine_grid(T_affine,patch.size())
		return F.grid_sample(patch, grid)

	def forward(self,x):
		
		conv1_out = self.conv1(x)
		conv2_out = self.conv2(conv1_out)
		conv3_out = self.conv3(conv2_out)
		conv3_out = conv3_out.view(-1,256)
		pred_affine = self.fc(conv3_out)
		pred_affine = pred_affine.view(-1,2,2)

		return pred_affine

