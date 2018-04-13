import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class SiameseNet(nn.Module):
	def __init__(self):
		super(SiameseNet,self).__init__()
		self.setup()

	def setup(self):
		print('Model:\t\tSiameseNet')

		self.conv1 = nn.Sequential(
			nn.Conv2d(in_channels = 1, out_channels = 96, kernel_size = 7, stride = 3),
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
			nn.Linear(in_features = 512, out_features = 512),
			nn.ReLU(inplace = True),
			nn.Linear(in_features = 512, out_features = 4),
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

		patches = x[0]
		p1 = patches[:,0,:,:].unsqueeze(1)
		p2 = patches[:,1,:,:].unsqueeze(1)

		T_affine = x[1]
		T_affine1 = T_affine[:,0:2,:]
		T_affine2 = T_affine[:,2:4,:]

		p1_af = self.STN(p1,T_affine1)
		p2_af = self.STN(p2,T_affine2)

		output1 = self.conv3(self.conv2(self.conv1(p1_af)))
		output2 = self.conv3(self.conv2(self.conv1(p2_af)))

		output1 = output1.view(-1,256)
		output2 = output2.view(-1,256)
		
		cat_out = torch.cat([output1,output2],dim=1)	
		
		pred_affine = self.fc(cat_out)
		pred_affine = pred_affine.view(-1,2,2)

		return pred_affine

