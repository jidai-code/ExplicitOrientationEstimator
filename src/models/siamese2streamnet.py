import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class Siamese2StreamNet(nn.Module):
	def __init__(self):
		super(Siamese2StreamNet,self).__init__()
		self.setup()

	def setup(self):
		print('Model:\t\tSiamese2StreamNet')

		# net part for center patch (32 x 32)
		self.conv_cp = nn.Sequential(
			nn.Conv2d(in_channels = 1, out_channels = 96, kernel_size = 4, stride = 2),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(kernel_size = 2, stride = 2),
			nn.Conv2d(in_channels = 96, out_channels = 192, kernel_size = 3),
			nn.ReLU(inplace = True),
			nn.Conv2d(in_channels = 192, out_channels = 256, kernel_size = 3),
			nn.ReLU(inplace = True),
			nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3),
			nn.ReLU(inplace = True),
			)

		# net part for downsample patch (32 x 32)
		self.conv_rz = nn.Sequential(
			nn.Conv2d(in_channels = 1, out_channels = 96, kernel_size = 4, stride = 2),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(kernel_size = 2, stride = 2),
			nn.Conv2d(in_channels = 96, out_channels = 192, kernel_size = 3),
			nn.ReLU(inplace = True),
			nn.Conv2d(in_channels = 192, out_channels = 256, kernel_size = 3),
			nn.ReLU(inplace = True),
			nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3),
			nn.ReLU(inplace = True),
			)

		# fully connected part
		self.fc = nn.Sequential(
			nn.Linear(in_features = 1024, out_features = 1024),
			nn.ReLU(inplace = True),
			nn.Linear(in_features = 1024, out_features = 4),
			)

		# initialize weights
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

		p1_c = p1_af[:,:,16:48,16:48]
		p2_c = p2_af[:,:,16:48,16:48]
		p1_o = F.avg_pool2d(p1_af,kernel_size=5,stride=2,padding=2)
		p2_o = F.avg_pool2d(p2_af,kernel_size=5,stride=2,padding=2)

		out_p1_c = self.conv_cp(p1_c)
		out_p2_c = self.conv_cp(p2_c)
		out_p1_o = self.conv_rz(p1_o)
		out_p2_o = self.conv_rz(p2_o)

		out_p1_c = out_p1_c.view(-1,256)
		out_p2_c = out_p2_c.view(-1,256)
		out_p1_o = out_p1_o.view(-1,256)
		out_p2_o = out_p2_o.view(-1,256)
		
		cat_out = torch.cat([out_p1_c,out_p2_c,out_p1_o,out_p2_o],dim=1)	
		
		pred_affine = self.fc(cat_out)
		pred_affine = pred_affine.view(-1,2,2)

		return pred_affine

