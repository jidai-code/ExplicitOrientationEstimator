import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

# define convolutional net
def conv(in_planes, out_planes, kernel_size, stride = 1, padding = 0):
	return nn.Sequential(
		nn.Conv2d(in_planes, out_planes, kernel_size = kernel_size, stride = stride, padding = padding, bias = False),
		nn.ReLU(inplace=True),
		nn.BatchNorm2d(out_planes),
		)

# base net
class base_net(nn.Module):
	def __init__(self):
		super().__init__()
		self.setup()
	
	def setup(self):
		return

	def initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				if m.bias is not None:
					init.uniform(m.bias)
				init.xavier_uniform(m.weight)

			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant(m.weight, 1)
				nn.init.constant(m.bias, 0)

			elif isinstance(m, nn.Linear):
				nn.init.normal(m.weight, 0, 0.01)
				nn.init.constant(m.bias, 0)

	def forward(self):
		return