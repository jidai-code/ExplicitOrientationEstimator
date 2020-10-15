import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

def conv2d(in_channels, out_channels, kernel_size, stride, padding):
	'''
	2d convolution layers
	'''
	return nn.Sequential(
		nn.Conv2d(in_channels = in_channels,out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = False),
		nn.BatchNorm2d(out_channels),
		nn.ReLU(inplace = True),
		)

def make_layers(cfg, in_channels):
	layers = []
	for l in cfg:
		if l == 'M':
			layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
		else:
			layers += [conv2d(in_channels=in_channels, out_channels=l, kernel_size=3, stride=1, padding=1)]
			in_channels = l	
	return nn.Sequential(*layers)

class BaseNet(nn.Module):
	'''
	basic network model
	'''
	def __init__(self):
		super().__init__()
		self.setup()
		self.initialize_weights()
		self.customize_weights()
	
	def setup(self):
		'''
		setup model
		'''
		return

	def initialize_weights(self):
		'''
		initialize weights in network
		'''
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				if m.bias is not None:
					init.uniform(m.bias)
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)

	def customize_weights(self):
		'''
		customize weights if needed by children
		'''
		return

	def forward(self):
		'''
		forward function
		'''
		return