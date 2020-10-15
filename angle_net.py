import torch
import torch.nn as nn
import torch.nn.functional as F
from net_utils import *
import numpy as np


class Angle_Net(BaseNet):
	def __init__(self):
		BaseNet.__init__(self)

	def setup(self):
		cfg = [64, 64, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256, 'M']
		self.ConvNet = make_layers(cfg=cfg, in_channels=1)
		self.FC = nn.Sequential(
			nn.Linear(in_features=1024, out_features=1024),
			nn.ReLU(inplace=True),
			nn.Linear(in_features=1024, out_features=1024),
			nn.ReLU(inplace=True),
			nn.Linear(in_features=1024, out_features=1),
			)

	def forward(self, x):


		patch0 = x[0]
		patch1 = x[1]

		conv_out0 = self.ConvNet(patch0)
		conv_out1 = self.ConvNet(patch1)

		conv_out0 = conv_out0.view(-1, 1024)
		conv_out1 = conv_out1.view(-1, 1024)

		pred_angle0 = self.FC(conv_out0)
		pred_angle1 = self.FC(conv_out1)

		return [pred_angle0, pred_angle1]




