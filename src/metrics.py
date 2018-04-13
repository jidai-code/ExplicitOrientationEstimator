import torch
import numpy as np

def l2norm(target_var,output_var):
	target_var = target_var.view(-1,4)
	output_var = output_var.view(-1,4)
	loss = torch.norm(target_var-output_var,p=2,dim=1).mean()
	return loss