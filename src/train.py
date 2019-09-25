import sys
import math
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from scipy.misc import imshow
import models
import datasets
from utils import *
from metrics import *
from schedule import train_schedule_1 as train_schedule
import torch.nn.functional as F

system_check()

iterN = 0
best_score = sys.float_info.max
b_size = 64
bins = 36
lvl = 2

print('========================Dataset=========================')
# choose training dataset
[train_data,valid_data] = datasets.__dict__[train_schedule['dataset']['name']](bins = bins,t_f = True, v_f=False)
train_loader = torch.utils.data.DataLoader(train_data, batch_size = b_size, num_workers = 6, shuffle = True)
train_length = len(train_loader)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size = b_size, num_workers = 6, shuffle = False)
valid_length = len(valid_loader)
print("Training:\t%i" % (train_length))
print("Valid:\t\t%i" % (valid_length))

# choose models
model = models.__dict__[train_schedule['model']['name']](lvl = lvl,bins = bins)
model = model.cuda()

# enable cudnn for static size data
cudnn.benchmark = True

# choose optimizer
optimizer = torch.optim.Adam(model.parameters(),lr = train_schedule['lr']['values'][0],betas = (0.9,0.999))

# logs
train_loss = data_monitor()	# container for loss data
valid_loss = data_monitor()	# container for loss data

train_logger = SummaryWriter(log_dir = train_schedule['log']['TB_path'], comment = 'training')	# tensorboard summary

## main loops
for epoch in range(train_schedule['epoch']['start'],train_schedule['epoch']['end']):
	adjust_lr(optimizer, epoch, train_schedule)
	## training session
	print('=========================TRAIN==========================')
	model.train()
	train_loss.reset()
	for batch_id, [patch_ts,T_affine, bin_id, scale] in enumerate(train_loader):

		# copy to GPU
		patch_cuda = patch_ts.cuda()
		T_affine_cuda = T_affine.cuda()
		bin_id_cuda = bin_id.cuda()
		scale_cuda = scale.cuda().float()
		
		# make differentiable
		patches_var = torch.autograd.Variable(patch_cuda)
		T_affine_var = torch.autograd.Variable(T_affine_cuda)
		bin_id_var = torch.autograd.Variable(bin_id_cuda)
		scale_var = torch.autograd.Variable(scale_cuda)

		inputs = [patches_var,T_affine_var]

		# forward
		pred_angle_var = model(inputs)

		loss = CEL(pred_angle_var, bin_id_var)

		# back prop
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		train_loss.update(loss.data[0],bin_id.size(0))

		train_logger.add_scalar('train_iter_loss', train_loss.val, iterN)
		
		print('|T|\tepoch: %03i/%03i; batch: %04i/%04i (%.2f%%); loss:%.3f (%.3f)' % (epoch,train_schedule['epoch']['end'],batch_id,train_length,100*batch_id/train_length,train_loss.val,train_loss.avg))
		iterN += 1

	train_logger.add_scalar('train_epoch_loss', train_loss.avg, epoch)

	## validation session
	print('=========================VALID==========================')
	model.eval()
	valid_loss.reset()
	for batch_id, [patch_ts,T_affine, bin_id, scale] in enumerate(valid_loader):

		# copy to GPU
		patch_cuda = patch_ts.cuda()
		T_affine_cuda = T_affine.cuda()
		bin_id_cuda = bin_id.cuda()
		scale_cuda = scale.cuda().float()
		
		# make differentiable
		patches_var = torch.autograd.Variable(patch_cuda,volatile = True)
		T_affine_var = torch.autograd.Variable(T_affine_cuda,volatile = True)
		bin_id_var = torch.autograd.Variable(bin_id_cuda,volatile = True)
		scale_var = torch.autograd.Variable(scale_cuda,volatile = True)

		inputs = [patches_var,T_affine_var]

		# forward
		pred_angle_var = model(inputs)
		
		loss = CEL(pred_angle_var, bin_id_var)

		valid_loss.update(loss.data[0],bin_id.size(0))
		
		print('|V|\tepoch: %03i/%03i; batch: %04i/%04i (%.2f%%); loss:%.3f (%.3f)' % (epoch,train_schedule['epoch']['end'],batch_id,valid_length,100*batch_id/valid_length,valid_loss.val,valid_loss.avg))

	train_logger.add_scalar('valid_epoch_loss', valid_loss.avg, epoch)

	## update weights if needed
	if (valid_loss.avg < best_score):
		best_score = valid_loss.avg
		torch.save(model.state_dict(), train_schedule['log']['weights_path'])