import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from scipy.misc import imshow
import models
import datasets
from utils import *
from metrics import *
from train_schedule import train_schedule_1 as schedule

system_check()

iterN = 0
best_score = sys.float_info.max

print('========================Dataset=========================')
# choose training dataset
[train_data,valid_data] = datasets.__dict__[schedule['dataset']['name']]()
train_loader = torch.utils.data.DataLoader(train_data, batch_size = 32, num_workers = 4, shuffle = True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size = 32, num_workers = 4, shuffle = False)
train_length = len(train_loader)
valid_length = len(valid_loader)

# choose models
model = models.__dict__[schedule['model']['name']]()
model = model.cuda()

# enable cudnn for static size data
cudnn.benchmark = True

# choose optimizer
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4,betas=(0.9,0.999))

# logs
train_loss = data_monitor()	# container for loss data
valid_loss = data_monitor()	# container for loss data
train_logger = SummaryWriter(log_dir=schedule['log']['TB_path'], comment='training')	# tensorboard summary

## main loops
for epoch in range(schedule['epoch']['start'],schedule['epoch']['end']):
	adjust_lr(optimizer, epoch, schedule)
	## training session
	print('=========================TRAIN==========================')
	model.train()
	train_loss.reset()
	for batch_id, [patch1,patch2,T_inv] in enumerate(train_loader):

		# copy to GPU
		patch1_cuda = patch1.cuda()
		patch2_cuda = patch2.cuda()
		T_inv_cuda = T_inv.cuda()

		# make differentiable
		patches_var = torch.autograd.Variable(torch.cat([patch1_cuda,patch2_cuda],dim=1))
		T_inv_var = torch.autograd.Variable(T_inv_cuda)

		# forward
		output_var = model(patches_var)

		loss = l2norm(T_inv_var,output_var)

		# back prop
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		train_loss.update(loss.data[0],T_inv_var.size(0))
		train_logger.add_scalar('train_error_iter', train_loss.val, iterN)
		print('|T|\tepoch: %03i/%03i; batch: %04i/%04i (%.2f%%); loss:%.3f (%.3f)' % (epoch,schedule['epoch']['end'],batch_id,train_length,100*batch_id/train_length,train_loss.val,train_loss.avg))
		iterN += 1
	train_logger.add_scalar('train_error_epoch', train_loss.avg, epoch)

	## validation session
	print('=========================VALID==========================')
	model.eval()
	valid_loss.reset()
	for batch_id, [patch1,patch2,T_inv] in enumerate(valid_loader):

		# copy to GPU
		patch1_cuda = patch1.cuda()
		patch2_cuda = patch2.cuda()
		T_inv_cuda = T_inv.cuda()

		# make differentiable
		patches_var = torch.autograd.Variable(torch.cat([patch1_cuda,patch2_cuda],dim=1),volatile = True)
		T_inv_var = torch.autograd.Variable(T_inv_cuda,volatile = True)

		# forward
		output_var = model(patches_var)

		loss = l2norm(T_inv_var,output_var)
		valid_loss.update(loss.data[0],T_inv_var.size(0))
		print('|V|\tepoch: %03i/%03i; batch: %04i/%04i (%.2f%%); loss:%.3f (%.3f)' % (epoch,schedule['epoch']['end'],batch_id,valid_length,100*batch_id/valid_length,valid_loss.val,valid_loss.avg))
	train_logger.add_scalar('valid_error_epoch', valid_loss.avg, epoch)

	if (valid_loss.avg < best_score):
		best_score = valid_loss.avg
		torch.save(model.state_dict(), schedule['log']['weights_path'])
