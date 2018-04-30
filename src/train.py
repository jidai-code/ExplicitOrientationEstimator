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

print('========================training========================')
# choose training dataset
train_data = datasets.__dict__[schedule['dataset']['name']]()
train_loader = torch.utils.data.DataLoader(train_data, batch_size = 32, num_workers = 4, shuffle = True)
train_length = len(train_loader)

# choose models
model = models.__dict__[schedule['model']['name']]()
model = model.cuda()

# enable cudnn for static size data
cudnn.benchmark = True

# choose optimizer
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4,betas=(0.9,0.999))

# logs
train_loss = data_monitor()	# container for loss data
train_logger = SummaryWriter(log_dir=schedule['log']['TB_path'], comment='training')	# tensorboard summary

# start training
print('=========================start==========================')
for epoch in range(schedule['epoch']['start'],schedule['epoch']['end']):
	adjust_lr(optimizer, epoch, schedule)
	model.train()
	train_loss.reset()
	for batch_id, [patches, T_affine, T_inv] in enumerate(train_loader):
	
		# copy to GPU
		patches_cuda = [patch.cuda() for patch in patches]
		T_affine_cuda = [T_a.cuda() for T_a in T_affine]
		T_inv_cuda = T_inv.cuda()

		# make differentiable
		patches_var = torch.autograd.Variable(torch.cat(patches_cuda,dim = 1))
		T_affine_var = torch.autograd.Variable(torch.cat(T_affine_cuda,dim = 1))
		T_inv_var = torch.autograd.Variable(T_inv_cuda)
		input_var = [patches_var,T_affine_var]

		# forward
		output_var = model(input_var)

		loss = l2norm(T_inv_var,output_var)

		# back prop
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		train_loss.update(loss.data[0],T_inv_var.size(0))
		train_logger.add_scalar('error_iter', train_loss.val, iterN)
		print('epoch: %03i/%03i; batch: %04i/%04i (%.2f%%); loss:%.3f (%.3f)' % (epoch,schedule['epoch']['end'],batch_id,train_length,100*batch_id/train_length,train_loss.val,train_loss.avg))
		iterN += 1
	train_logger.add_scalar('error_epoch', train_loss.avg, epoch)
	if (train_loss.avg < best_score):
		best_score = train_loss.avg
		torch.save(model.state_dict(), schedule['log']['weights_path'])
