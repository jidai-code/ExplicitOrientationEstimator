import sys
import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from dataset import NotreDame
from scipy.misc import imshow
from models.stacknet import StackNet

iterN = 0
best_score = sys.float_info.max

# data monitor class
class data_monitor(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.avg = 0
		self.val = 0
		self.count = 0
		self.sum = 0

	def update(self, val, n = 1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def l2norm(target_var,output_var):
	target_var = target_var.view(-1,4)
	output_var = output_var.view(-1,4)
	loss = torch.norm(target_var-output_var,p=2,dim=1).mean()
	return loss

train_data = NotreDame()
train_loader = torch.utils.data.DataLoader(train_data, batch_size = 32, num_workers = 4, shuffle = True)
train_length = len(train_loader)

model = StackNet()
model = model.cuda()

cudnn.benchmark = True
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001,betas=(0.9,0.999))

train_loss = data_monitor()
train_logger = SummaryWriter(log_dir = '/home/jdai/Documents/OrientationNet/experiments/stacknet/', comment = 'training')

for epoch in range(0,100):
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
		print('epoch: %i/%i; batch: %i/%i (%.2f%%); loss:%.3f (%.3f)' % (epoch,100,batch_id,train_length,100*batch_id/train_length,train_loss.val,train_loss.avg))
		iterN += 1
	train_logger.add_scalar('error_epoch', train_loss.avg, epoch)
	if (train_loss.avg < best_score):
		best_score = train_loss.avg
		torch.save(model.state_dict(), '/home/jdai/Documents/OrientationNet/experiments/stacknet/checkpoint.pth.tar')