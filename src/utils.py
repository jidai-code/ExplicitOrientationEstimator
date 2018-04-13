import platform
import torch

# loss monitor
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

# adjust learning rate according to schedule
def adjust_lr(optimizer, epoch, schedule):
	checkpoints = schedule['lr']['ckpts']
	values = schedule['lr']['values'][1:]
	assert(len(values) == len(checkpoints))
	for i in range(len(checkpoints)):
		if (epoch == checkpoints[i]):
			for param_group in optimizer.param_groups:
				param_group['lr'] = values[i]
			print('adjust learning rate to:%.6f' % (values[i]))

# print system info
def system_check():
	print('======================system check======================')
	print('Python:\t\t%s' % (platform.python_version()))
	print('PyTorch:\t%s' % (torch.__version__))
	print('GPU:\t\t%s' % (torch.cuda.get_device_name(0)))

