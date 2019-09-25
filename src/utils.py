import platform
import torch
import math
import torch.nn.functional as F

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
	print('========================TransNet========================')
	print('Python:\t\t%s' % (platform.python_version()))
	print('PyTorch:\t%s' % (torch.__version__))
	print('GPU:\t\t%s' % (torch.cuda.get_device_name(0)))


# regress predict angles
def get_angle(pred_bins,bins):
	pred_bin_max = F.log_softmax(pred_bins, 1)
	m_value,m_indice = torch.max(pred_bin_max,1)
	m_indice = m_indice.data.cpu().numpy()

	pred_id = m_indice[0]
	pred_id_l = (pred_id-1) % bins
	pred_id_r = (pred_id+1) % bins

	y0 = pred_bin_max[0,pred_id_l].data.cpu()
	y1 = pred_bin_max[0,pred_id].data.cpu()
	y2 = pred_bin_max[0,pred_id_r].data.cpu()
	A = float(y0/2-y1+y2/2)
	B = float(y2/2-y0/2)
	C = float(y1)
	x_hat = - B / (2*A)
	pred_angle = (x_hat+pred_id)*(2*math.pi/bins)
	return pred_angle
