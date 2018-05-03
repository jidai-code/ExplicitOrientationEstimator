import random
import sys
import time
import math
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from scipy.misc import imread, imshow
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import models
from utils import *
from metrics import *
from train_schedule import train_schedule_1 as schedule

root_path = '/home/jd/Downloads/NotreDame/'
point_list_path = '/home/jd/Downloads/NotreDame/notredame.out'
image_list_path = '/home/jd/Downloads/NotreDame/list.txt'

def load_image_and_match(root = root_path, point_list_path = point_list_path, image_list_path = image_list_path):
	size_path = '/home/jd/Downloads/NotreDame/info.txt'
	margin = 128
	with open(point_list_path,'r') as fp:
		line = fp.readline()
		point_list_file = []
		while line:
			point_list_file.append(line.strip('\n'))
			line = fp.readline()

	with open(image_list_path,'r') as fp:
		line = fp.readline()
		image_list = []
		while line:
			image_list.append(line.strip('\n'))
			line = fp.readline()

	with open(size_path,'r') as fp:
		line = fp.readline()
		size_list = []
		while line:
			tline = line.strip('\n')
			tsplit = tline.split(',')
			tsplit[0] = int(tsplit[0])//2
			tsplit[1] = int(tsplit[1])//2
			size_list.append(tsplit)
			line = fp.readline()

	metadata = point_list_file[1].split()
	camera_num = int(metadata[0])
	matches_num = int(metadata[1])

	print('NotreDame dataset')
	print('%i matched points found in %i cameras' % (matches_num, camera_num))

	line_one = camera_num * 5 + 4
	line_end = camera_num * 5 + 4 + (matches_num - 1) * 3

	matches_long = []

	for i in range(line_one,line_end + 1,3):
		matches_long.append(point_list_file[i].split())

	match_short = []

	for i in range(len(matches_long)):
		sample_match = matches_long[i]
		match_piece = []
		for j in range(int(sample_match[0])):
			match_ind = []
			imgid = int(sample_match[j * 4 + 1])
			c_x = float(sample_match[j * 4 + 3])
			c_y = float(sample_match[j * 4 + 4])
			if ((c_x + margin) < size_list[imgid][1]) and ((c_x - margin) > -size_list[imgid][1]):
				if ((c_y + margin) < size_list[imgid][0]) and ((c_y - margin) > -size_list[imgid][0]):
					match_ind.append(int(sample_match[j * 4 + 1]))
					match_ind.append(float(sample_match[j * 4 + 3]))
					match_ind.append(float(sample_match[j * 4 + 4]))
					match_piece.append(match_ind)
		if (len(match_piece) >= 2):
			match_short.append(match_piece)

	return match_short, image_list

[match_short, image_list] = load_image_and_match()

model = models.__dict__['StackNet']()
model = model.cuda()
oldweights = torch.load('/home/jd/Documents/TransNet/TransNet/experiments/StackNet/checkpoint.pth.tar')
if 'state_dict' in oldweights.keys():
	model.load_state_dict(oldweights['state_dict'])
else:
	model.load_state_dict(oldweights)
model.eval()

for idd in range(3000,4001):
	candidates = match_short[idd]
	patch1 = candidates[0]
	patch2 = candidates[1]

	x1,y1 = patch1[1:3]
	x2,y2 = patch2[1:3]

	img1 = imread('%s%s' % ('/home/jd/Downloads/NotreDame/', image_list[patch1[0]])).astype(np.float32)
	img2 = imread('%s%s' % ('/home/jd/Downloads/NotreDame/', image_list[patch2[0]])).astype(np.float32)
	img1 = img1 / 255.0
	img2 = img2 / 255.0

	imh1,imw1, _ = img1.shape
	imh2,imw2, _ = img2.shape

	img1 = np.transpose(img1,[2,0,1])
	img2 = np.transpose(img2,[2,0,1])

	img1_ts = torch.from_numpy(img1).unsqueeze(0)
	img2_ts = torch.from_numpy(img2).unsqueeze(0)

	t_id = np.matrix([[1,0,0],[0,1,0]],dtype=np.float32)
	t_id_ts = torch.from_numpy(t_id).unsqueeze(0)
	grid_id1 = F.affine_grid(t_id_ts,torch.Size([1,3,64,64]))
	grid_id2 = F.affine_grid(t_id_ts,torch.Size([1,3,64,64]))

	grid_id1[:,:,:,0] = grid_id1[:,:,:,0] * 64 / imw1
	grid_id1[:,:,:,1] = grid_id1[:,:,:,1] * 64 / imh1
	grid_id1[:,:,:,0] = grid_id1[:,:,:,0] + 2 * x1 / (imw1 - 1)
	grid_id1[:,:,:,1] = grid_id1[:,:,:,1] - 2 * y1 / (imh1 - 1)

	grid_id2[:,:,:,0] = grid_id2[:,:,:,0] * 64 / imw2
	grid_id2[:,:,:,1] = grid_id2[:,:,:,1] * 64 / imh2
	grid_id2[:,:,:,0] = grid_id2[:,:,:,0] + 2 * x2 / (imw2 - 1)
	grid_id2[:,:,:,1] = grid_id2[:,:,:,1] - 2 * y2 / (imh2 - 1)

	patch1_ts = F.grid_sample(img1_ts,grid_id1).data
	patch2_ts = F.grid_sample(img2_ts,grid_id2).data

	patch1_cuda = patch1_ts.cuda()
	patch2_cuda = patch2_ts.cuda()
	patches_var = torch.autograd.Variable(torch.cat([patch1_cuda,patch2_cuda],dim=1))
	output_var = model(patches_var)
	output = output_var.data
	T_1 = torch.FloatTensor(torch.Size([1,2,3])).fill_(0)
	T_1[0,0:2,0:2] = output[0,0:2,0:2]
	grid_id3 = F.affine_grid(T_1,torch.Size([1,3,64,64]))
	grid_id3[:,:,:,0] = grid_id3[:,:,:,0] * 64 / imw2
	grid_id3[:,:,:,1] = grid_id3[:,:,:,1] * 64 / imh2
	grid_id3[:,:,:,0] = grid_id3[:,:,:,0] + 2 * x2 / (imw2 - 1)
	grid_id3[:,:,:,1] = grid_id3[:,:,:,1] - 2 * y2 / (imh2 - 1)

	patch3_ts = F.grid_sample(img2_ts,grid_id3).data

	patch1 = np.transpose(patch1_ts.numpy().squeeze(0),[1,2,0])
	patch2 = np.transpose(patch2_ts.numpy().squeeze(0),[1,2,0])
	patch3 = np.transpose(patch3_ts.numpy().squeeze(0),[1,2,0])

	imshow(np.concatenate([patch1,patch2,patch3],axis=1))