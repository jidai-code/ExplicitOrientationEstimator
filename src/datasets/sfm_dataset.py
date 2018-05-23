import random
import sys
import time
import math
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from scipy.misc import imread, imshow

# progress bar (copied from "https://gist.github.com/vladignatyev/06860ec2040cb497f0f3")
def progress(count, total):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s\r' % (bar, percents, '%'))
    sys.stdout.flush()

def load_image_and_match(root,point_list_path,image_list_path,size_path):

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
			tsplit[2] = int(tsplit[2])
			size_list.append(tsplit)
			line = fp.readline()

	metadata = point_list_file[1].split()
	camera_num = int(metadata[0])
	matches_num = int(metadata[1])

	print('Name:\t\tNotreDame')
	#print('\t%i matched points found in %i cameras' % (matches_num, camera_num))

	line_one = camera_num * 5 + 4
	line_end = camera_num * 5 + 4 + (matches_num - 1) * 3

	matches_long = []

	for i in range(line_one,line_end + 1,3):
		matches_long.append(point_list_file[i].split())

	patch_list = []

	for i in range(len(matches_long)):
		sample_match = matches_long[i]
		for j in range(int(sample_match[0])):
			match_ind = []
			imgid = int(sample_match[j * 4 + 1])
			c_x = float(sample_match[j * 4 + 3])
			c_y = float(sample_match[j * 4 + 4])
			if (size_list[imgid][2] == 1):
				if ((int(c_x) + margin) < size_list[imgid][1]) and ((int(c_x) - margin) > -size_list[imgid][1]):
					if ((int(c_y) + margin) < size_list[imgid][0]) and ((int(c_y) - margin) > -size_list[imgid][0]):
						match_ind.append(int(sample_match[j * 4 + 1]))
						match_ind.append(float(sample_match[j * 4 + 3]))
						match_ind.append(float(sample_match[j * 4 + 4]))
						patch_list.append(match_ind)

	return patch_list, image_list

class torch_dataset(data.Dataset):
	def __init__(self, patch_list,image_list,bins,fast):
		self.patch_list = patch_list
		self.image_list = image_list
		self.bins = bins
		self.fast = fast
		if self.fast:
			self.images = []
			self.load_images()

	def load_images(self):
		time0 = time.time()
		for i in range(len(self.image_list)):
			progress(i,len(self.image_list)-1)
			self.images.append(imread('%s%s' % ('/home/jdai/Downloads/NotreDame/', self.image_list[i])))
		print('\n\tcompleted in %.3f sec' % (time.time()-time0))	

	def read_patch_fast(self, patch_info):
		img = self.images[patch_info[0]].astype(np.float32)
		imh,imw,_ = img.shape
		c_x,c_y = patch_info[1:3]	# get location of patch center
		c_id = int(c_x + imw/2)	# normalize x to grid
		r_id = int(imh/2 - c_y)	# normalize y to grid
		patch = img[r_id-64:r_id+64,c_id-64:c_id+64,:] / 255.0
		patch = np.transpose(patch,[2,0,1])
		patch_ts = torch.from_numpy(patch)
		return patch_ts


	def load_patch(self, patch_info):
		img = imread('%s%s' % ('/home/jdai/Downloads/NotreDame/', self.image_list[patch_info[0]])).astype(np.float32)
		imh,imw,_ = img.shape
		c_x,c_y = patch_info[1:3]	# get location of patch center
		c_id = int(c_x + imw/2)	# normalize x to grid
		r_id = int(imh/2 - c_y)	# normalize y to grid
		patch = img[r_id-64:r_id+64,c_id-64:c_id+64,:] / 255.0
		patch = np.transpose(patch,[2,0,1])
		patch_ts = torch.from_numpy(patch)
		return patch_ts

	def get_affine(self):
		# generate resize scale and rotation theta (rad)
		scale = float(random.uniform(2/3,3/2))
		theta = random.uniform(0,2*math.pi)

		# disect T_affine into 3 components for more trackability
		T_scale = np.matrix([[scale,0,0],[0,scale,0],[0,0,1]])	# scale T
		T_rot	= np.matrix([[math.cos(theta),math.sin(theta),0],[-math.sin(theta),math.cos(theta),0],[0,0,1]])	# rot T

		# create affine T
		T_affine = T_rot * T_scale

		T_affine = T_affine[0:2,0:3].astype(np.float32)

		T_affine = torch.from_numpy(T_affine)

		bin_id = round(theta / (2 * math.pi / self.bins)) % self.bins

		return T_affine, bin_id, scale

	def __len__(self):
		return len(self.patch_list)

	def __getitem__(self, index):
		patch_info = self.patch_list[index]
		if self.fast:
			patch_ts = self.read_patch_fast(patch_info)
		else:
			patch_ts = self.load_patch(patch_info)
		[T_affine,bin_id,scale] = self.get_affine()
		return [patch_ts,T_affine,bin_id,scale]

def NotreDame(bins):
	tv_split = 0.99
	root = '/home/jdai/Downloads/NotreDame/'
	point_list_path = '/home/jdai/Downloads/NotreDame/notredame.out'
	image_list_path = '/home/jdai/Downloads/NotreDame/list.txt'
	size_path = '/home/jdai/Downloads/NotreDame/info.txt'
	patch_list, image_list = load_image_and_match(root,point_list_path,image_list_path,size_path)

	## split into training and validation parts
	patch_t_list = patch_list[:int(len(patch_list) * tv_split)]
	patch_v_list = patch_list[int(len(patch_list) * tv_split):]
	train_data = torch_dataset(patch_t_list,image_list,bins,fast = True)
	valid_data = torch_dataset(patch_v_list,image_list,bins,fast = False)
	return [train_data,valid_data]