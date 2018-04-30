import random
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from scipy.misc import imread, imshow




def load_image_and_match():
	root = '/home/jdai/Downloads/NotreDame/'
	point_list_path = '/home/jdai/Downloads/NotreDame/notredame.out'
	image_list_path = '/home/jdai/Downloads/NotreDame/list.txt'
	size_path = '/home/jdai/Downloads/NotreDame/info.txt'
	
	margin = 64
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

	print('NotreDame dataset')
	print('%i matched points found in %i cameras' % (matches_num, camera_num))

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
				if ((c_x + margin) < size_list[imgid][1]) and ((c_x - margin) > -size_list[imgid][1]):
					if ((c_y + margin) < size_list[imgid][0]) and ((c_y - margin) > -size_list[imgid][0]):
						match_ind.append(int(sample_match[j * 4 + 1]))
						match_ind.append(float(sample_match[j * 4 + 3]))
						match_ind.append(float(sample_match[j * 4 + 4]))
						patch_list.append(match_ind)

	return patch_list, image_list

def NotreDame():
	patch_list, image_list = load_image_and_match()
	train_data = torch_dataset(patch_list,image_list)
	return train_data

class torch_dataset(data.Dataset):
	def __init__(self, patch_list,image_list):
		self.patch_list = patch_list
		self.image_list = image_list
	
	def load_patch(self, patch_info):
		img = imread('%s%s' % ('/home/jdai/Downloads/NotreDame/', self.image_list[patch_info[0]])).astype(np.float32)
		h,w,_ = img.shape
		x,y = patch_info[1:3]	# get location of patch center
		c_id = int(w/2 + x - 0.5)	# normalize x to grid
		r_id = int(h/2 - y - 0.5)	# normalize y to grid
		patch = img[r_id-64:r_id+64,c_id-64:c_id+64,:]
		patch = (patch - patch.mean())/patch.std()
		patch = np.transpose(patch,[2,0,1])
		patch_ts = torch.from_numpy(patch)
		return patch_ts

	def __len__(self):
		return len(self.patch_list)

	def __getitem__(self, index):
		patch_info = self.patch_list[index]
		return self.load_patch(patch_info)

train_data = NotreDame()
train_loader = torch.utils.data.DataLoader(train_data, batch_size = 32, num_workers = 4, shuffle = True)
train_length = len(train_loader)
for batch_id, patches in enumerate(train_loader):
	patches_var = torch.autograd.Variable(patches.cuda())

	print('batch:%i/%i' % (batch_id,train_length))