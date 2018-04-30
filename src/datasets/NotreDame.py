import random
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from scipy.misc import imread, imshow


root_path = '/home/jdai/Downloads/NotreDame/'
point_list_path = '/home/jdai/Downloads/NotreDame/notredame.out'
image_list_path = '/home/jdai/Downloads/NotreDame/list.txt'

def load_image_and_match(root = root_path, point_list_path = point_list_path, image_list_path = image_list_path):
	size_path = '/home/jdai/Downloads/NotreDame/photosize.txt'
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

def NotreDame():
	match_short, image_list = load_image_and_match()
	train_data = torch_dataset(match_short,image_list)
	return train_data

class torch_dataset(data.Dataset):
	def __init__(self, match_short,image_list):
		self.match_list = match_short
		self.image_list = image_list
	
	def load_patch(self, patch_info):
		img = imread('%s%s' % ('/home/jdai/Downloads/NotreDame/', self.image_list[patch_info[0]]),'L').astype(np.float32)
		h,w = img.shape
		c_x,c_y = patch_info[1:3]	# get location of patch center
		c_x = int(c_x + w/2)	# normalize x to grid
		c_y = int(h/2 - c_y)	# normalize y to grid
		patch = img[c_y-64:c_y+64,c_x-64:c_x+64]
		# p_m = patch.mean()
		# p_std = patch.std()
		# if not(np.isfinite(p_std) and (p_std!=0)):
		# 	p_std = 255.0
		patch = (patch - patch.mean())/patch.std()
		patch = torch.from_numpy(patch).unsqueeze(0)
		return patch

	def __len__(self):
		return len(self.match_list)

	def __getitem__(self, index):
		candidates = self.match_list[index]
		im_id = random.sample(range(0,len(candidates)),2)
		imgs = []
		for i in im_id:
			imgs.append(self.load_patch(candidates[i]))
		#for i in range(len(imgs)):
		#	imgs[i] = torch.from_numpy(np.transpose(imgs[i],(2,0,1))).float()

		return imgs




