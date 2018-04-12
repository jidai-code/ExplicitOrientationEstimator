import random
import math
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from scipy.misc import imread, imshow

def readMetaData(root):

	path = '%s/info.txt' % (root)

	with open(path,'r') as fp:
		line = fp.readline()
		point_id = []
		while line:
			tline = line.strip('\n')
			tsplit = tline.split(' ')
			point_id.append(int(tsplit[0]))
			line = fp.readline()

	meta_data = []
	if_new_pt = False
	previous_pt = -1

	for i in range(len(point_id)):
		# read current point index
		current_pt = point_id[i]

		# computer img_id, row_id, col_id
		img_id = i // 256
		row_id = (i % 256) // 16
		col_id = i % 16
	
		# check if processing new point
		if (previous_pt != current_pt):
			if_new_pt = True
		else:
			if_new_pt = False

		# if processing new point, save info of old point and create new list for new point
		if if_new_pt:
			if 'pt_data' in locals():
				meta_data.append(pt_data)
			pt_data = []

		pt_data.append([img_id,row_id,col_id])
		previous_pt = current_pt
	
	# append last point
	meta_data.append(pt_data)

	return meta_data

def readPatch(root,img_id,row_id,col_id):
	img = imread('%s/patches%04i.bmp' % (root, img_id))
	return img[row_id*64:(row_id+1)*64,col_id*64:(col_id+1)*64]

def showPoint(root,meta_data,pid):
	pt_info = meta_data[pid]
	patches = []
	for i in range(len(pt_info)):
		patches.append(readPatch(root,pt_info[i][0],pt_info[i][1],pt_info[i][2]))
	imshow(np.concatenate(patches,axis = 1))

class torch_dataset(data.Dataset):
	def __init__(self, meta_data, root):
		self.meta_data = meta_data
		self.root = root

	def __len__(self):
		return len(self.meta_data)

	def load_patch(self, patch_info):

		# unload corresponding image id, row id and column id
		img_id = patch_info[0]
		row_id = patch_info[1]
		col_id = patch_info[2]

		# read image and crop
		img = imread('%s/patches%04i.bmp' % (self.root, img_id)).astype(np.float32)
		patch = img[row_id*64:(row_id+1)*64,col_id*64:(col_id+1)*64]

		# normalize and convert to tensor
		patch = (patch - patch.mean()) / patch.std()
		patch = torch.from_numpy(patch).unsqueeze(0)

		return patch

	def create_affine(self):
		# generate resize scale and rotation theta (rad)
		scale1 = random.uniform(0.5,0.75)
		theta1 = random.uniform(-math.pi,math.pi)
		scale2 = random.uniform(0.5,0.75)
		theta2 = theta1 + random.uniform(-math.pi/2,math.pi/2)

		# disect T_affine into 3 components for more trackability
		T_scale1 = np.matrix([[scale1,0,0],[0,scale1,0],[0,0,1]])	# scale T
		T_shear1 = np.matrix([[1,random.uniform(-0.2,0.2),0],[random.uniform(-0.2,0.2),1,0],[0,0,1]])	# shear T
		T_rot1	= np.matrix([[math.cos(theta1),math.sin(theta1),0],[-math.sin(theta1),math.cos(theta1),0],[0,0,1]])	# rot T

		T_scale2 = np.matrix([[scale2,0,0],[0,scale2,0],[0,0,1]])	# scale T
		T_shear2 = np.matrix([[1,random.uniform(-0.2,0.2),0],[random.uniform(-0.2,0.2),1,0],[0,0,1]])	# shear T
		T_rot2	= np.matrix([[math.cos(theta2),math.sin(theta2),0],[-math.sin(theta2),math.cos(theta2),0],[0,0,1]])	# rot T
		# combine to T_affine and fine inverse T_affine (Ground Truth)
		T_affine1 = T_rot1 * T_shear1 * T_scale1
		T_affine2 = T_rot2 * T_shear2 * T_scale2
		T_inv = T_affine1 * np.linalg.inv(T_affine2)

		T_affine1 = T_affine1[0:2,0:3].astype(np.float32)
		T_affine2 = T_affine2[0:2,0:3].astype(np.float32)
		T_inv = T_inv[0:2,0:2].astype(np.float32)

		# convert to tensor
		T_affine1 = torch.from_numpy(T_affine1)
		T_affine2 = torch.from_numpy(T_affine2)
		T_inv = torch.from_numpy(T_inv)
		#T_inv = T_inv.view(-1,2,3)

		return [T_affine1,T_affine2], T_inv

	def __getitem__(self, index):
		# fetch point
		candidate = self.meta_data[index]

		# randomly pick two patches corresponding that point
		pc_id = random.sample(range(0,len(candidate)),2)

		# read patches from images
		patches = []
		for i in pc_id:
			patches.append(self.load_patch(candidate[i]))

		# create random affine transformation and GT
		T_affine, T_inv = self.create_affine()

		return patches, T_affine, T_inv



def NotreDame():
	root = '/home/jdai/Downloads/datas/notredame'
	meta_data = readMetaData(root)
	train_data = torch_dataset(meta_data,root)
	return train_data


