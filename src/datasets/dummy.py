import random
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from scipy.misc import imread, imshow

root_path = '/home/jdai/Downloads/NotreDame/'
image_list_path = '/home/jdai/Downloads/NotreDame/list.txt'
image_info = '/home/jdai/Downloads/NotreDame/info.txt'

with open(image_list_path,'r') as fp:
	line = fp.readline()
	image_list = []
	while line:
		image_list.append(line.strip('\n'))
		line = fp.readline()
with open(image_info,'w') as fw:
	for i in range(len(image_list)):
		img = imread('%s%s' % (root_path,image_list[i]))
		ndim = img.ndim
		h = img.shape[0]
		w = img.shape[1]
		fw.write('%i,%i,%i\n' % (h,w,ndim//3))
		print('%.2f%%\t%i x %i (%s)' % (100*i/len(image_list),h,w,'color' if (ndim==3) else 'bw'))

