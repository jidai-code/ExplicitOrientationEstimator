from angle_net import Angle_Net
import torch
import numpy as np
import cv2 as cv
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from matplotlib import pyplot as plt

'''
utils from kwang moo yi's code
'''
# opencv
IDX_X, IDX_Y, IDX_SIZE, IDX_ANGLE, IDX_RESPONSE, IDX_OCTAVE = (
    0, 1, 2, 3, 4, 5)  # , IDX_CLASSID not used
# vgg affine
IDX_a, IDX_b, IDX_c = (6, 7, 8)
# vlfeat Affine [A0, A1; A2, A3]?
# NOTE the row-major colon-major adaptation here
IDX_A0, IDX_A2, IDX_A1, IDX_A3 = (9, 10, 11, 12)

def loadKpListFromTxt(kp_file_name):

    # Open keypoint file for read
    kp_file = open(kp_file_name, 'rb')

    # skip the first two lines
    kp_line = kp_file.readline()
    kp_line = kp_file.readline()

    kp_list = []
    num_elem = -1
    while True:
        # read a line from file
        kp_line = kp_file.readline()
        # check EOF
        if not kp_line:
            break
        # split read information
        kp_info = kp_line.split()
        parsed_kp_info = []
        for idx in range(len(kp_info)):
            parsed_kp_info += [float(kp_info[idx])]
        parsed_kp_info = np.asarray(parsed_kp_info)

        if num_elem == -1:
            num_elem = len(parsed_kp_info)
        else:
            assert num_elem == len(parsed_kp_info)

        # IMPORTANT: make sure this part corresponds to the one in
        # opencv_kp_list_2_kp_list

        # check if we have all the kp list info
        if len(parsed_kp_info) == 6:       # if we only have opencv info
            # Compute a,b,c for vgg affine
            a = 1. / (parsed_kp_info[IDX_SIZE]**2)
            b = 0.
            c = 1. / (parsed_kp_info[IDX_SIZE]**2)
            parsed_kp_info = np.concatenate((parsed_kp_info, [a, b, c]))

        if len(parsed_kp_info) == 9:       # if we don't have the Affine warp
            parsed_kp_info = np.concatenate(
                (parsed_kp_info, np.zeros((4,))))
            parsed_kp_info = update_affine(parsed_kp_info)

        assert len(parsed_kp_info) == 13  # make sure we have everything!

        kp_list += [parsed_kp_info]

    # Close keypoint file
    kp_file.close()

    return kp_list

## update affine
def update_affine(kp):
    # Compute A0, A1, A2, A3Variable

    S = np.asarray([[kp[IDX_a], kp[IDX_b]],
                       [kp[IDX_b], kp[IDX_c]]])
    invS = np.linalg.inv(S)
    a = np.sqrt(invS[0, 0])
    b = invS[0, 1] / max(a, 1e-18)
    A = np.asarray([[a, 0],
                       [b, np.sqrt(max(invS[1, 1] - b**2, 0))]])

    # We need to rotate first!
    cos_val = np.cos(np.deg2rad(kp[IDX_ANGLE]))
    sin_val = np.sin(np.deg2rad(kp[IDX_ANGLE]))
    R = np.asarray([[cos_val, -sin_val],
                       [sin_val, cos_val]])

    A = np.dot(A, R)

    kp[IDX_A0] = A[0, 0]
    kp[IDX_A1] = A[0, 1]
    kp[IDX_A2] = A[1, 0]
    kp[IDX_A3] = A[1, 1]

    return kp

def getSinglePatchRawData(img, kp, random_rotation, offset, scale, skew):
    assert img.dtype == 'float32'
    scaleMultiplier = 4 
    param_nPatchSize = 32
    upright_kp = kp.copy()
    upright_kp[IDX_ANGLE] = random_rotation
    upright_kp = update_affine(upright_kp)

    # The Affine Matrix (with orientation)
    UprightA = np.asarray([[upright_kp[IDX_A0], upright_kp[IDX_A1]],
                           [upright_kp[IDX_A2], upright_kp[IDX_A3]]])
    # UprightA = np.asarray([[1,skew[0]],[skew[1],1]]).dot(UprightA)
    UprightA = UprightA.dot(np.asarray([[1,skew[0]],[skew[1],1]]))

    # Rescale the uprightA according to parameters (looking at larger region!)
    UprightA *= scaleMultiplier * scale

    # Add bias in the translation vector so that we get from -1 to 1
    t = np.asarray([[upright_kp[IDX_X] + offset[0], upright_kp[IDX_Y] + offset[1]]]).T + \
        np.dot(UprightA, np.asarray([[-1], [-1]]))

    # Transform in OpenCV representation

    # scaled uprightA so that we can use cv2.warpAffine (0~patchsize)
    M = np.concatenate(
        (UprightA / (float(param_nPatchSize) * 0.5), t),
        axis=1
    )

    # TODO:THIS PART MIGHT BE SLOW!!!
    patch_data = cv.warpAffine(
        img, M, (param_nPatchSize, param_nPatchSize),
        flags=cv.WARP_INVERSE_MAP + cv.INTER_CUBIC,
        borderMode=cv.BORDER_REFLECT101)

    return patch_data

if __name__ == '__main__':
	'''
	load model
	'''
	model_angle = Angle_Net()
	path = 'ckpt.pth.tar'
	ckpt = torch.load(path)
	model_angle.load_state_dict(ckpt['state_dict'])

	model_angle.cuda()
	cudnn.benchmark = True
	model_angle.eval()

	'''
	load image and keypoints
	'''
	img = cv.imread('image.png').astype(np.float32)
	img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	kpts = loadKpListFromTxt('keypoints.txt')

	'''
	main function
	'''
	for idx in range(len(kpts)):

		kpt = kpts[idx]

		angle = np.random.uniform(-np.pi, np.pi)

		'''
		sample two patches with randomly generated angle difference
		'''
		patch0 = getSinglePatchRawData(img, kpt, 0, [0, 0], 1, [0, 0])
		patch1 = getSinglePatchRawData(img, kpt, np.degrees(angle), [0, 0], 1, [0, 0])

		'''
		pre-processing
		'''
		mean = 128
		std = 128
		patch0_ts = torch.from_numpy(patch0)
		patch0_ts.sub_(float(mean)).div_(float(std))
		patch1_ts = torch.from_numpy(patch1)
		patch1_ts.sub_(float(mean)).div_(float(std))

		with torch.no_grad():
			patch0_var = Variable(patch0_ts.unsqueeze(0).unsqueeze(0).cuda(async = True))
			patch1_var = Variable(patch1_ts.unsqueeze(0).unsqueeze(0).cuda(async = True))

		[pred_angle0, pred_angle1] = model_angle([patch0_var, patch1_var])

		pred_angle0_deg = np.rad2deg(pred_angle0.item())
		pred_angle1_deg = np.rad2deg(pred_angle1.item())

		'''
		resample two new patches based on the estimated orientation, the new patches should be identical
		'''
		patch0_rect = getSinglePatchRawData(img, kpt, -pred_angle0_deg, [0, 0], 1, [0, 0])
		patch1_rect = getSinglePatchRawData(img, kpt, np.degrees(angle)-pred_angle1_deg, [0, 0], 1, [0, 0])

		'''
		2 x 2 plot:
		first row: two patches sampled with random angle difference, titles are estimated angle from network
		second row: two newly sampled patches after rectification
		'''
		plt.subplot(2,2,1)
		plt.imshow(patch0)
		plt.title('estimated angle0: %.2f deg'%(np.rad2deg(pred_angle0.item())))
		plt.subplot(2,2,2)
		plt.imshow(patch1)
		plt.title('estimated angle1: %.2f deg'%(np.rad2deg(pred_angle1.item())))
		plt.subplot(2,2,3)
		plt.imshow(patch0_rect)
		plt.title('rectified patch0')
		plt.subplot(2,2,4)
		plt.imshow(patch1_rect)
		plt.title('rectified patch1')
		plt.show()

	