from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import glob
import h5py
import torch
import argparse
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from time import time
from PIL import Image
# from libtiff import TIFF
import tifffile
# from unet3d_mala import UNet3D_MALA
from unet3d import UNet3D
from collections import OrderedDict

import multiprocessing
from joblib import Parallel
from joblib import delayed


def aug(im, i):
	if i == 0:
		return im
	else:
		im = np.squeeze(im.cpu().numpy())
		if i == 1: im = np.rot90(im)
		elif i == 2: im = np.rot90(im, 2)
		elif i == 3: im = np.rot90(im, 3)
		elif i == 4: im = np.fliplr(im)
		elif i == 5: im = np.flipud(im)
		elif i == 6: im = np.fliplr(np.rot90(im))
		elif i == 7: im = np.flipud(np.rot90(im))
		else: raise NotImplementedError
		im = np.expand_dims(np.expand_dims(im, axis=0), axis=0)
		im = torch.from_numpy(im.copy()).to(device)
		return im


def inv_aug(pred, i):
	if i == 0:
		return pred
	else:
		pred = np.squeeze(pred.cpu().numpy())
		pred = np.transpose(pred, [1, 2, 0])
		if i == 1: pred = np.rot90(pred, 3)
		elif i == 2: pred = np.rot90(pred, 2)
		elif i == 3: pred = np.rot90(pred)
		elif i == 4: pred = np.fliplr(pred)
		elif i == 5: pred = np.flipud(pred)
		elif i == 6: pred = np.rot90(np.fliplr(pred), 3)
		elif i == 7: pred = np.rot90(np.flipud(pred), 3)
		else: raise NotImplementedError
		pred = np.transpose(pred, [2, 0, 1])
		pred = np.expand_dims(pred, axis=0)
		pred = torch.from_numpy(pred.copy()).to(device)
		return pred


def forward(model, input, aug):
	if aug:
		raise NotImplementedError
	else:
		return model(input)


def grow_py(img, seed, t):
	"""
	img: ndarray, ndim=3
	    An image volume.

	seed: tuple, len=3
	    Region growing starts from this point.

	t: int
	    The image neighborhood radius for the inclusion criteria.
	"""
	seg = np.zeros(img.shape, dtype=np.bool)
	checked = np.zeros_like(seg)

	seg[seed] = True
	checked[seed] = True
	needs_check = get_nbhd(seed, checked, img.shape)

	while len(needs_check) > 0:
		pt = needs_check.pop()

		# Its possible that the point was already checked and was
		# put in the needs_check stack multiple times.
		if checked[pt]: continue

		checked[pt] = True

		# Handle borders.
		imin = max(pt[0]-t, 0)
		imax = min(pt[0]+t, img.shape[0]-1)
		jmin = max(pt[1]-t, 0)
		jmax = min(pt[1]+t, img.shape[1]-1)
		kmin = max(pt[2]-t, 0)
		kmax = min(pt[2]+t, img.shape[2]-1)

		if img[pt] >= img[imin:imax+1, jmin:jmax+1, kmin:kmax+1].mean():
			# Include the voxel in the segmentation and
			# add its neighbors to be checked.
			seg[pt] = True
			needs_check += get_nbhd(pt, checked, img.shape)

	return seg


def get_nbhd(pt, checked, dims):
	nbhd = []

	if (pt[0] > 0) and not checked[pt[0]-1, pt[1], pt[2]]:
		nbhd.append((pt[0]-1, pt[1], pt[2]))
	if (pt[1] > 0) and not checked[pt[0], pt[1]-1, pt[2]]:
		nbhd.append((pt[0], pt[1]-1, pt[2]))
	if (pt[2] > 0) and not checked[pt[0], pt[1], pt[2]-1]:
		nbhd.append((pt[0], pt[1], pt[2]-1))

	if (pt[0] < dims[0]-1) and not checked[pt[0]+1, pt[1], pt[2]]:
		nbhd.append((pt[0]+1, pt[1], pt[2]))
	if (pt[1] < dims[1]-1) and not checked[pt[0], pt[1]+1, pt[2]]:
		nbhd.append((pt[0], pt[1]+1, pt[2]))
	if (pt[2] < dims[2]-1) and not checked[pt[0], pt[1], pt[2]+1]:
		nbhd.append((pt[0], pt[1], pt[2]+1))

	return nbhd


parser = argparse.ArgumentParser()
parser.add_argument('-in', '--input-path', type=str, default=None)
parser.add_argument('-out', '--output-path', type=str, default='../affs')
parser.add_argument('-ite', '--iterations', type=int, default=100000)
args = parser.parse_args()

# load input
# hdf_name = args.input_path
# hdf = h5py.File(os.path.join('./', hdf_name))
# raw = np.asarray(hdf['/volumes/raw'])
input_path = args.input_path
model_path = '../models'
raw = np.asarray(tifffile.imread(input_path))

# restore model
# model = UNet3D_MALA()
model = UNet3D()
ckpt = 'model-%d.ckpt' % args.iterations
ckpt_path = os.path.join(model_path, ckpt)
checkpoint = torch.load(ckpt_path)

new_state_dict = OrderedDict()
state_dict = checkpoint['model_weights']
for k, v in state_dict.items():
	# name = k[7:] # remove module.
	name = k
	new_state_dict[name] = v
model.load_state_dict(new_state_dict)
if torch.cuda.is_available() == False:
	raise AttributeError('GPU is not available')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# initialize output
stride = 56
in_shape = [84, 268, 268]
out_shape = [56, 56, 56]
output = np.zeros([1, 3, raw.shape[0] - (in_shape[0] - out_shape[0]),
                   raw.shape[1] - (in_shape[1] - out_shape[1]),
                   raw.shape[2] - (in_shape[2] - out_shape[2])], np.float32)
output = torch.Tensor(output)

# generate mask
raw_ = sitk.GetImageFromArray(raw)
mask = sitk.ConnectedThreshold(raw_, seedList=[(0,0,0)], lower=0, upper=0.00001)
mask = sitk.GetArrayFromImage(mask)

mask = mask[14:-14, 106:-106, 106:-106]
mask = mask.astype(np.uint8)
mask = torch.Tensor(mask)
with torch.no_grad():
	mask = torch.nn.functional.pad(mask, ((in_shape[2] - out_shape[2]) // 2, (in_shape[2] - out_shape[2]) // 2,
	                                      (in_shape[1] - out_shape[1]) // 2, (in_shape[1] - out_shape[1]) // 2,
	                                      (in_shape[0] - out_shape[0]) // 2, (in_shape[0] - out_shape[0]) // 2), value=1)	
mask = mask.data.numpy()
mask = np.squeeze(mask)
#Image.fromarray(mask[50]*255).show()
mask = mask.astype(np.bool)

# inference loop
with torch.no_grad():
	part = 0
	zz = list(np.arange(0, raw.shape[0] - in_shape[0], stride)) + [raw.shape[0] - in_shape[0]]
	for z in zz:
		part += 1
		print('Part %d / %d' % (part, len(zz)))
		for y in tqdm(list(np.arange(0, raw.shape[1] - in_shape[1], stride)) + [raw.shape[1] - in_shape[1]]):
			for x in list(np.arange(0, raw.shape[2] - in_shape[2], stride)) + [raw.shape[2] - in_shape[2]]:
				input = raw[z : z + in_shape[0], y : y + in_shape[1], x : x + in_shape[2]]
				input = input.astype(np.float32) / 255.0
				input = np.expand_dims(np.expand_dims(input, axis=0), axis=0)
				input = torch.Tensor(input).to(device)
				pred = forward(model, input, aug=False)
				output[:, :, z : z + out_shape[0], y : y + out_shape[1], x : x + out_shape[2]] = pred.data.cpu()
	
	output = torch.nn.functional.pad(output, ((in_shape[2] - out_shape[2]) // 2, (in_shape[2] - out_shape[2]) // 2,
                                              (in_shape[1] - out_shape[1]) // 2, (in_shape[1] - out_shape[1]) // 2,
                                              (in_shape[0] - out_shape[0]) // 2, (in_shape[0] - out_shape[0]) // 2))

# format
output = output.data.numpy()
output = np.squeeze(output, axis=0)
output[output>1] = 1
output[output<0] = 0

def _write_aff(aff, i, j):
	Image.fromarray((aff*65535).astype(np.uint32)).save(os.path.join(args.output_path, '%d.%d.png' % (i, j)))

def _write_mask(mask, i):
	Image.fromarray(mask.astype(np.uint8)*255).save(os.path.join(args.output_path, 'm%d.png' % i))

num_cores = multiprocessing.cpu_count()
parallel = Parallel(n_jobs=num_cores, backend='threading')

# save to pngs
os.mkdir(args.output_path)
for i in range(0, 3):
	parallel(delayed(_write_aff)(output[i, j, :, :], i, j) for j in range(0, output.shape[1]))	
parallel(delayed(_write_mask)(mask[i, :, :], i) for i in range(0, mask.shape[0]))
