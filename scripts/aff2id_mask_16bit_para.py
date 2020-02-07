import os
import h5py
import waterz
import numpy as np
import mahotas
import cremi
import argparse
from PIL import Image
from tqdm import tqdm
from scipy import ndimage
from scipy.ndimage import binary_erosion
from scipy.ndimage import binary_dilation
from time import time

import multiprocessing
from joblib import Parallel
from joblib import delayed


def get_seeds(boundary, method='grid', next_id=1, seed_distance=10):
	if method == 'grid':
		height = boundary.shape[0]
		width  = boundary.shape[1]
		seed_positions = np.ogrid[0:height:seed_distance, 0:width:seed_distance]
		num_seeds_y = seed_positions[0].size
		num_seeds_x = seed_positions[1].size
		num_seeds = num_seeds_x*num_seeds_y
		seeds = np.zeros_like(boundary).astype(np.int32)
		seeds[seed_positions] = np.arange(next_id, next_id + num_seeds).reshape((num_seeds_y,num_seeds_x))

	if method == 'minima':
		minima = mahotas.regmin(boundary)
		seeds, num_seeds = mahotas.label(minima)
		seeds += next_id
		seeds[seeds==next_id] = 0

	if method == 'maxima_distance':
		distance = mahotas.distance(boundary<0.5)
		maxima = mahotas.regmax(distance)
		seeds, num_seeds = mahotas.label(maxima)
		seeds += next_id
		seeds[seeds==next_id] = 0

	return seeds, num_seeds


def watershed(affs, seed_method, use_mahotas_watershed=True):
	affs_xy = 1.0 - 0.5 * (affs[1] + affs[2])
	depth  = affs_xy.shape[0]

	next_id = 1
	fragments = np.zeros_like(affs[0]).astype(np.uint64)
	for z in tqdm(range(depth)):
		seeds, num_seeds = get_seeds(affs_xy[z], next_id=next_id, method=seed_method)
		if use_mahotas_watershed:
			fragments[z] = mahotas.cwatershed(affs_xy[z], seeds)
		else:
			fragments[z] = ndimage.watershed_ift((255.0*affs_xy[z]).astype(np.uint8), seeds)
		next_id += num_seeds

	return fragments


def shave(im, border):
	if len(im.shape) == 4:
		return im[:, :, border[0] : -border[0], border[1] : -border[1]]
	elif len(im.shape) == 3:
		return im[:, border[0] : -border[0], border[1] : -border[1]]
	elif len(im.shape) == 2:
		return im[border[0] : -border[0], border[1] : -border[1]]
	else:
		raise NotImplementedError


parser = argparse.ArgumentParser()
parser.add_argument('-in', '--input-path', type=str, default='../affs')
parser.add_argument('-out', '--output-path', type=str, default='../inference')
parser.add_argument('-nu', '--num', type=int, default=500)
parser.add_argument('-t', '--thresd', type=float, default=0.5)
args = parser.parse_args()

aff_filename = args.input_path
output_filename = args.output_path
if not os.path.exists(output_filename):
	os.makedirs(output_filename)

depth = args.num
# load aff and mask
num_cores = multiprocessing.cpu_count()
parallel = Parallel(n_jobs=num_cores, backend='threading')

def _read_aff(i, j):
	return np.expand_dims(np.asarray(Image.open(os.path.join(aff_filename, '%d.%d.png' % (i, j)))).astype(np.float32) / 65535.0, axis=0)

def _read_mask(i):
	return np.expand_dims((np.asarray(Image.open(os.path.join(aff_filename, 'm%d.png' % i)))).astype(np.bool), axis=0)

affs = []
for i in range(0, 3):
	affs.append(np.expand_dims(np.concatenate(parallel(delayed(_read_aff)(i, j) for j in range(0, depth)), axis=0), axis=0))
affs = np.concatenate(affs, axis=0)
mask = np.concatenate(parallel(delayed(_read_mask)(i) for i in range(0, depth)), axis=0)

# dilate mask
# dilate_mask = 56
# #Image.fromarray(mask[20,:,:].astype(np.uint8)*255).save('temp.png')
# for i in range(len(mask)):
# 	mask[i] = binary_dilation(mask[i], iterations=dilate_mask, border_value=True)
#Image.fromarray(((~mask[20,:,:]).astype(np.uint8).astype(np.float32)*affs[1,20,:,:]*255).astype(np.uint8)).save('temp3.png')

# mask affs
for d in range(3):
	affs[d][mask] = 0

# watershed
fragments = watershed(affs, 'maxima_distance')

# mask fragments
fragments[mask] = 0

# agglomerate
threshold = [args.thresd]
sf = 'OneMinus<EdgeStatisticValue<RegionGraphType, MeanAffinityProvider<RegionGraphType, ScoreValue>>>'
#sf = 'OneMinus<QuantileAffinity<RegionGraphType, 85, ScoreValue>>'
segmentation = list(waterz.agglomerate(affs, threshold, None, fragments, scoring_function=sf, discretize_queue=256))

# save
out_file = os.path.join(output_filename, 'segmentation.hdf')
seg = segmentation[0]
out = h5py.File(out_file, 'w')
out.create_dataset('labels', data=seg, dtype=seg.dtype, compression='gzip')
out.close()
