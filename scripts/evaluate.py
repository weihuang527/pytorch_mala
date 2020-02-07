from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import h5py
import numpy as np
import argparse
from voi import voi
from rand import adapted_rand


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-seg', '--segmentation', type=str, default='../inference/segmentation.hdf')
	parser.add_argument('-gt', '--groundtruth', type=str, default='../data/gt.hdf')
	args = parser.parse_args()
	
	test = h5py.File(args.segmentation, 'r')
	test = np.asarray(test['labels']).astype(np.int64)
	
	gt = h5py.File(args.groundtruth, 'r')
	gt = np.asarray(gt['labels'])  # /volumes/labels/neuron_ids
	# no_gt = gt>=np.uint64(-10)
	# gt[no_gt] = 0
	gt = gt.astype(np.int64)
	test = test[14:-14,106:-106,106:-106]
	gt = gt[14:-14,106:-106,106:-106]
	
	(voi_split, voi_merge) = voi(test, gt)
	rand = adapted_rand(test, gt)
	print('voi split: ' + str(voi_split))
	print('voi merge: ' + str(voi_merge))
	print('rand: ' + str(rand))
