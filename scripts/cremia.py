from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import cv2
import torch
import h5py
import numpy as np
import random
import torchvision
import tifffile
from PIL import Image
import multiprocessing
from joblib import Parallel
from joblib import delayed
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from addaffinity import mknhood3d, seg_to_affgraph
from growboundary import GrowBoundary


class CremiA_Train(Dataset):
	def __init__(self, folder_name, patch_size, invalid_border=106, num_slice=500,
                 random_fliplr=True, random_flipud=True, random_flipz=True, random_rotation=True,
                 color_jitter=True, brightness=0.1, contrast=0.1, saturation=0.1,
                 elastic_trans=True, alpha_range=100, sigma=10, shave=20,
	             random_defect=False, prob_missing=0.03, prob_low_contrast=0.01, contrast_range=[0.1,0.3],
	             random_dropout=False, prob_dropout=0.25, dropout_ratio=0.05):
		super(CremiA_Train, self).__init__()
		# multiprocess settings
		num_cores = multiprocessing.cpu_count()
		self.parallel = Parallel(n_jobs=num_cores, backend='threading')
		self.use_mp = True
		
		# basic settings
		self.crop_size = patch_size
		self.invalid_border = invalid_border
		self.num_slice = num_slice
		
		# simple augmentations
		self.random_fliplr = random_fliplr
		self.random_flipud = random_flipud
		self.random_flipz = random_flipz
		self.random_rotation = random_rotation
		
		# color augmentations
		self.color_jitter = color_jitter
		self.brightness = brightness
		self.contrast = contrast
		self.saturation = saturation
		
		# elastic transform
		self.elastic_trans = elastic_trans
		self.alpha_range = alpha_range
		self.sigma = sigma
		self.shave = shave
		
		# defect augmentations
		self.random_defect = random_defect
		self.prob_missing = prob_missing
		self.prob_low_contrast = prob_low_contrast
		self.contrast_range = contrast_range
		
		# dropout content
		self.random_dropout = random_dropout
		self.prob_dropout = prob_dropout
		self.dropout_ratio = dropout_ratio
		
		# extend crop size
		self.crop_size[1] = self.crop_size[1] + 2 * self.shave if self.elastic_trans else self.crop_size[1]
		self.crop_size[2] = self.crop_size[2] + 2 * self.shave if self.elastic_trans else self.crop_size[2]
		
		# color jitter
		self.cj = torchvision.transforms.ColorJitter(self.brightness, self.contrast, self.saturation, hue=0)
		
		# load ims
		# self._raw_input = []
		# for i in np.arange(1, self.num_slice + 1, 1):
		# 	self._raw_input.append(np.expand_dims(np.array(Image.open(os.path.join(folder_name, '%d.png' % i))), axis=0))
		# # (152, 1462, 1462)
		# self._raw_input = np.concatenate(self._raw_input, axis=0)
		self._raw_input = tifffile.imread(os.path.join(folder_name, 'grayscale_maps_500.tif'))
		
		# load lbs
		# self._raw_label0 = []
		# for i in np.arange(1, self.num_slice + 1, 1):
		# 	self._raw_label0.append(np.expand_dims(np.array(Image.open(os.path.join(folder_name + '_affinities', '%d_0.png' % i))), axis=0))
		# self._raw_label0 = np.expand_dims(np.concatenate(self._raw_label0, axis=0), axis=0)
		
		# self._raw_label1 = []
		# for i in np.arange(1, self.num_slice + 1, 1):
		# 	self._raw_label1.append(np.expand_dims(np.array(Image.open(os.path.join(folder_name + '_affinities', '%d_1.png' % i))), axis=0))
		# self._raw_label1 = np.expand_dims(np.concatenate(self._raw_label1, axis=0), axis=0)
		
		# # (2, 152, 1462, 1462)
		# self._raw_labels = np.concatenate([self._raw_label0, self._raw_label1], axis=0)
		nhood = mknhood3d(1)
		with h5py.File(os.path.join(folder_name, 'groundtruth.h5'),'r') as f:
			label = f['stack'][:]
		
		growboundary = GrowBoundary(label, steps=2, only_xy=True)
		label = growboundary.process()
		self._raw_labels = seg_to_affgraph(label, nhood)
		self._raw_labels = (self._raw_labels * 255).astype(np.uint8)
	
	def __getitem__(self, index):
		# random crop
		i = random.randint(0, self._raw_input.shape[0] - self.crop_size[0])
		j = random.randint(0, self._raw_input.shape[1] - self.crop_size[1])
		k = random.randint(0, self._raw_input.shape[2] - self.crop_size[2])
		
		im = self._raw_input[i:i+self.crop_size[0], j:j+self.crop_size[1], k:k+self.crop_size[2]]
		lb = self._raw_labels[:, i:i+self.crop_size[0], j:j+self.crop_size[1], k:k+self.crop_size[2]]
		
		# combine (3, C, H, W)
		im = np.expand_dims(im, axis=0)
		im_lb = np.concatenate([im, lb], axis=0)
		
		# random flip
		if self.random_fliplr and random.uniform(0, 1) < 0.5:
			im_lb = self._fliplr(im_lb)
		if self.random_flipud and random.uniform(0, 1) < 0.5:
			im_lb = self._flipud(im_lb)
		if self.random_flipz and random.uniform(0, 1) < 0.5:
			im_lb = np.flip(im_lb, axis=1)
		
		# random rotation
		if self.random_rotation:
			r = random.randint(0, 3)
			if r: im_lb = self._rotate(im_lb, r)
		
		# split (1/2, C, H, W)
		im, lb = np.split(im_lb, [1])
		im = np.squeeze(im, axis=0)
		
		# random brightness, contrast and saturation
		if self.color_jitter:
			im = self._color_jitter(im)
		
		""" debug 
		im = im.copy(); lb = lb.copy()
		self._draw_grid(im[0, :, :], gray_level=255)
		self._draw_grid(lb[1, 0, :, :], gray_level=0) """
		
		# elastic transform
		if self.elastic_trans:
			im, lb = self._elastic_transform(im, lb)
		#Image.fromarray(0.9 * im[1, :, :] + 0.1 * lb[1, 1, :, :]).show()
		
		# random defect
		im = im.astype(np.float32) / 255.0
		if self.random_defect:
			for i in range(0, im.shape[0]):
				if random.uniform(0, 1) < self.prob_missing:
					im[i] = np.zeros([im.shape[1], im.shape[2]], dtype=np.uint8)
					break
			for i in range(0, im.shape[0]):
				if random.uniform(0, 1) < self.prob_low_contrast:
					mean = np.mean(im[i])
					im[i] -= mean
					im[i] *= np.random.uniform(self.contrast_range[0], self.contrast_range[1])
					im[i] += mean
					break
		
		# dropout
		lb = lb.astype(np.float32) / 255.0
		if self.random_dropout:
			kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
			for i in range(0, im.shape[0]):
				if random.uniform(0, 1) < self.prob_dropout:
					lb_ = cv2.erode(lb[0, i, :, :], kernel)					
					lb_mask = lb_==1
					dp_mask = np.random.uniform(0, 1, size=[im.shape[1], im.shape[2]])
					dp_mask[dp_mask < self.dropout_ratio] = 0
					dp_mask[dp_mask >= self.dropout_ratio] = 1
					mask = lb_mask.astype(np.uint8) * (1 - dp_mask)
					im[i] *= 1 - mask
		
		# shave border
		lb = self._shave(lb, [self.invalid_border, self.invalid_border])
		lb = lb[:, 14:-14, :, :]
		#Image.fromarray(0.9 * im[1, :, :] + 0.1 * np.pad(lb[1, 1, :, :], self.invalid_border, self._pad_with, padder=0)).show()
		
		im = np.expand_dims(im, axis=0)
		return im, lb
	
	def __len__(self):
		return int(sys.maxsize)
	
	@staticmethod
	def _shave(im, border):
		if len(im.shape) == 4:
			return im[:, :, border[0] : -border[0], border[1] : -border[1]]
		elif len(im.shape) == 3:
			return im[:, border[0] : -border[0], border[1] : -border[1]]
		elif len(im.shape) == 2:
			return im[border[0] : -border[0], border[1] : -border[1]]
		else:
			raise NotImplementedError
	
	@staticmethod
	def _pad_with(vector, pad_width, iaxis, kwargs):
		pad_value = kwargs.get('padder', 10)
		vector[:pad_width[0]] = pad_value
		vector[-pad_width[1]:] = pad_value
		return vector
	
	def _fliplr(self, im_lb):
		outputs = []
		for sub_vol in im_lb:
			results = []
			for input in sub_vol:
				results.append(np.expand_dims(np.fliplr(input), axis=0))
			outputs.append(np.expand_dims(np.concatenate(results, axis=0), axis=0))
		return np.concatenate(outputs, axis=0)
	
	def _flipud(self, im_lb):
		outputs = []
		for sub_vol in im_lb:
			results = []
			for input in sub_vol:
				results.append(np.expand_dims(np.flipud(input), axis=0))
			outputs.append(np.expand_dims(np.concatenate(results, axis=0), axis=0))
		return np.concatenate(outputs, axis=0)
	
	def _rotate(self, im_lb, r):
		outputs = []
		for sub_vol in im_lb:
			results = []
			for input in sub_vol:
				results.append(np.expand_dims(np.rot90(input, r), axis=0))
			outputs.append(np.expand_dims(np.concatenate(results, axis=0), axis=0))
		return np.concatenate(outputs, axis=0)
	
	def _color_jitter(self, im):
		results = []
		for input in im:
			results.append(np.expand_dims(np.asarray(self.cj(Image.fromarray(input))), axis=0))
		return np.concatenate(results, axis=0)
	
	@staticmethod
	def _draw_grid(im, grid_size=50, gray_level=255):
		for i in range(0, im.shape[1], grid_size):
			cv2.line(im, (i, 0), (i, im.shape[0]), color=(gray_level,))
		for j in range(0, im.shape[0], grid_size):
			cv2.line(im, (0, j), (im.shape[1], j), color=(gray_level,))
	
	@staticmethod
	def _map(input, indices, shape):
		return np.expand_dims(map_coordinates(input, indices, order=1).reshape(shape), axis=0)
	
	def _elastic_transform(self, image_in, label_in, random_state=None):
		"""Elastic deformation of image_ins as described in [Simard2003]_.
		.. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
		   Convolutional Neural Networks applied to Visual Document Analysis", in
		   Proc. of the International Conference on Document Analysis and
		   Recognition, 2003.
		"""
		alpha = np.random.uniform(0, self.alpha_range)
		
		if random_state is None:
			random_state = np.random.RandomState(None)
		
		shape = image_in.shape[1:]
		
		dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), self.sigma, mode='constant', cval=0) * alpha
		dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), self.sigma, mode='constant', cval=0) * alpha
		
		x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
		indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
		
		if self.use_mp:
			image_out = np.concatenate(self.parallel(delayed(self._map)(input, indices, shape) for input in image_in), axis=0)
		else:
			image_out = []
			for input in image_in:
				image_out.append(np.expand_dims(map_coordinates(input, indices, order=1).reshape(shape), axis=0))
			image_out = np.concatenate(image_out, axis=0)
		
		if self.use_mp:
			label_out = []
			for sub_vol in label_in:
				results = np.concatenate(self.parallel(delayed(self._map)(input, indices, shape) for input in sub_vol), axis=0)
				label_out.append(np.expand_dims(results, axis=0))
			label_out = np.concatenate(label_out, axis=0)
		else:
			label_out = []
			for sub_vol in label_in:
				results = []
				for input in sub_vol:
					results.append(np.expand_dims(map_coordinates(input, indices, order=1).reshape(shape), axis=0))
				label_out.append(np.expand_dims(np.concatenate(results, axis=0), axis=0))
			label_out = np.concatenate(label_out, axis=0)
		
		image_out = self._shave(image_out, [self.shave, self.shave])
		label_out = self._shave(label_out, [self.shave, self.shave])
		
		# format label
		label_out[label_out >= 128] = 255
		label_out[label_out < 128] = 0
		
		return image_out, label_out


class CremiA_Provider(object):
	def __init__(self, stage, folder_name, **kwargs):
			#patch_size, batch_size, num_workers, is_cuda=True):
		self.stage = stage
		if self.stage == 'train':
			self.cremia_data = CremiA_Train(folder_name, kwargs['patch_size'])
			self.batch_size = kwargs['batch_size']
			self.num_workers = kwargs['num_workers']
		elif self.stage == 'valid':
			self.cremia_data = CremiA_Valid(folder_name)
		else:
			raise AttributeError('Stage must be train/valid')
		self.is_cuda = True
		self.data_iter = None
		self.iteration = 0
		self.epoch = 1
	
	def __len__(self):
		return self.cremia_data.num_per_epoch
	
	def build(self):
		if self.stage == 'train':
			self.data_iter = iter(DataLoader(dataset=self.cremia_data, batch_size=self.batch_size, num_workers=self.num_workers,
                                             shuffle=False, drop_last=False, pin_memory=True))
		else:
			self.data_iter = iter(DataLoader(dataset=self.cremia_data, batch_size=1, num_workers=0,
                                             shuffle=False, drop_last=False, pin_memory=True))
	
	def next(self):
		if self.data_iter is None:
			self.build()
		try:
			batch = self.data_iter.next()
			self.iteration += 1
			if self.is_cuda:
				batch[0] = batch[0].cuda()
				batch[1] = batch[1].cuda()
			return batch[0], batch[1]
		except StopIteration:
			self.epoch += 1
			self.build()
			self.iteration += 1
			batch = self.data_iter.next()
			if self.is_cuda:
				batch[0] = batch[0].cuda()
				batch[1] = batch[1].cuda()
			return batch[0], batch[1]


if __name__ == '__main__':
	""""""
	from time import time
	cremia_train = CremiA_Train('/dev/shm/sample_A_padded_cropped', patch_size=[84, 268, 268])
	t = time()
	for _ in range(0, 100):
		im, lb = iter(cremia_train).__next__()
	print(time() - t)
	# 66.48s
	
	
	"""
	from time import time
	provider = CremiA_Provider('train', '/dev/shm/sample_A_padded_cropped', patch_size=[84, 268, 268], batch_size=1, num_workers=0)
	t1 = time()
	for _ in range(0, 100):
		batch = provider.next()
		#print('break')
	
	print(time() - t1)
	print(batch[0].shape, batch[1].shape)
	print(provider.epoch, provider.iteration)
	
	print('break')
	# worker0: 68.61
	# worker1: 70.15
"""