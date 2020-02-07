from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import re
import argparse
import logging
import numpy as np
from time import time
from datetime import datetime
from PIL import Image
from libtiff import TIFF

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from unet3d import UNet3D
from cremia import CremiA_Provider
import pdb


parser = argparse.ArgumentParser()
# project settings
parser.add_argument('-np', '--num-process', type=int, default=4)
parser.add_argument('-df', '--display-freq', type=int, default=100)
parser.add_argument('-vf', '--valid-freq', type=int, default=100)
parser.add_argument('-sf', '--save-freq', type=int, default=1000)
parser.add_argument('-dp', '--data-path', type=str, default='../data')
parser.add_argument('-sp', '--save-path', type=str, default='../models')
parser.add_argument('-cp', '--cache-path', type=str, default='../caches')
parser.add_argument('-re', '--resume', action='store_true', default=False)
# training settings
parser.add_argument('-bs', '--batch-size', type=int, default=2)
parser.add_argument('-wd', '--weight-decay', type=float, default=None)
parser.add_argument('-ti', '--total-iters', type=int, default=400000)
parser.add_argument('-di', '--decay-iters', type=int, default=400000)
parser.add_argument('-wi', '--warmup-iters', type=int, default=0)
parser.add_argument('-bl', '--base-lr', type=float, default=0.0001)
parser.add_argument('-el', '--end-lr', type=float, default=0.00005)
parser.add_argument('-pw', '--power', type=float, default=1.5)
opt = parser.parse_args()


def init_project():
	def init_logging(path):
		logging.basicConfig(
			    level    = logging.INFO,
			    format   = '%(message)s',
			    datefmt  = '%m-%d %H:%M',
			    filename = path,
			    filemode = 'w')
	
		# define a Handler which writes INFO messages or higher to the sys.stderr
		console = logging.StreamHandler()
		console.setLevel(logging.INFO)
	
		# set a format which is simpler for console use
		formatter = logging.Formatter('%(message)s')
		# tell the handler to use this format
		console.setFormatter(formatter)
		logging.getLogger('').addHandler(console)
	
	if torch.cuda.is_available() is False:
		raise AttributeError('No GPU available')
	
	prefix = datetime.now().strftime('%Y.%m.%d-%H:%M:%S')
	if opt.resume is False:
		if not os.path.exists(opt.save_path):
			os.makedirs(opt.save_path)
		if not os.path.exists(opt.cache_path):
			os.makedirs(opt.cache_path)
	init_logging(os.path.join(opt.save_path, prefix + '.log'))
	logging.info(opt)


def load_dataset():
	print('Caching datasets ... ', end='', flush=True)
	t1 = time()
	train_provider = CremiA_Provider('train', opt.data_path, patch_size=[84, 268, 268],
	                                 batch_size=opt.batch_size, num_workers=opt.num_process)
	valid_provider = train_provider
	print('Done (time: %.2fs)' % (time() - t1))
	return train_provider, valid_provider


def build_model():
	print('Building model on ', end='', flush=True)
	t1 = time()
	device = torch.device('cuda:0')
	model = UNet3D().to(device)
	
	cuda_count = torch.cuda.device_count()
	if cuda_count > 1:
		if opt.batch_size % cuda_count == 0:
			print('%d GPUs ... ' % cuda_count, end='', flush=True)
			model = nn.DataParallel(model)
		else:
			raise AttributeError('Batch size (%d) cannot be equally divided by GPU number (%d)' % (opt.batch_size, cuda_count))
	else:
		print('a single GPU ... ', end='', flush=True)
	print('Done (time: %.2fs)' % (time() - t1))
	return model


def resume_params(model, optimizer, resume):
	if resume:
		t1 = time()
		last_iter = 0
		for files in os.listdir(opt.save_path):
			if 'model' in files:
				it = int(re.sub('\D', '', files))
				if it > last_iter:
					last_iter = it
		model_path = os.path.join(opt.save_path, 'model-%d.ckpt' % last_iter)
	
		print('Resuming weights from %s ... ' % model_path, end='', flush=True)
		if os.path.isfile(model_path):
			checkpoint = torch.load(model_path)
			model.load_state_dict(checkpoint['model_weights'])
			optimizer.load_state_dict(checkpoint['optimizer_weights'])
		else:
			raise AttributeError('No checkpoint found at %s' % model_path)
		print('Done (time: %.2fs)' % (time() - t1))
		print('valid %d, loss = %.4f' % (checkpoint['current_iter'], checkpoint['valid_result']))
		return model, optimizer, checkpoint['current_iter']
	else:
		return model, optimizer, 0


def loop(train_provider, valid_provider, model, criterion, optimizer, iters):
	def shave(im, border):
		if len(im.shape) == 3:
			return im[:, border[0] : -border[0], border[1] : -border[1]]
		elif len(im.shape) == 4:
			return im[:, :, border[0] : -border[0], border[1] : -border[1]]
		else: raise NotImplementedError
	
	def calculate_lr(iters):
		if iters < opt.warmup_iters:
			current_lr = (opt.base_lr - opt.end_lr) * pow(float(iters) / opt.warmup_iters, opt.power) + opt.end_lr
		else:
			if iters < opt.decay_iters:
				current_lr = (opt.base_lr - opt.end_lr) * pow(1 - float(iters - opt.warmup_iters) / opt.decay_iters, opt.power) + opt.end_lr
			else:
				current_lr = opt.end_lr
		return current_lr
	
	def write_to_tiff(name, im_arr):
		tif = TIFF.open(name, mode='w')
		for im in im_arr:
			im = Image.fromarray(im)
			tif.write_image(im, compression=None)
		tif.close()
	
	rcd_time = []
	sum_time = 0
	sum_loss = 0
	while iters <= opt.total_iters:
		# train
		iters += 1
		t1 = time()
		input, target = train_provider.next()
		
		# decay learning rate
		if opt.end_lr == opt.base_lr:
			current_lr = opt.base_lr
		else:
			current_lr = calculate_lr(iters)
			for param_group in optimizer.param_groups:
				param_group['lr'] = current_lr
		
		optimizer.zero_grad()
		pred = model(input)
		# pdb.set_trace()
		loss = criterion(pred, target)
		loss.backward()
		if opt.weight_decay is not None:
			for group in optimizer.param_groups:
				for param in group['params']:
					param.data = param.data.add(-opt.weight_decay * group['lr'], param.data)
		optimizer.step()
		
		sum_loss += loss.item()
		sum_time += time() - t1
		
		# log train
		if iters % opt.display_freq == 0:
			rcd_time.append(sum_time)
			logging.info('step %d, loss = %.4f (wt: *10, lr: %.8f, et: %.2f sec, rd: %.2f min)'
		                 % (iters, sum_loss / opt.display_freq * 10, current_lr, sum_time,
			               (opt.total_iters - iters) / opt.display_freq * np.mean(np.asarray(rcd_time)) / 60))
			sum_time = 0
			sum_loss = 0
		
		# valid
		if iters % opt.valid_freq == 0:
			input1 = (np.squeeze(input[0].data.cpu().numpy()) * 255).astype(np.uint8)
			input1 = input1[14:-14, 106:-106, 106:-106]
			input1 = input1[0]
			targets = (np.squeeze(target[0].data.cpu().numpy()) * 255).astype(np.uint8)
			targets = targets[0,0]
			pred = np.squeeze(pred[0].data.cpu().numpy())
			pred[pred>1] = 1; pred[pred<0] = 0
			preds = (pred * 255).astype(np.uint8)
			preds = preds[0,0]
			im_cat = np.concatenate([input1, preds, targets], axis=1)
			Image.fromarray(im_cat).save(os.path.join(opt.cache_path, 'iter-%d.png' % iters))
		
		# save
		if iters % opt.save_freq == 0:
			# states = {'current_iter': iters, 'valid_result': None,
		    #           'model_weights': model.state_dict(), 'optimizer_weights': optimizer.state_dict()}
			states = {'current_iter': iters, 'valid_result': None,
		              'model_weights': model.state_dict()}
			torch.save(states, os.path.join(opt.save_path, 'model-%d.ckpt' % iters))


if __name__ == '__main__':
	init_project()
	train_provider, _ = load_dataset()
	model = build_model()
	optimizer = optim.Adam(model.parameters(), lr=opt.base_lr, betas=(0.9, 0.999), eps=1e-8, amsgrad=False)
	model, optimizer, init_iters = resume_params(model, optimizer, opt.resume)
	
	loop(train_provider, None, model, F.mse_loss, optimizer, init_iters)
