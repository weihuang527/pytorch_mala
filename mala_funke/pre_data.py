import tifffile
import os
import numpy as np 
import h5py

raw_path = '../data/FIB_segmentaion/grayscale_maps_500.tif'
label_path = '../data/FIB_segmentaion/groundtruth.h5'
output_file = '../data/fib.hdf'

raw = tifffile.imread(raw_path)
with h5py.File(label_path,'r') as f:
    label = f['stack'][:]
label = label[:500, :500, :500]

dataset_raw = 'volumes/raw'
dataset_label = 'volumes/labels/neuron_ids'
f = h5py.File(output_file, 'w')
f.create_dataset(dataset_raw, data=raw, dtype=raw.dtype, compression="gzip")
f.create_dataset(dataset_label, data=label, dtype=label.dtype, compression="gzip")
# 添加属性
f[dataset_raw].attrs['resolution'] = np.array([8,8,8])
f.close()