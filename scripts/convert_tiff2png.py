from libtiff import TIFFfile
import os 
import numpy as np 
from PIL import Image

input_file = '../data/FIB_segmentaion/grayscale_maps_500.tif'
out_path = '../data/raw'
if not os.path.exists(out_path):
    os.makedirs(out_path)

tif = TIFFfile(input_file)
data, _ = tif.get_samples()
data = data[0]
print('the shape of data: ', data.shape)

for k in range(data.shape[0]):
    Image.fromarray(data[k]).save(os.path.join(out_path, 'raw_'+str(k).zfill(4)+'.png'))

print('Done')