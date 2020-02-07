#!/bin/bash

for loop in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
    echo "thresd=${loop}"
    python2 aff2id_mask_16bit_para.py -t=$loop
	python evaluate.py
done