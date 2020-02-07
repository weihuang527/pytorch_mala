pytorch_mala
============

Introduction:  
This code referenced from [funkey/mala](https://github.com/funkey/mala). Paper is [Large Scale Image Segmentation with Structured Loss Based Deep Learning for Connectome Reconstruction](https://ieeexplore.ieee.org/abstract/document/8364622/)  


0. Environment  
    docker image: renwu527/auto-emseg:v4.1  

1. Preprocessing  
    Convert segmentation label to boundary  
    See the *cremia.py* from line 106 to line 109  
    *GrowBoundary* is to make the boundary of the label thicker  
    *seg_to_affgraph* is to convert label to boundary  
    For the different training data, we must change the name of raw image and labels in the line 86 and 102 (in the *cremia.py*). For different data formats (png, 3d tiff, hdf and so on), we need to change the way of reading slightly.  

2. Training  
    ```python
    python main.py -dp='../data'
    ```
    *-dp* is the path of the training data, including raw image and labels.  

3. Inference  
    ```python
    python inference_mask_16bit_para_v2.py -in='../data/***.tif' -out='../affs' -ite=100000
    ```
    *-in* is the path of the data need to be predicted.  
    *-out* is the outpath of predicted affinity and mask.  
    *-ite* is the iteration times of trained model from the step 2.  

4. Over-segmentation and agglomeration  
    ```python
    python2 aff2id_mask_16bit_para.py -in='../affs' -out='../inference' -nu=500 -t=0.5
    ```
    *-nu* is the depth of inference data (the z dimension).  
    *-t* is the threshold used to agglomerate.  

5. Evaluate  
    ```python
    python evaluate.py -seg='../inference/segmentation.hdf' -gt='../data/gt.hdf'
    ```
    *-seg* is the output of the step 4.  
    *-gt* is the groundtruth of inference data.  

