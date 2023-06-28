import os
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math
from torch.nn import functional as F 
import cv2
from sklearn.metrics import roc_curve, auc
import nibabel as nib
import scipy.io as sio 





img = nib.load('saliency_map.nii.gz').get_fdata()


# ori_data = img * 10
# nii_image_mri = nib.Nifti1Image(ori_data, np.eye(4))
# nib.save(nii_image_mri, 'saliency_map.nii.gz')


max_val = img.max()
min_val = img.min()
output_ = (img - min_val) / (max_val - min_val)   

 
np.save('saliency_map.npy', output_)




  