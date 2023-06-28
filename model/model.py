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
from mri_pet_dataset import MRIandPETdataset
from mri_pet_dataset_test import TestDataset
from densenet import *
from SSIM_loss import SSIM 
import nibabel as nib
import scipy 
from densenet_mri import *
from densenet_pet import *

from densenet import *
import scipy.io as sio 

class Discriminator(nn.Module):

    def __init__(self, opt, conv_dim=16):
        super(Discriminator, self).__init__()
        self.conv_dim = conv_dim
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.down_sampling = nn.MaxPool3d(kernel_size=3, stride=2)

        self.conv1 = nn.Conv3d(1, conv_dim, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn1 =nn.BatchNorm3d(conv_dim)
        self.conv2 = nn.Conv3d(conv_dim, conv_dim*2, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn2 = nn.BatchNorm3d(conv_dim*2)
        self.conv3 = nn.Conv3d(conv_dim*2, conv_dim*4, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn3 = nn.BatchNorm3d(conv_dim*4)
        self.conv4 = nn.Conv3d(conv_dim*4, conv_dim*8, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn4 = nn.BatchNorm3d(conv_dim*8) 
        self.conv5 = nn.Conv3d(conv_dim*8, conv_dim*16, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn5 = nn.BatchNorm3d(conv_dim*16) 
        self.conv6 = nn.Conv3d(conv_dim*16, conv_dim*32, kernel_size=3, stride=1, padding=0, bias=True)
        self.bn6 = nn.BatchNorm3d(conv_dim*32) 

        self.fc = nn.Linear(conv_dim*32*1, 1)

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(self.relu(self.bn1(h)))
        h = self.conv3(self.relu(self.bn2(h)))
        h = self.conv4(self.relu(self.bn3(h)))
        h = self.conv5(self.relu(self.bn4(h)))
        h = self.conv6(self.relu(self.bn5(h)))
        if h.shape[0] == 1:
            h = self.relu(h)
        else:
            h = self.relu(self.bn6(h))
        h = h.view(h.size()[0], -1)
        h = self.fc(h)
        h = self.sigmoid(h)
        
        return h


class Generator_local(nn.Module):
    """Generator Unet structure"""

    def __init__(self, opt, conv_dim=8):
        super(Generator_local, self).__init__()
        self.conv_dim = conv_dim
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.down_sampling = nn.MaxPool3d(kernel_size=3, stride=2)
        self.tanh = nn.Tanh()

        #Encoder
        self.tp_conv1 = nn.Conv3d(1, conv_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm3d(conv_dim)
        self.tp_conv2 = nn.Conv3d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm3d(conv_dim)

        self.tp_conv3 = nn.Conv3d(conv_dim, conv_dim*2, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm3d(conv_dim*2)
        self.tp_conv4 = nn.Conv3d(conv_dim*2, conv_dim*2, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4 = nn.BatchNorm3d(conv_dim*2)  

        self.tp_conv5 = nn.Conv3d(conv_dim*2, conv_dim*4, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn5 = nn.BatchNorm3d(conv_dim*4)
        self.tp_conv6 = nn.Conv3d(conv_dim*4, conv_dim*4, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn6 = nn.BatchNorm3d(conv_dim*4)  

        self.tp_conv7 = nn.Conv3d(conv_dim*4, conv_dim*8, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn7 = nn.BatchNorm3d(conv_dim*8)
        self.tp_conv8 = nn.Conv3d(conv_dim*8, conv_dim*8, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn8 = nn.BatchNorm3d(conv_dim*8)  

        self.rbn = nn.Conv3d(conv_dim*8, conv_dim*8, kernel_size=3, stride=1, padding=1, bias=True)

        #Decoder
        self.tp_conv9 = nn.ConvTranspose3d(conv_dim*8, conv_dim*4, kernel_size=3, stride=2, padding=0, output_padding=1, bias=True)
        self.bn9 = nn.BatchNorm3d(conv_dim*4)
        self.tp_conv10 = nn.Conv3d(conv_dim*8, conv_dim*4, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn10 = nn.BatchNorm3d(conv_dim*4)
        self.tp_conv11 = nn.Conv3d(conv_dim*4, conv_dim*4, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn11 = nn.BatchNorm3d(conv_dim*4)

        self.tp_conv12 = nn.ConvTranspose3d(conv_dim*4, conv_dim*2, kernel_size=3, stride=2, padding=(0, 0, 0), output_padding=(0, 1, 0), bias=True)
        self.bn12 = nn.BatchNorm3d(conv_dim*2)
        self.tp_conv13 = nn.Conv3d(conv_dim*4, conv_dim*2, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn13 = nn.BatchNorm3d(conv_dim*2)
        self.tp_conv14 = nn.Conv3d(conv_dim*2, conv_dim*2, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn14 = nn.BatchNorm3d(conv_dim*2)

        self.tp_conv15 = nn.ConvTranspose3d(conv_dim*2, conv_dim*1, kernel_size=3, stride=2, padding=(0, 0, 0), output_padding=(1, 1, 1), bias=True)
        self.bn15 = nn.BatchNorm3d(conv_dim*1)
        self.tp_conv16 = nn.Conv3d(conv_dim*2, conv_dim*1, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn16 = nn.BatchNorm3d(conv_dim*1)
        self.tp_conv17 = nn.Conv3d(conv_dim*1, conv_dim*1, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn17 = nn.BatchNorm3d(conv_dim*1)

        self.conv_an_0 = nn.Conv3d(conv_dim*1, conv_dim*1, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_an_1 = nn.Conv3d(conv_dim*1, conv_dim*1, kernel_size=1, stride=1, padding=0, bias=True)
        self.convback = nn.Conv3d(conv_dim*1, conv_dim*1, kernel_size=3, stride=1, padding=1, bias=True)

        self.tp_conv18 = nn.Conv3d(conv_dim*1, 1, kernel_size=3, stride=1, padding=1, bias=True)
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
   

        h = self.tp_conv1(z)
        h = self.bn1(h)
        h = self.tp_conv2(self.relu(h))
        h = self.bn2(h)
        skip3 = h
        h = self.down_sampling(self.relu(h))

        h = self.tp_conv3(h)
        h = self.bn3(h)
        h = self.tp_conv4(self.relu(h))
        h = self.bn4(h)
        skip2 = h
        h = self.down_sampling(self.relu(h))

        h = self.tp_conv5(h)
        h = self.bn5(h)
        h = self.tp_conv6(self.relu(h))
        h = self.bn6(h)
        skip1 = h
        h = self.down_sampling(self.relu(h))        

        h = self.tp_conv7(h)
        h = self.bn7(h)
        h = self.tp_conv8(self.relu(h))
        h = self.bn8(h)
        c1 = h       

        #RNB
        h = self.rbn(self.relu(c1))
        h = self.bn8(h)
        h = self.rbn(self.relu(h))
        h = self.bn8(h)
        c2 = h + c1

        h = self.rbn(self.relu(c2))
        h = self.bn8(h)
        h = self.rbn(self.relu(h))
        h = self.bn8(h)
        c3 = h + c2

        h = self.rbn(self.relu(c3))
        h = self.bn8(h)
        h = self.rbn(self.relu(h))
        h = self.bn8(h)
        c4 = h + c3        

        h = self.rbn(self.relu(c4))
        h = self.bn8(h)
        h = self.rbn(self.relu(h))
        h = self.bn8(h)
        c5 = h + c4

        h = self.rbn(self.relu(c5))
        h = self.bn8(h)
        h = self.rbn(self.relu(h))
        h = self.bn8(h)
        c6 = h + c5

        h = self.rbn(self.relu(c6))
        h = self.bn8(h)
        h = self.rbn(self.relu(h))
        h = self.bn8(h)
        c7 = h + c6
        #RBN

        h = self.tp_conv9(self.relu(c7))
        h = self.bn9(h)
        h = torch.cat([h, skip1], 1)
        h = self.tp_conv10(self.relu(h))
        h = self.bn10(h)
        h = self.tp_conv11(self.relu(h))
        h = self.bn11(h)

        h = self.tp_conv12(self.relu(h))
        h = self.bn12(h)
        h = torch.cat([h, skip2], 1)
        h = self.tp_conv13(self.relu(h))
        h = self.bn13(h)
        h = self.tp_conv14(self.relu(h))
        h = self.bn14(h)
        

        h = self.tp_conv15(self.relu(h))
        h = self.bn15(h)
        h = torch.cat([h, skip3], 1)
        h_end = self.tp_conv16(self.relu(h))
        h_end = self.bn16(h_end)
        h = self.conv_an_0(self.relu(h_end))
        h = self.conv_an_1(h)
        h_sigmoid = self.sigmoid(h)

        h = self.tp_conv17(self.relu(h_end))
        h = h * h_sigmoid 
        h = self.bn17(h)

        h = self.tp_conv18(self.relu(h))
        h = self.sigmoid(h)

        return h

class Generator_fusion(nn.Module):
    """Generator Unet structure"""

    def __init__(self, opt, conv_dim=16):
        super(Generator_fusion, self).__init__()
        self.conv_dim = conv_dim
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


        self.tp_conv1 = nn.Conv3d(2, conv_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm3d(conv_dim)

        self.tp_conv2 = nn.Conv3d(conv_dim, conv_dim*2, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm3d(conv_dim*2)

        self.tp_conv3 = nn.Conv3d(conv_dim*2, conv_dim*4, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm3d(conv_dim*4)

        self.tp_conv4 = nn.Conv3d(conv_dim*4, conv_dim*2, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4 = nn.BatchNorm3d(conv_dim*2) 

        self.tp_conv5 = nn.Conv3d(conv_dim*2, conv_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn5 = nn.BatchNorm3d(conv_dim)

        self.tp_conv6 = nn.Conv3d(conv_dim, 1, kernel_size=3, stride=1, padding=1, bias=True)




        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, y):
   
        h = torch.cat([x, y], 1)

        h = self.tp_conv1(h)
        h = self.bn1(h)

        h = self.tp_conv2(self.relu(h))
        h = self.bn2(h)

        h = self.tp_conv3(h)
        h = self.bn3(h)

        h = self.tp_conv4(self.relu(h))
        h = self.bn4(h)

        h = self.tp_conv5(h)
        h = self.bn5(h)

        h = self.tp_conv6(self.relu(h))
        h = self.sigmoid(h)

        return h

class Generator(nn.Module):
    """Generator Unet structure"""

    def __init__(self, opt, conv_dim=8):
        super(Generator, self).__init__()
        self.conv_dim = conv_dim
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.down_sampling = nn.MaxPool3d(kernel_size=3, stride=2)
        self.tanh = nn.Tanh()

        #Encoder
        self.tp_conv1 = nn.Conv3d(1, conv_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm3d(conv_dim)
        self.tp_conv2 = nn.Conv3d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm3d(conv_dim)

        self.tp_conv3 = nn.Conv3d(conv_dim, conv_dim*2, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm3d(conv_dim*2)
        self.tp_conv4 = nn.Conv3d(conv_dim*2, conv_dim*2, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4 = nn.BatchNorm3d(conv_dim*2)  

        self.tp_conv5 = nn.Conv3d(conv_dim*2, conv_dim*4, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn5 = nn.BatchNorm3d(conv_dim*4)
        self.tp_conv6 = nn.Conv3d(conv_dim*4, conv_dim*4, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn6 = nn.BatchNorm3d(conv_dim*4)  

        self.tp_conv7 = nn.Conv3d(conv_dim*4, conv_dim*8, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn7 = nn.BatchNorm3d(conv_dim*8)
        self.tp_conv8 = nn.Conv3d(conv_dim*8, conv_dim*8, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn8 = nn.BatchNorm3d(conv_dim*8)  

        self.rbn = nn.Conv3d(conv_dim*8, conv_dim*8, kernel_size=3, stride=1, padding=1, bias=True)

        #Decoder
        self.tp_conv9 = nn.ConvTranspose3d(conv_dim*16, conv_dim*8, kernel_size=3, stride=2, padding=0, output_padding=1, bias=True)
        self.bn9 = nn.BatchNorm3d(conv_dim*8)
        self.tp_conv10 = nn.Conv3d(conv_dim*12, conv_dim*6, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn10 = nn.BatchNorm3d(conv_dim*6)
        self.tp_conv11 = nn.Conv3d(conv_dim*6, conv_dim*6, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn11 = nn.BatchNorm3d(conv_dim*6)

        self.tp_conv12 = nn.ConvTranspose3d(conv_dim*6, conv_dim*3, kernel_size=3, stride=2, padding=(0, 0, 0), output_padding=(0, 1, 0), bias=True)
        self.bn12 = nn.BatchNorm3d(conv_dim*3)
        self.tp_conv13 = nn.Conv3d(conv_dim*5, conv_dim*2, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn13 = nn.BatchNorm3d(conv_dim*2)
        self.tp_conv14 = nn.Conv3d(conv_dim*2, conv_dim*2, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn14 = nn.BatchNorm3d(conv_dim*2)

        self.tp_conv15 = nn.ConvTranspose3d(conv_dim*2, conv_dim*1, kernel_size=3, stride=2, padding=(0, 0, 0), output_padding=(1, 1, 1), bias=True)
        self.bn15 = nn.BatchNorm3d(conv_dim*1)
        self.tp_conv16 = nn.Conv3d(conv_dim*2, conv_dim*1, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn16 = nn.BatchNorm3d(conv_dim*1)
        self.tp_conv17 = nn.Conv3d(conv_dim*1, conv_dim*1, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn17 = nn.BatchNorm3d(conv_dim*1)

        self.conv_an_0 = nn.Conv3d(conv_dim*1, conv_dim*1, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_an_1 = nn.Conv3d(conv_dim*1, conv_dim*1, kernel_size=1, stride=1, padding=0, bias=True)
        # self.convback = nn.Conv3d(conv_dim*1, conv_dim*1, kernel_size=3, stride=1, padding=1, bias=True)

        self.tp_conv18 = nn.Conv3d(conv_dim*1, 1, kernel_size=3, stride=1, padding=1, bias=True)

        self.tp_conv_mapping = nn.ConvTranspose3d(62, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn_map = nn.BatchNorm3d(64)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, z, map_z):
   

        h = self.tp_conv1(z)
        h = self.bn1(h)
        h = self.tp_conv2(self.relu(h))
        h = self.bn2(h)
        skip3 = h
        h = self.down_sampling(self.relu(h))

        h = self.tp_conv3(h)
        h = self.bn3(h)
        h = self.tp_conv4(self.relu(h))
        h = self.bn4(h)
        skip2 = h
        h = self.down_sampling(self.relu(h))

        h = self.tp_conv5(h)
        h = self.bn5(h)
        h = self.tp_conv6(self.relu(h))
        h = self.bn6(h)
        skip1 = h
        h = self.down_sampling(self.relu(h))        

        h = self.tp_conv7(h)
        h = self.bn7(h)
        h = self.tp_conv8(self.relu(h))
        h = self.bn8(h)
        c1 = h       

        #RNB
        h = self.rbn(self.relu(c1))
        h = self.bn8(h)
        h = self.rbn(self.relu(h))
        h = self.bn8(h)
        c2 = h + c1

        h = self.rbn(self.relu(c2))
        h = self.bn8(h)
        h = self.rbn(self.relu(h))
        h = self.bn8(h)
        c3 = h + c2

        h = self.rbn(self.relu(c3))
        h = self.bn8(h)
        h = self.rbn(self.relu(h))
        h = self.bn8(h)
        c4 = h + c3        

        h = self.rbn(self.relu(c4))
        h = self.bn8(h)
        h = self.rbn(self.relu(h))
        h = self.bn8(h)
        c5 = h + c4

        h = self.rbn(self.relu(c5))
        h = self.bn8(h)
        h = self.rbn(self.relu(h))
        h = self.bn8(h)
        c6 = h + c5

        h = self.rbn(self.relu(c6))
        h = self.bn8(h)
        h = self.rbn(self.relu(h))
        h = self.bn8(h)
        c7 = h + c6
        #RBN
        
        maped = self.tp_conv_mapping(map_z)
        maped = self.bn_map(maped)
        new_cat = torch.cat([c7, maped], 1)

        h = self.tp_conv9(self.relu(new_cat))
        h = self.bn9(h)
        h = torch.cat([h, skip1], 1)
        h = self.tp_conv10(self.relu(h))
        h = self.bn10(h)
        h = self.tp_conv11(self.relu(h))
        h = self.bn11(h)

        h = self.tp_conv12(self.relu(h))
        h = self.bn12(h)
        h = torch.cat([h, skip2], 1)
        h = self.tp_conv13(self.relu(h))
        h = self.bn13(h)
        h = self.tp_conv14(self.relu(h))
        h = self.bn14(h)
        

        h = self.tp_conv15(self.relu(h))
        h = self.bn15(h)
        h = torch.cat([h, skip3], 1)
        h_end = self.tp_conv16(self.relu(h))
        h_end = self.bn16(h_end)
        h = self.conv_an_0(self.relu(h_end))
        h = self.conv_an_1(h)
        h_sigmoid = self.sigmoid(h)

        h = self.tp_conv17(self.relu(h_end))
        h = h * h_sigmoid 
        h = self.bn17(h)

        h = self.tp_conv18(self.relu(h))
        h = self.sigmoid(h)

        return h

class CoopNets(nn.Module):
    def __init__(self, opts):
        super(CoopNets, self).__init__()
        self.opts = opts


    def train(self):
        D = Discriminator(self.opts).cuda()
        G = Generator(self.opts).cuda()

        #classifer
        # T = densenet21().cuda() 
        T = densenet21_pet().cuda() 

        # AD vs NC
        # PET
        # T.load_state_dict(torch.load('./classification_models/Densenet_01/mri_to_pet_mapping/PET/PET79_TLoss0.0036_TrainACC1.0_TestACC0.8967_TestSEN0.8889_TestSPE0.9027_TestAUC0.9421_F1S0.8827_T2.pth'))#'''!!!!!!!!!!!!!!!!!!genPET1to2!!!!!!!!!!!!!!!!!!!!!!!!'''
        
        # AD vs NC        
        # T2
        # T.load_state_dict(torch.load('./classification_models/Densenet_01/mri_to_pet_mapping/T2/PET63_TLoss0.0_TrainACC0.9966_TestACC0.8675_TestSEN0.8175_TestSPE0.9187_TestAUC0.9088_F1S0.8619_T2.pth'))#'''!!!!!!!!!!!!!!!!!!genPET1to2!!!!!!!!!!!!!!!!!!!!!!!!'''
           
        # pMCI vs sMCI
        # PET
        T.load_state_dict(torch.load('./classification_models/Densenet_01/mri_to_pet_mapping/PET/PET79_TLoss0.0036_TrainACC1.0_TestACC0.8967_TestSEN0.8889_TestSPE0.9027_TestAUC0.9421_F1S0.8827_T2.pth'))#'''!!!!!!!!!!!!!!!!!!genPET1to2!!!!!!!!!!!!!!!!!!!!!!!!'''
        


        T_Mapping = densenet21_mri().cuda()
        # AD vs NC
        # PET
        # T_Mapping.load_state_dict(torch.load('./classification_models/Densenet_01/mri_to_pet_mapping/Mapping/Mapping55_TLoss0.3757_TrainACC1.0_TestACC0.8754_TestSEN0.8264_TestSPE0.9135_TestAUC0.9211_F1S0.853.pth'))#'''!!!!!!!!!!!!!!!!!!genPET1to2!!!!!!!!!!!!!!!!!!!!!!!!'''
        
        # AD vs NC
        # T2
        # T_Mapping.load_state_dict(torch.load('./classification_models/Densenet_01/mri_to_pet_mapping/Mapping_t2/Mapping69_TLoss0.4658_TrainACC1.0_TestACC0.8434_TestSEN0.7778_TestSPE0.9106_TestAUC0.8882_F1S0.834.pth'))#'''!!!!!!!!!!!!!!!!!!genPET1to2!!!!!!!!!!!!!!!!!!!!!!!!'''
 

        # pMCI vs sMCI
        # PET
        T.load_state_dict(torch.load('./classification_models/Densenet_01/mri_to_pet_mapping/PET/PET79_TLoss0.0036_TrainACC1.0_TestACC0.8967_TestSEN0.8889_TestSPE0.9027_TestAUC0.9421_F1S0.8827_T2.pth'))#'''!!!!!!!!!!!!!!!!!!genPET1to2!!!!!!!!!!!!!!!!!!!!!!!!'''
        

        #load dataset
        dataset_train = MRIandPETdataset()
        data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size = self.opts.batch_size, shuffle = True, num_workers = self.opts.num_workers)
        data_loader_valid = torch.utils.data.DataLoader(dataset_train, batch_size = self.opts.batch_size_ValidAndTest, shuffle = False, num_workers = self.opts.num_workers)

        dataset_test = TestDataset()
        data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = self.opts.batch_size_ValidAndTest, shuffle = False, num_workers = self.opts.num_workers)

        #optimizer
        dis_optimizer = torch.optim.Adam(D.parameters(), lr=self.opts.lr_dis,
                                         betas=[self.opts.beta1_dis, 0.999])
        gen_optimizer = torch.optim.Adam(G.parameters(), lr=self.opts.lr_gen,
                                         betas=[self.opts.beta1_gen, 0.999])

        if not os.path.exists(self.opts.ckpt_dir):
            os.makedirs(self.opts.ckpt_dir)
        if not os.path.exists(self.opts.output_dir):
            os.makedirs(self.opts.output_dir)
        logfile = open(self.opts.ckpt_dir + '/log', 'w+')


        l1_loss = nn.L1Loss().cuda()
        mse_loss =nn.MSELoss().cuda()
        ssim_loss = SSIM().cuda()
        cirterion = nn.CrossEntropyLoss().cuda()
        criterion_bce = nn.BCELoss().cuda()

        saliency_MAP_ = np.load('saliency_map.npy')
        saliency_MAP_ = Variable(torch.Tensor(saliency_MAP_).cuda(), requires_grad=False)
    
        #training process
        for epoch in range(self.opts.num_epoch):           ######### begin #########
            start_time = time.time()

            #loss
            gen_loss_epoch, dis_loss_epoch = [], []
            loss_1, loss_2, loss_3, loss_4 = [], [], [], []


            #train
            for train_data in data_loader_train:
                
                labels = train_data[1]
                labels_ = Variable(labels).cuda()
                images = train_data[0]
                _batch_size = images.size()[0]

                input_data1 = images[:, 0, :, :, :].view(_batch_size, 1, 76, 94, 76)
                input_data = Variable(torch.Tensor(input_data1).cuda(), requires_grad=False)
                input_data_ = Variable(torch.Tensor(input_data1).cuda(), requires_grad=False)
                
                obs_data1 = images[:, 1, :, :, :].view(_batch_size, 1, 76, 94, 76)
                obs_data = Variable(torch.Tensor(obs_data1).cuda(), requires_grad=False)  
                obs_data_ = Variable(torch.Tensor(obs_data1).cuda(), requires_grad=False)

                real_y = Variable(torch.ones((_batch_size, 1)).cuda())
                fake_y = Variable(torch.zeros((_batch_size, 1)).cuda())
                real_yy = Variable(torch.ones((_batch_size, 1)).cuda())

                mapping = T_Mapping(input_data)[1].cuda()
                mapping_ = T_Mapping(input_data)[1].cuda()


##########################################################################
##########################################################################
##########################################################################
                if epoch < 50:

                    for cj in range(1):

                        #Discriminator
                        for p in G.parameters():
                            p.requires_grad = False
                        for p in D.parameters():
                            p.requires_grad = True

                        dis_optimizer.zero_grad()

                        gen_res = G(input_data, mapping)
                        d_g = D(gen_res)
                        d_o = D(obs_data)                    

                        loss6 = criterion_bce(d_g, fake_y[:_batch_size])
                        loss7 = criterion_bce(d_o, real_y[:_batch_size])

                        dis_loss = loss6 + loss7 

                        dis_loss.backward()
                        dis_optimizer.step()

                    for ck in range(1):

                        #Generator
                        for p in G.parameters():
                            p.requires_grad = True
                        for p in D.parameters():
                            p.requires_grad = False

                        gen_optimizer.zero_grad()

                        gen_ress = G(input_data_, mapping_)
                        gen_res_local = gen_ress * saliency_MAP_
                        obs_data_local = obs_data_.detach() * saliency_MAP_ 
                        result_t = T(gen_ress)

                        d_gg = D(gen_ress)
                        loss1 = mse_loss(gen_ress, obs_data_) + mse_loss(gen_res_local, obs_data_local) 
                        loss2 = l1_loss(gen_ress, obs_data_) + l1_loss(gen_res_local, obs_data_local) 
                        loss3 = ssim_loss(gen_ress, obs_data_) + ssim_loss(gen_res_local, obs_data_local) 
                        loss4 = cirterion(result_t, labels_)
                        loss5 = criterion_bce(d_gg, real_yy[:_batch_size])


                        a = loss1.cpu().data
                        b = loss2.cpu().data
                        c = loss3.cpu().data


                        max_value = max(a, b, c)
                        a_value = int(math.log(max_value/a, 10))
                        b_value = int(math.log(max_value/b, 10))
                        c_value = int(math.log(max_value/c, 10))

                        theta_a = 1
                        theta_b = 1
                        theta_c = 1


                        if a_value > 0:
                            theta_a = 10**a_value

                        if b_value > 0:
                            theta_b = 10**b_value

                        if c_value > 0:
                            theta_c = 10**c_value                    

                        gen_loss = loss5 + theta_a * loss1 + theta_b * loss2 + theta_c * loss3 


                        gen_loss.backward(retain_graph=True)
                        gen_optimizer.step()


                    gen_loss_epoch.append(gen_loss.cpu().data)
                    dis_loss_epoch.append(dis_loss.cpu().data)
                  
##########################################################################
##########################################################################
##########################################################################
                else:

                    for p in G.parameters():
                        p.requires_grad = True
                    for p in D.parameters():
                        p.requires_grad = False

                    gen_optimizer.zero_grad()

                    gen_res = G(input_data_, mapping_)
                    result_t = T(gen_res)
                    d_g = D(gen_res)

                    gen_res_local = gen_res * saliency_MAP_
                    obs_data_local = obs_data_.detach() * saliency_MAP_
                    # loss1 = mse_loss(gen_res, obs_data) + mse_loss(gen_local_res, obs_data_local) + mse_loss(gen_fusion, obs_data)
                    # loss2 = l1_loss(gen_res, obs_data) + l1_loss(gen_local_res, obs_data_local) + l1_loss(gen_fusion, obs_data)
                    # loss3 = ssim_loss(gen_res, obs_data) + ssim_loss(gen_local_res, obs_data_local) + ssim_loss(gen_fusion, obs_data)
                    loss1 = mse_loss(gen_res, obs_data_) #+ mse_loss(gen_res_local, obs_data_local) 
                    loss2 = l1_loss(gen_res, obs_data_) #+ l1_loss(gen_res_local, obs_data_local) 
                    loss3 = ssim_loss(gen_res, obs_data_) #+ ssim_loss(gen_res_local, obs_data_local) 
                    loss4 = cirterion(result_t, labels_)
                    loss5 = criterion_bce(d_g, real_y[:_batch_size])

                    a = loss1.cpu().data
                    b = loss2.cpu().data
                    c = loss3.cpu().data


                    max_value = max(a, b, c)
                    a_value = int(math.log(max_value/a, 10))
                    b_value = int(math.log(max_value/b, 10))
                    c_value = int(math.log(max_value/c, 10))

                    theta_a = 1
                    theta_b = 1
                    theta_c = 1

                    if a_value > 0:
                        theta_a = 10**a_value

                    if b_value > 0:
                        theta_b = 10**b_value

                    if c_value > 0:
                        theta_c = 10**c_value

                    gen_loss = theta_a * loss1 + theta_b * loss2 + theta_c * loss3  + loss4 + loss5                 #
                    gen_loss.backward(retain_graph=True)
                    gen_optimizer.step()
                    gen_loss_epoch.append(gen_loss.cpu().data)


#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
            if epoch < 50:

                #valid 
                TP_ = 0
                FP_ = 0
                FN_ = 0
                TN_ = 0
                for val_trian_data in data_loader_valid:

                    val_trian_imgs = val_trian_data[0]
                    val_trian_labels = val_trian_data[1]
                    val_trian_labels_ = Variable(val_trian_labels).cuda()
                    val_trian_data_batch_size = val_trian_imgs.size()[0]


                    input_data = val_trian_imgs[:, 0, :, :, :].view(val_trian_data_batch_size, 1, 76, 94, 76)
                    input_data = Variable(torch.Tensor(input_data).cuda(), requires_grad=False)

                    obs_data = val_trian_imgs[:, 1, :, :, :].view(val_trian_data_batch_size, 1, 76, 94, 76)
                    obs_data = Variable(torch.Tensor(obs_data).cuda(), requires_grad=False)


                    mapping = T_Mapping(input_data)[1].cuda()
                    gen_res = G(input_data, mapping)

                    result_c = T(gen_res)

                    out_c = F.softmax(result_c, dim=1)
                    _, predicted = torch.max(out_c.data, 1)
                    PREDICTED_ = predicted.data.cpu().numpy()
                    REAL_ = val_trian_labels_.data.cpu().numpy()


                    if PREDICTED_ == 1 and REAL_ == 1:
                        TP_ += 1
                    elif PREDICTED_ == 1 and REAL_ == 0:
                        FP_ += 1
                    elif PREDICTED_ == 0 and REAL_ == 1:
                        FN_ += 1 
                    elif PREDICTED_ == 0 and REAL_ == 0:
                        TN_ += 1
                    else:
                        continue

                train_acc = (TP_ + TN_)/(TP_ + TN_ + FP_ + FN_)

                ori_data = np.squeeze(input_data.data.cpu().numpy())
                ori_data = ori_data *255

                real_data = np.squeeze(obs_data.data.cpu().numpy())
                real_data = real_data *255

                gen_data = np.squeeze(gen_res.data.cpu().numpy())
                gen_data = gen_data *255

                if len(gen_data.shape) == 3:
                    for i in range(76):

                        img_2d_ = gen_data[:, :, i]
                        img_fileName_ =  str(i) + '_2g.png'
                        end_path_ = os.path.join(self.opts.output_dir_train, img_fileName_)
                        cv2.imwrite(end_path_, img_2d_)

                        img_2d_r = real_data[:, :, i]
                        img_fileName_r =  str(i) + '_1pet.png'
                        end_path_r = os.path.join(self.opts.output_dir_train, img_fileName_r)
                        cv2.imwrite(end_path_r, img_2d_r)

                        img_2d_o = ori_data[:, :, i]
                        img_fileName_o =  str(i) + '_0mri.png'
                        end_path_o = os.path.join(self.opts.output_dir_train, img_fileName_o)
                        cv2.imwrite(end_path_o, img_2d_o)              

                elif len(gen_data.shape) == 4:
                    for i in range(76):

                        img_2d_ = gen_data[0, :, :, i]
                        img_fileName_ =  str(i) + '_2g.png'
                        end_path_ = os.path.join(self.opts.output_dir_train, img_fileName_)
                        cv2.imwrite(end_path_, img_2d_)

                        img_2d_r = real_data[:, :, i]
                        img_fileName_r =  str(i) + '_1pet.png'
                        end_path_r = os.path.join(self.opts.output_dir_train, img_fileName_r)
                        cv2.imwrite(end_path_r, img_2d_r)

                        img_2d_o = ori_data[:, :, i]
                        img_fileName_o =  str(i) + '_0mri.png'
                        end_path_o = os.path.join(self.opts.output_dir_train, img_fileName_o)
                        cv2.imwrite(end_path_o, img_2d_o)
                
                #test
                TP = 0
                FP = 0
                FN = 0            
                TN = 0

                labels = []
                scores = []
                for val_test_data in data_loader_test:

                    val_test_imgs = val_test_data[0]
                    val_test_labels = val_test_data[1]
                    val_test_labels_ = Variable(val_test_labels).cuda()
                    val_test_data_batch_size = val_test_imgs.size()[0]

                    input_data = val_test_imgs[:, 0, :, :, :].view(val_test_data_batch_size, 1, 76, 94, 76)
                    input_data = Variable(torch.Tensor(input_data).cuda(), requires_grad=False)

                    obs_data = val_test_imgs[:, 1, :, :, :].view(val_test_data_batch_size, 1, 76, 94, 76)
                    obs_data = Variable(torch.Tensor(obs_data).cuda(), requires_grad=False)

                    mapping = T_Mapping(input_data)[1].cuda()
                    gen_res = G(input_data, mapping)


                    result_c_ = T(gen_res)
                    out_c = F.softmax(result_c_, dim=1)
                    score = out_c[0][1].data.cpu().item()
                    score = round(score, 4)
                    scores.append(score)
                    
                    _, predicted__ = torch.max(out_c.data, 1)

                    PREDICTED = predicted__.data.cpu().numpy()
                    REAL = val_test_labels_.data.cpu().numpy()
                    labels.append(REAL)

                    if PREDICTED == 1 and REAL == 1:
                        TP += 1
                    elif PREDICTED == 1 and REAL == 0:
                        FP += 1
                    elif PREDICTED == 0 and REAL == 1:
                        FN += 1 
                    elif PREDICTED == 0 and REAL == 0:
                        TN += 1
                    else:
                        continue

                test_acc = (TP + TN)/(TP + TN + FP + FN)
                test_sen = TP/(TP + FN)
                test_spe = TN/(FP + TN)

                fpr, tpr, thresholds = roc_curve(labels, scores)
                roc_auc = auc(fpr, tpr)

                ori_data = np.squeeze(input_data.data.cpu().numpy())
                ori_data = ori_data *255

                real_data = np.squeeze(obs_data.data.cpu().numpy())
                real_data = real_data *255                   

                gen_data = np.squeeze(gen_res.data.cpu().numpy())
                gen_data = gen_data *255

                if len(gen_data.shape) == 3:
                    for i in range(76):

                        img_2d_ = gen_data[:, :, i]
                        img_fileName_ =  str(i) + '_2g.png'
                        end_path_ = os.path.join(self.opts.output_dir_test, img_fileName_)
                        cv2.imwrite(end_path_, img_2d_)

                        img_2d_r = real_data[:, :, i]
                        img_fileName_r =  str(i) + '_1pet.png'
                        end_path_r = os.path.join(self.opts.output_dir_test, img_fileName_r)
                        cv2.imwrite(end_path_r, img_2d_r)

                        img_2d_o = ori_data[:, :, i]
                        img_fileName_o =  str(i) + '_0mri.png'
                        end_path_o = os.path.join(self.opts.output_dir_test, img_fileName_o)
                        cv2.imwrite(end_path_o, img_2d_o)

                elif len(gen_data.shape) == 4:
                    for i in range(76):

                        img_2d_ = gen_data[0, :, :, i]
                        img_fileName_ =  str(i) + '_2g.png'
                        end_path_ = os.path.join(self.opts.output_dir_test, img_fileName_)
                        cv2.imwrite(end_path_, img_2d_)

                        img_2d_r = real_data[:, :, i]
                        img_fileName_r =  str(i) + '_1pet.png'
                        end_path_r = os.path.join(self.opts.output_dir_test, img_fileName_r)
                        cv2.imwrite(end_path_r, img_2d_r)

                        img_2d_o = ori_data[0, :, :, i]
                        img_fileName_o =  str(i) + '_0mri.png'
                        end_path_o = os.path.join(self.opts.output_dir_test, img_fileName_o)
                        cv2.imwrite(end_path_o, img_2d_o)                
                

                end_time = time.time()

                print(
                    'Epoch {:d}/{:d}'.format(epoch + 1, self.opts.num_epoch),
                    )


                print(
                    'Gen_loss: {:.4f}'.format(np.mean(gen_loss_epoch)),
                    'Dis_loss: {:.4f}'.format(np.mean(dis_loss_epoch)),
                    'Train_ACC:{:.4f} {}/{}'.format(round(train_acc, 4), (TP_ + TN_), (TP_ + TN_ + FP_ + FN_)),
                    'Test_ACC:{:.4f} {}/{}'.format(round(test_acc, 4), (TP + TN), (TP + TN + FP + FN)),
                    'Test_SEN:{:.4f} {}/{}'.format(round(test_sen, 4), TP , (TP + FN)),
                    'Test_SPE:{:.4f} {}/{}'.format(round(test_spe, 4), TN, (FP + TN)),
                    'Test_AUC:{:.4f}'.format(round(roc_auc, 4) )
                    )

                if epoch % self.opts.log_epoch == 0:

                    torch.save(D, self.opts.ckpt_dir + '/dis_ckpt_{}_Loss{}_TrainACC{}_TestACC{}_TestSEN{}_TestSPE{}_TestAUC{}.pth'.format(
                        epoch + 1,
                        np.mean(dis_loss_epoch), 
                        round(train_acc, 4),
                        round(test_acc, 4),
                        round(test_sen, 4),
                        round(test_spe, 4),
                        round(roc_auc, 4)               
                    ))

                    torch.save(G, self.opts.ckpt_dir + '/gen_ckpt_{}_Loss{}_TrainACC{}_TestACC{}_TestSEN{}_TestSPE{}_TestAUC{}.pth'.format(
                        epoch + 1,
                        np.mean(gen_loss_epoch), 
                        round(train_acc, 4),
                        round(test_acc, 4),
                        round(test_sen, 4),
                        round(test_spe, 4),
                        round(roc_auc, 4)                       
                    ))

#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################

            else:
                #valid 
                TP_ = 0
                FP_ = 0
                FN_ = 0
                TN_ = 0
                for val_trian_data in data_loader_valid:

                    val_trian_imgs = val_trian_data[0]
                    val_trian_labels = val_trian_data[1]
                    val_trian_labels_ = Variable(val_trian_labels).cuda()
                    val_trian_data_batch_size = val_trian_imgs.size()[0]


                    input_data = val_trian_imgs[:, 0, :, :, :].view(val_trian_data_batch_size, 1, 76, 94, 76)
                    input_data = Variable(torch.Tensor(input_data).cuda(), requires_grad=False)

                    obs_data = val_trian_imgs[:, 1, :, :, :].view(val_trian_data_batch_size, 1, 76, 94, 76)
                    obs_data = Variable(torch.Tensor(obs_data).cuda(), requires_grad=False)

                    mapping = T_Mapping(input_data)[1].cuda()
                    gen_res = G(input_data, mapping)


                    result_c = T(gen_res)                   


                    out_c = F.softmax(result_c, dim=1)
                    _, predicted = torch.max(out_c.data, 1)
                    PREDICTED_ = predicted.data.cpu().numpy()
                    REAL_ = val_trian_labels_.data.cpu().numpy()


                    if PREDICTED_ == 1 and REAL_ == 1:
                        TP_ += 1
                    elif PREDICTED_ == 1 and REAL_ == 0:
                        FP_ += 1
                    elif PREDICTED_ == 0 and REAL_ == 1:
                        FN_ += 1 
                    elif PREDICTED_ == 0 and REAL_ == 0:
                        TN_ += 1
                    else:
                        continue

                train_acc = (TP_ + TN_)/(TP_ + TN_ + FP_ + FN_)

                ori_data = np.squeeze(input_data.data.cpu().numpy())
                ori_data = ori_data *255

                real_data = np.squeeze(obs_data.data.cpu().numpy())
                real_data = real_data *255

                gen_data = np.squeeze(gen_res.data.cpu().numpy())
                gen_data = gen_data *255

                if len(gen_data.shape) == 3:
                    for i in range(76):

                        img_2d_ = gen_data[:, :, i]
                        img_fileName_ =  str(i) + '_2g.png'
                        end_path_ = os.path.join(self.opts.output_dir_train, img_fileName_)
                        cv2.imwrite(end_path_, img_2d_)

                        img_2d_r = real_data[:, :, i]
                        img_fileName_r =  str(i) + '_1pet.png'
                        end_path_r = os.path.join(self.opts.output_dir_train, img_fileName_r)
                        cv2.imwrite(end_path_r, img_2d_r)

                        img_2d_o = ori_data[:, :, i]
                        img_fileName_o =  str(i) + '_0mri.png'
                        end_path_o = os.path.join(self.opts.output_dir_train, img_fileName_o)
                        cv2.imwrite(end_path_o, img_2d_o)              

                elif len(gen_data.shape) == 4:
                    for i in range(76):

                        img_2d_ = gen_data[0, :, :, i]
                        img_fileName_ =  str(i) + '_2g.png'
                        end_path_ = os.path.join(self.opts.output_dir_train, img_fileName_)
                        cv2.imwrite(end_path_, img_2d_)

                        img_2d_r = real_data[:, :, i]
                        img_fileName_r =  str(i) + '_1pet.png'
                        end_path_r = os.path.join(self.opts.output_dir_train, img_fileName_r)
                        cv2.imwrite(end_path_r, img_2d_r)

                        img_2d_o = ori_data[:, :, i]
                        img_fileName_o =  str(i) + '_0mri.png'
                        end_path_o = os.path.join(self.opts.output_dir_train, img_fileName_o)
                        cv2.imwrite(end_path_o, img_2d_o)
                
                #test
                TP = 0
                FP = 0
                FN = 0            
                TN = 0

                labels = []
                scores = []
                for val_test_data in data_loader_test:

                    val_test_imgs = val_test_data[0]
                    val_test_labels = val_test_data[1]
                    val_test_labels_ = Variable(val_test_labels).cuda()
                    val_test_data_batch_size = val_test_imgs.size()[0]

                    input_data = val_test_imgs[:, 0, :, :, :].view(val_test_data_batch_size, 1, 76, 94, 76)
                    input_data = Variable(torch.Tensor(input_data).cuda(), requires_grad=False)

                    obs_data = val_test_imgs[:, 1, :, :, :].view(val_test_data_batch_size, 1, 76, 94, 76)
                    obs_data = Variable(torch.Tensor(obs_data).cuda(), requires_grad=False)

                    mapping = T_Mapping(input_data)[1].cuda()
                    gen_res = G(input_data, mapping)


                    result_c_ = T(gen_res)
                    out_c = F.softmax(result_c_, dim=1)
                    score = out_c[0][1].data.cpu().item()
                    score = round(score, 4)
                    scores.append(score)
                    
                    _, predicted__ = torch.max(out_c.data, 1)

                    PREDICTED = predicted__.data.cpu().numpy()
                    REAL = val_test_labels_.data.cpu().numpy()
                    labels.append(REAL)

                    if PREDICTED == 1 and REAL == 1:
                        TP += 1
                    elif PREDICTED == 1 and REAL == 0:
                        FP += 1
                    elif PREDICTED == 0 and REAL == 1:
                        FN += 1 
                    elif PREDICTED == 0 and REAL == 0:
                        TN += 1
                    else:
                        continue

                test_acc = (TP + TN)/(TP + TN + FP + FN)
                test_sen = TP/(TP + FN)
                test_spe = TN/(FP + TN)

                fpr, tpr, thresholds = roc_curve(labels, scores)
                roc_auc = auc(fpr, tpr)

                ori_data = np.squeeze(input_data.data.cpu().numpy())
                ori_data = ori_data *255

                real_data = np.squeeze(obs_data.data.cpu().numpy())
                real_data = real_data *255

                gen_data = np.squeeze(gen_res.data.cpu().numpy())
                gen_data = gen_data *255


                if len(gen_data.shape) == 3:
                    for i in range(76):

                        img_2d_ = gen_data[:, :, i]
                        img_fileName_ =  str(i) + '_2g.png'
                        end_path_ = os.path.join(self.opts.output_dir_test, img_fileName_)
                        cv2.imwrite(end_path_, img_2d_)

                        img_2d_r = real_data[:, :, i]
                        img_fileName_r =  str(i) + '_1pet.png'
                        end_path_r = os.path.join(self.opts.output_dir_test, img_fileName_r)
                        cv2.imwrite(end_path_r, img_2d_r)

                        img_2d_o = ori_data[:, :, i]
                        img_fileName_o =  str(i) + '_0mri.png'
                        end_path_o = os.path.join(self.opts.output_dir_test, img_fileName_o)
                        cv2.imwrite(end_path_o, img_2d_o)

                elif len(gen_data.shape) == 4:
                    for i in range(76):

                        img_2d_ = gen_data[0, :, :, i]
                        img_fileName_ =  str(i) + '_2g.png'
                        end_path_ = os.path.join(self.opts.output_dir_test, img_fileName_)
                        cv2.imwrite(end_path_, img_2d_)

                        img_2d_r = real_data[:, :, i]
                        img_fileName_r =  str(i) + '_1pet.png'
                        end_path_r = os.path.join(self.opts.output_dir_test, img_fileName_r)
                        cv2.imwrite(end_path_r, img_2d_r)

                        img_2d_o = ori_data[0, :, :, i]
                        img_fileName_o =  str(i) + '_0mri.png'
                        end_path_o = os.path.join(self.opts.output_dir_test, img_fileName_o)
                        cv2.imwrite(end_path_o, img_2d_o)                
                

                end_time = time.time()

                print(
                    'Epoch {:d}/{:d}'.format(epoch + 1, self.opts.num_epoch),
                    )

                print(
                    'Gen_loss: {:.4f}'.format(np.mean(gen_loss_epoch)),
                    'Train_ACC:{:.4f} {}/{}'.format(round(train_acc, 4), (TP_ + TN_), (TP_ + TN_ + FP_ + FN_)),
                    'Test_ACC:{:.4f} {}/{}'.format(round(test_acc, 4), (TP + TN), (TP + TN + FP + FN)),
                    'Test_SEN:{:.4f} {}/{}'.format(round(test_sen, 4), TP , (TP + FN)),
                    'Test_SPE:{:.4f} {}/{}'.format(round(test_spe, 4), TN, (FP + TN)),
                    'Test_AUC:{:.4f}'.format(round(roc_auc, 4) )
                    )

                if epoch % self.opts.log_epoch == 0:
                    torch.save(G, self.opts.ckpt_dir + '/gen_ckpt_{}_Loss{}_TrainACC{}_TestACC{}_TestSEN{}_TestSPE{}_TestAUC{}.pth'.format(
                        epoch + 1,
                        np.mean(gen_loss_epoch), 
                        round(train_acc, 4),
                        round(test_acc, 4),
                        round(test_sen, 4),
                        round(test_spe, 4),
                        round(roc_auc, 4)                       
                    ))

        logfile.close()  



#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################



    def test(self):

        Gen = Generator(self.opts).cuda()
        SAMPLE_PATH = './OASIS_T1_PET_T2_ADCN_255'
        WORKERS = 0
        BATCH_SIZE = 1

        dataset = TestDataset()
        data_loader_all = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)

        # ckpt_gen = './checkpoint_t1_pet/gen_ckpt_157_Loss0.8287625908851624_TrainACC1.0_TestACC0.845_TestSEN0.875_TestSPE0.8216_TestAUC0.9236.pth'
        ckpt_gen = './checkpoint_t1_pet/gen_ckpt_86_Loss1.0860906839370728_TrainACC1.0_TestACC0.8693_TestSEN0.7917_TestSPE0.9297_TestAUC0.9063.pth'
        Gen = torch.load(ckpt_gen)

        T = densenet21().cuda() 
        # T.load_state_dict(torch.load('./classification_models/Densenet_01/pet/36_TLoss0.0036_TrainACC1.0_TestACC0.9027_TestSEN0.8542_TestSPE0.9405_TestAUC0.9479_T.pth'))#'''!!!!!!!!!!!!!!!!!!genPET1to2!!!!!!!!!!!!!!!!!!!!!!!!'''
        T.load_state_dict(torch.load('./classification_models/Densenet_01/petnew/38_TLoss0.0057_TrainACC1.0_TestACC0.9179_TestSEN0.875_TestSPE0.9514_TestAUC0.9465_T.pth'))#'''!!!!!!!!!!!!!!!!!!genPET1to2!!!!!!!!!!!!!!!!!!!!!!!!'''     
        # T.eval()

        T_Mapping = densenet21_mri().cuda()
        T_Mapping.load_state_dict(torch.load('./classification_models/Densenet_01/mri_to_pet_mapping/Mapping/Mapping55_TLoss0.3757_TrainACC1.0_TestACC0.8754_TestSEN0.8264_TestSPE0.9135_TestAUC0.9211_F1S0.853.pth'))#'''!!!!!!!!!!!!!!!!!!genPET1to2!!!!!!!!!!!!!!!!!!!!!!!!'''
 
        method = 'd'
        iteration = 0
        TP = 0
        FP = 0
        FN = 0
        TN = 0

        labels = []
        scores = []

        for val_test_data in data_loader_all:
            iteration += 1


            val_test_imgs = val_test_data[0]
            val_test_labels = val_test_data[1]
            val_test_labels_ = Variable(val_test_labels).cuda()

            label_judge = val_test_labels.data.cpu().item()

            val_test_data_batch_size = val_test_imgs.size()[0]
            # fname = val_test_data[2][0].split('.')[0]


            fake_images_ = val_test_imgs[:, 0, :, :, :].view(val_test_data_batch_size, 1, 76, 94, 76)
            fake_images_ = Variable(fake_images_.cuda(), requires_grad=False)

            real_images_ = val_test_imgs[:, 1, :, :, :].view(val_test_data_batch_size, 1, 76, 94, 76)
            real_images_ = Variable(real_images_.cuda(), requires_grad=False)

            mapping = T_Mapping(fake_images_)[1].cuda()
            generated = Gen(fake_images_, mapping)                           


            result_c = T(generated)
            out_c = F.softmax(result_c, dim=1)

            score = out_c[0][1].data.cpu().item()
            score = round(score, 4)
            scores.append(score)
            
            _, predicted__ = torch.max(out_c.data, 1)

            PREDICTED = predicted__.data.cpu().numpy()
            REAL = val_test_labels_.data.cpu().numpy()
            labels.append(REAL)

            if PREDICTED == 1 and REAL == 1:
                TP += 1
            elif PREDICTED == 1 and REAL == 0:
                FP += 1
            elif PREDICTED == 0 and REAL == 1:
                FN += 1 
            elif PREDICTED == 0 and REAL == 0:
                TN += 1
            else:
                continue


            ori_data = np.squeeze(fake_images_.data.cpu().numpy())
            ori_data = ori_data# * 255  

            real_data = np.squeeze(real_images_.data.cpu().numpy())
            real_data = real_data# *255

            gen_data = np.squeeze(generated.data.cpu().numpy())         
            gen_data = gen_data# * 255                                   

            file_name1 = os.path.join(SAMPLE_PATH,'{}_fake.mat'.format(iteration))
            sio.savemat(file_name1, {'data':gen_data})
            file_name2 = os.path.join(SAMPLE_PATH,'{}_pet.mat'.format(iteration))
            sio.savemat(file_name2, {'data':real_data})
            file_name3 = os.path.join(SAMPLE_PATH,'{}_mri.mat'.format(iteration))
            sio.savemat(file_name3, {'data':ori_data})

        test_acc = (TP + TN)/((TP + TN + FP + FN) +0.0001)
        test_sen = TP/((TP + FN)+0.0001)
        test_spe = TN/((FP + TN)+0.0001)

        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        print(
                'Test_ACC:{:.4f} {}/{}'.format(round(test_acc, 4), (TP + TN), (TP + TN + FP + FN)),
                'Test_SEN:{:.4f} {}/{}'.format(round(test_sen, 4), TP , (TP + FN)),
                'Test_SPE:{:.4f} {}/{}'.format(round(test_spe, 4), TN, (FP + TN)),
                'Test_AUC:{:.4f}'.format(round(roc_auc, 4) ),
            )


#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################



    def impute(self):

        Gen = Generator(self.opts).cuda()

        SAMPLE_PATH = './OASIS_T1_PET_T2_ADCN_255'

        WORKERS = 0
        BATCH_SIZE = 1

        dataset = TestDataset()
        data_loader_all = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)


        ### MRI-T1 -> PET first
        ckpt_gen = './checkpoint_mri_pet/gen_ckpt_157_Loss0.8287625908851624_TrainACC1.0_TestACC0.845_TestSEN0.875_TestSPE0.8216_TestAUC0.9236.pth'
        # ckpt_gen = './checkpoint_mri_pet/gen_ckpt_86_Loss1.0860906839370728_TrainACC1.0_TestACC0.8693_TestSEN0.7917_TestSPE0.9297_TestAUC0.9063.pth' #this one
        
        # ckpt_gen = './checkpoint/gen_ckpt_76_Loss165.65223693847656_TrainACC0.9102_TestACC0.8655_TestSEN0.7834_TestSPE0.93_TestAUC0.8694.pth'
        

        ### MRI-T1 -> MRI-T2
        # ckpt_gen = './checkpoint_t1_t2/gen_ckpt_298_Loss0.9006105065345764_TrainACC1.0_TestACC0.8594_TestSEN0.7619_TestSPE0.9593_TestAUC0.8576.pth'


        Gen = torch.load(ckpt_gen)


        ### MRI-T1 -> PET
        T = densenet21().cuda() 
        # T.load_state_dict(torch.load('./classification_models/Densenet_01/pet/36_TLoss0.0036_TrainACC1.0_TestACC0.9027_TestSEN0.8542_TestSPE0.9405_TestAUC0.9479_T.pth'))
        T.load_state_dict(torch.load('./classification_models/petnew/38_TLoss0.0057_TrainACC1.0_TestACC0.9179_TestSEN0.875_TestSPE0.9514_TestAUC0.9465_T.pth'))     
        # T.eval()

        ### MRI-T1 -> MRI-T2
        # T = densenet21_pet().cuda() 
        # T.load_state_dict(torch.load('./classification_models/Densenet_01/mri_to_pet_mapping/T2/PET63_TLoss0.0_TrainACC0.9966_TestACC0.8675_TestSEN0.8175_TestSPE0.9187_TestAUC0.9088_F1S0.8619_T2.pth'))     
        # T.eval()        


        ### MRI-T1 -> PET
        T_Mapping = densenet21_mri().cuda()
        T_Mapping.load_state_dict(torch.load('./mapping/Mapping55_TLoss0.3757_TrainACC1.0_TestACC0.8754_TestSEN0.8264_TestSPE0.9135_TestAUC0.9211_F1S0.853.pth'))

        ### MRI-T1 -> MRI-T2
        # T_Mapping = densenet21_mri().cuda()        
        # T_Mapping.load_state_dict(torch.load('/media/sdd/gaoxingyu/Project/FusionSaliency_GAN_Gau1/classification_models/Densenet_01/mri_to_pet_mapping/Mapping_t2/Mapping69_TLoss0.4658_TrainACC1.0_TestACC0.8434_TestSEN0.7778_TestSPE0.9106_TestAUC0.8882_F1S0.834.pth'))#'''!!!!!!!!!!!!!!!!!!genPET1to2!!!!!!!!!!!!!!!!!!!!!!!!'''
 

        iteration = 0
        TP = 0
        FP = 0
        FN = 0
        TN = 0

        labels = []
        scores = []

        for val_test_data in data_loader_all:
            iteration += 1


            val_test_imgs = val_test_data[0]
            val_test_labels = val_test_data[1]
            val_test_labels_ = Variable(val_test_labels).cuda()

            label_judge = val_test_labels.data.cpu().item()

            val_test_data_batch_size = val_test_imgs.size()[0]
            fname = val_test_data[2][0].split('.')[0]


            fake_images_ = val_test_imgs[:, 0, :, :, :].view(val_test_data_batch_size, 1, 76, 94, 76)
            fake_images_ = Variable(fake_images_.cuda(), requires_grad=False)

            real_images_ = val_test_imgs[:, 1, :, :, :].view(val_test_data_batch_size, 1, 76, 94, 76)
            real_images_ = Variable(real_images_.cuda(), requires_grad=False)


            mapping = T_Mapping(fake_images_)[1].cuda()
            generated = Gen(fake_images_, mapping)                           


            ori_data = np.squeeze(fake_images_.data.cpu().numpy())
            ori_data = ori_data * 255  

            real_data = np.squeeze(real_images_.data.cpu().numpy())
            real_data = real_data *255

            gen_data = np.squeeze(generated.data.cpu().numpy())         
            gen_data = gen_data * 255                                   


            # mri_pet = np.stack((ori_data, real_data, gen_data), axis=0)

            # nii_image_mri = nib.Nifti1Image(ori_data, np.eye(4))
            # nib.save(nii_image_mri, 'mri.nii.gz')

            # nii_image_pet = nib.Nifti1Image(real_data, np.eye(4))
            # nib.save(nii_image_pet, 'pet.nii.gz')

            # nii_image_gen = nib.Nifti1Image(gen_data, np.eye(4))
            # nib.save(nii_image_gen, 'gen.nii.gz')

            csize = ori_data.shape
            img4d = np.zeros(shape=(3, csize[0], csize[1], csize[2]), dtype='float32')
            img4d[0, :, :, :] = ori_data
            img4d[1, :, :, :] = gen_data
            img4d[2, :, :, :] = real_data

            file_name = os.path.join(SAMPLE_PATH, fname + '.mat') 
            sio.savemat(file_name, {'data':img4d})  
