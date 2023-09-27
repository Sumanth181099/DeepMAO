import os
from os import path, mkdir, listdir, makedirs
import sys
import shutil
import random
import math
import gdal
import glob
import timeit
import copy
import argparse 
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Pool, Queue, Process
from functools import partial
import rasterio
from rasterio import features
from affine import Affine
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms
import skimage
import skimage.segmentation
from skimage import measure, io
from skimage.morphology import square, erosion, dilation, remove_small_objects, remove_small_holes
from skimage.color import label2rgb
from scipy import ndimage
from shapely.wkt import dumps, loads
from shapely.geometry import shape, Polygon
import cv2
from PIL import Image
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
import base

import geopandas as gpd
import rasterio as rs
from rasterio.plot import show  # imshow for raster
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import SegmentationModel, SegmentationHead
from segmentation_models_pytorch.base import modules as md

from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)
from segmentation_models_pytorch.decoders.unet import UnetDecoder
from torch.utils.tensorboard import SummaryWriter
from math import ceil
from typing import Optional, Union, List
from utils_cutmix import *
from sklearn.manifold import TSNE
#import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

print(os.path.basename(__file__))

os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
cudnn.benchmark = False
cudnn.deterministic = True
torch.backends.cudnn.enabled = False
#cudnn.enabled = config.CUDNN.ENABLED

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
        
    def __call__(self, model_output):
        return (model_output[self.category, :, : ]*self.mask).sum()
        #print((model_output[self.category, :, : ]*self.mask).sum())
        #print((self.mask).sum())
        #return (self.mask).sum()
################################# MODEL 

############DeepMAO##################

class DeepMAO(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "efficientnet-b3",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = False,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        in_channels: int = 3,
        classes: int = 3
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
                                
        classes = 3
        
        self.decoder = UnetDecoder(
            encoder_channels=(3,40,32,48,136,384),
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
        )

        self.OClayer1 = nn.Conv2d(40,56,kernel_size=3, stride=1, padding=1)
        self.OC1_bn = nn.BatchNorm2d(56)
        self.OClayer2 = nn.Conv2d(56,64,kernel_size=3, stride=1, padding=1)
        self.OC2_bn = nn.BatchNorm2d(64)
        self.OClayer3 = nn.Conv2d(64,128,kernel_size=3, stride=1, padding=2, dilation=2)
        self.OC3_bn = nn.BatchNorm2d(128)
        self.OClayer4 = nn.Conv2d(128,16,kernel_size=3, stride=1, padding=2, dilation=2)
        self.OC4_bn = nn.BatchNorm2d(16)

        

 
    def forward(self, x):
        
        features = self.encoder(x)
        
        OCout = F.relu(self.OC1_bn(F.interpolate(self.OClayer1(features[1]),scale_factor =(1.2,1.2)))) 
        
        _,_,h1,w1 = features[0].shape #512
        
        OCout = F.relu(self.OC2_bn(F.interpolate(self.OClayer2(OCout), scale_factor =(1.2,1.2))))
        

        
        OCout = F.relu(self.OC3_bn(F.interpolate(self.OClayer3(OCout), scale_factor =(1.2,1.2))))
        
        OCout = F.relu(self.OC4_bn(F.interpolate(self.OClayer4(OCout), scale_factor =(1.15,1.15))))
       

        
        OCout = F.interpolate(OCout, size = (h1,w1))#625 to 512
        

        logit = self.decoder(*features) #16
        
        _,_,h,w = logit.shape
        
        
        if(logit.shape==OCout.shape):
            logit = torch.add(OCout, logit)
        else:
            OCout = F.interpolate(OCout,size=(h,w),mode='bilinear')
            logit = torch.add(OCout, logit)

        logit = self.segmentation_head(logit)
        
        return logit

#################### Simple Unet model ############################
class Simple_Unet(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "efficientnet-b3",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = False,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        in_channels: int = 3,
        classes: int = 3
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
                                
        classes = 3
        
        self.decoder = UnetDecoder(
            encoder_channels=(3,40,32,48,136,384),
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
        )

    def forward(self, x):
        
        features = self.encoder(x)
        logit = self.decoder(*features)
        logit = self.segmentation_head(logit)
        
        return logit
    
####################CALCULATE IOU####################

ref_folder = '/home/user/Perception/SN6_dataset/val_set/AOI_11_Rotterdam/masks' #this is the path of validation labels that are needed for function: calculate_acc_iou
file_names = sorted(os.listdir(ref_folder))

def calculate_acc_iou(pred_folder): #pred_folder is the folder where predictions are stored (inside '/wdata_***/pred_fold_{0}_0'). 
    inters_acum = 0
    union_acum = 0
    correct_acum = 0
    total_acum = 0
    izz=0

    result ="SpaceNet6\t\tIoU %\tacc %\n"

    for file_name in file_names:

        file_name_ref = file_name
        ref_path = path.join(ref_folder, file_name_ref)
        
        _, tail = os.path.split(file_name)
        file_name_pred = tail[:-4]+'.tif'
        pred_path = path.join(pred_folder,file_name_pred)

        ref = skimage.io.imread(ref_path)
        ref = (ref/255.0).astype(int)
        ref = ref[:,:,:5]
        pred = skimage.io.imread(pred_path)

        
        pred = 1.0 / (1 + np.exp(-pred)) #makes it from 0 to 1

        '''if izz==5:
            izz+=1
            print(file_name)'''
        
        izz+=1
        pred = torch.from_numpy(pred)

        pred = pred.numpy().reshape(-1)

        pred = (pred>0.5).astype(int)
        ref = ref.reshape(-1)


        inters = ref & pred
        union = ref | pred
        correct = ref == pred

        inters_count = np.count_nonzero(inters)
        union_count = np.count_nonzero(union)
        correct_count = np.count_nonzero(correct)
        total_count = ref.size

        inters_acum+=inters_count
        union_acum+=union_count
        correct_acum+=correct_count
        total_acum+=total_count

    overall_iou = inters_acum/float(union_acum)
    overall_acc = correct_acum/float(total_acum)
    result += "{0}\t\t{1}%\t{2}%\n".format("Overall", round(overall_iou*100, 2), round(overall_acc*100, 2))
    print(result)

    return round(overall_iou*100, 2)

############################## DATASET

def _blend(img1, img2, alpha):
    return img1 * alpha + (1 - alpha) * img2

_alpha = np.asarray([0.25, 0.25, 0.25, 0.25]).reshape((1, 1, 4))

def _grayscale(img):
    return np.sum(_alpha * img, axis=2, keepdims=True)

def saturation(img, alpha):
    gs = _grayscale(img)
    return _blend(img, gs, alpha)

def brightness(img, alpha):
    gs = np.zeros_like(img)
    return _blend(img, gs, alpha)

def contrast(img, alpha):
    gs = _grayscale(img)
    gs = np.repeat(gs.mean(), 4)
    return _blend(img, gs, alpha)

def parse_img_id(file_path, orients):
    file_name = file_path.split('/')[-1]
    stripname = '_'.join(file_name.split('_')[-4:-2])

    direction = int(orients.loc[stripname]['direction'])
    direction = torch.from_numpy(np.reshape(np.asarray([direction]), (1,1,1))).float()

    val = int(orients.loc[stripname]['val'])
    strip = torch.Tensor(np.zeros((len(orients.index), 1, 1))).float()
    strip[val] = 1

    coord = np.asarray([orients.loc[stripname]['coord_y']])
    coord = torch.from_numpy(np.reshape(coord, (1,1,1))).float() - 0.5
    return direction, strip, coord



class MyData_EO(Dataset): #THIS class is for EO, change name to 'MyData' if training/validating on EO
    def __init__(self, image_sar_paths, image_rgb_paths,  label_paths, train, test, crop_size = None,
        rot_prob = 0.3, scale_prob = 0.5, color_aug_prob = 0.0, fliplr_prob = 0.0, train_min_building_size=0,normalize=False, reorder_bands=0):
        super().__init__()
        self.image_sar_paths = image_sar_paths
        self.image_rgb_paths = image_rgb_paths
        self.label_paths = label_paths
        self.train = train
        self.test = test
        self.crop_size = crop_size
        self.rot_prob = rot_prob
        self.scale_prob = scale_prob
        self.color_aug_prob = color_aug_prob
        self.fliplr_prob = fliplr_prob
        self.train_min_building_size = train_min_building_size
        self.normalize = normalize
        self.orients = pd.read_csv(rot_out_path, index_col = 0)
        self.orients['val'] = list(range(len(self.orients.index)))
        self.reorder_bands = reorder_bands

    def __len__(self):
        return len(self.image_rgb_paths)

    def __getitem__(self,idx):
        if not self.test:
            sar = skimage.io.imread(self.image_sar_paths[idx])
            rgb = skimage.io.imread(self.image_rgb_paths[idx])
            
            rgb_full = sar
            
        m = np.where((rgb.sum(axis=2) > 0).any(1))
        ymin, ymax = np.amin(m), np.amax(m) + 1
        m = np.where((rgb.sum(axis=2) > 0).any(0))
        xmin, xmax = np.amin(m), np.amax(m) + 1

        if not self.test:
            rgb = rgb[ymin:ymax, xmin:xmax]
            sar = sar[ymin:ymax, xmin:xmax]
            msk = skimage.io.imread(self.label_paths[idx])
            msk_full = msk
            msk = msk[ymin:ymax, xmin:xmax]
            
    
        if self.train:
            msk = skimage.io.imread(self.label_paths[idx])
            msk = msk[ymin:ymax, xmin:xmax]

            pad = max(0, self.crop_size - sar.shape[0])
            msk = cv2.copyMakeBorder(msk, 0, pad, 0, 0, cv2.BORDER_CONSTANT, 0.0)
            rgb = cv2.copyMakeBorder(rgb, 0, pad, 0, 0, cv2.BORDER_CONSTANT, 0.0)
            
            sar = cv2.copyMakeBorder(sar, 0, pad, 0, 0, cv2.BORDER_CONSTANT, 0.0)
            x0 = random.randint(0, sar.shape[1] - self.crop_size)
            y0 = random.randint(0, sar.shape[0] - self.crop_size)
            
            msk = msk[y0 : y0 + self.crop_size, x0 : x0 + self.crop_size]
            
            rgb = rgb[y0 : y0 + self.crop_size, x0 : x0 + self.crop_size]

            sar = sar[y0 : y0 + self.crop_size, x0 : x0 + self.crop_size]

            rgb_512 = copy.deepcopy(rgb)
            
            msk_512 = copy.deepcopy(msk)

            if random.random() < args.rot_prob:
                rot_mat = cv2.getRotationMatrix2D((sar.shape[0] // 2, sar.shape[1] // 2), random.randint(0, 10) - 5, 1.0)
                msk = cv2.warpAffine(msk, rot_mat, msk.shape[:2], flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
                
                rgb = cv2.warpAffine(rgb, rot_mat, rgb.shape[:2], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                sar = cv2.warpAffine(sar, rot_mat, sar.shape[:2], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                

            if random.random() < args.scale_prob:
                rot_mat = cv2.getRotationMatrix2D((rgb.shape[0] // 2, rgb.shape[1] // 2), 0, random.uniform(0.5,2.0))
                msk = cv2.warpAffine(msk, rot_mat, msk.shape[:2], flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
                
                rgb = cv2.warpAffine(rgb, rot_mat, rgb.shape[:2], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                sar = cv2.warpAffine(sar, rot_mat, sar.shape[:2], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


        
            
            if random.random() < self.fliplr_prob:
                msk = np.fliplr(msk)
                
                rgb = np.fliplr(rgb)
                sar = np.fliplr(sar)
            

            direction, strip, coord = parse_img_id(self.image_sar_paths[idx], self.orients)
            if direction.item():
                rgb = np.fliplr(np.flipud(rgb))
                
                msk = np.fliplr(np.flipud(msk))
                

            sar = torch.from_numpy(sar.transpose((2, 0, 1)).copy()).float()
            
            weights = np.ones_like(msk[:,:,:1], dtype=float)
            regionlabels, regioncount = measure.label(msk[:,:,0], background=0, connectivity=1, return_num=True)
            regionproperties = measure.regionprops(regionlabels)

            weights_512 = np.ones_like(msk_512[:,:,:1], dtype=float)
            regionlabels_512, regioncount_512 = measure.label(msk_512[:,:,0], background=0, connectivity=1, return_num=True)
            regionproperties_512 = measure.regionprops(regionlabels_512)

            for bl in range(regioncount):
                if regionproperties[bl].area < self.train_min_building_size:
                    msk[:,:,0][regionlabels == bl+1] = 0
                    msk[:,:,1][regionlabels == bl+1] = 0
                weights[regionlabels == bl+1] = 1024.0 / regionproperties[bl].area
            for bl in range(regioncount_512):
                if regionproperties_512[bl].area < self.train_min_building_size:
                    msk_512[:,:,0][regionlabels_512 == bl+1] = 0
                    msk_512[:,:,1][regionlabels_512 == bl+1] = 0
                weights_512[regionlabels_512 == bl+1] = 1024.0 / regionproperties_512[bl].area

            msk_512[:, :, :3] = (msk_512[:, :, :3] > 1) * 1
            msk[:, :, :3] = (msk[:, :, :3] > 1) * 1
            weights = torch.from_numpy(weights.transpose((2, 0, 1)).copy()).float()
            msk = torch.from_numpy(msk.transpose((2, 0, 1)).copy()).float()
            rgb_512 = torch.from_numpy(rgb_512.transpose((2, 0, 1)).copy()).float()
            msk_512 = torch.from_numpy(msk_512.transpose((2, 0, 1)).copy()).float()
            rgb = torch.from_numpy(rgb.transpose((2, 0, 1)).copy()).float()
        else:
            rgb_512 = torch.Tensor(0)
            msk_512 = torch.Tensor(0)
            x0 = torch.Tensor(0)
            y0 = torch.Tensor(0)
            
            direction, _,_ = parse_img_id(self.image_sar_paths[idx], self.orients)
            
            if direction.item():
                sar = np.fliplr(np.flipud(sar))
            if self.reorder_bands == 1:
                sar = sar[[2,3,0,1]]
            elif self.reorder_bands == 2:
                sar = sar[[1,3,0,2]]
            elif self.reorder_bands == 3:
                sar = sar[[0,3,1,2]]
            sar = torch.from_numpy(sar.transpose((2, 0, 1)).copy()).float()
            rgb = torch.from_numpy(rgb.transpose((2, 0, 1)).copy()).float()
            idx_msk=weights = regioncount = torch.Tensor([0])
            msk = torch.from_numpy(msk.transpose((2, 0, 1)).copy()).float()
            
        return {"mask": msk, "rgb": rgb, "rgb_512": rgb_512, "msk_512": msk_512, "rgb_full": rgb_full, "mask_full": msk_full, "x0" : x0, "y0" : y0, 'img_name': self.image_sar_paths[idx],
                'ymin': ymin, 'xmin': xmin, 'b_count': regioncount, 'weights': weights, "direction": direction, "sar":sar}
    
class MyData_SAR(Dataset): #THIS class is for SAR, change name to 'MyData' if training/validating on SAR
    def __init__(self, image_sar_paths, image_rgb_paths,  label_paths, train, test, crop_size = None,
        rot_prob = 0.3, scale_prob = 0.5, color_aug_prob = 0.0, fliplr_prob = 0.0, train_min_building_size=0,normalize=False,reorder_bands=0):
        super().__init__()
        self.image_sar_paths = image_sar_paths
        self.image_rgb_paths = image_rgb_paths
        self.label_paths = label_paths
        self.train = train
        self.test = test
        self.crop_size = crop_size
        self.rot_prob = rot_prob
        self.scale_prob = scale_prob
        self.color_aug_prob = color_aug_prob
        self.fliplr_prob = fliplr_prob
        self.train_min_building_size = train_min_building_size
        self.normalize = normalize
        self.orients = pd.read_csv(rot_out_path, index_col = 0)
        self.orients['val'] = list(range(len(self.orients.index)))
        self.reorder_bands = reorder_bands

    def __len__(self):
        return len(self.image_sar_paths)

    def __getitem__(self,idx):
        if not self.test:
            sar = skimage.io.imread(self.image_sar_paths[idx])
            sar_full = sar
            
        m = np.where((sar.sum(axis=2) > 0).any(1))
        ymin, ymax = np.amin(m), np.amax(m) + 1
        m = np.where((sar.sum(axis=2) > 0).any(0))
        xmin, xmax = np.amin(m), np.amax(m) + 1

        if not self.test:
            sar = sar[ymin:ymax, xmin:xmax]
            msk = skimage.io.imread(self.label_paths[idx])
            msk_full = msk
            msk = msk[ymin:ymax, xmin:xmax]
            
    
        if self.train:
            msk = skimage.io.imread(self.label_paths[idx])
            msk = msk[ymin:ymax, xmin:xmax]

            pad = max(0, self.crop_size - sar.shape[0])
            msk = cv2.copyMakeBorder(msk, 0, pad, 0, 0, cv2.BORDER_CONSTANT, 0.0)

            sar = cv2.copyMakeBorder(sar, 0, pad, 0, 0, cv2.BORDER_CONSTANT, 0.0)
            
            if random.random() < args.rot_prob:
                rot_mat = cv2.getRotationMatrix2D((sar.shape[0] // 2, sar.shape[1] // 2), random.randint(0, 10) - 5, 1.0)
                msk = cv2.warpAffine(msk, rot_mat, msk.shape[:2], flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
                #sar_rot = cv2.warpAffine(sar, rot_mat, sar.shape[:2], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                sar = cv2.warpAffine(sar, rot_mat, sar.shape[:2], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

            if random.random() < args.scale_prob:
                rot_mat = cv2.getRotationMatrix2D((sar.shape[0] // 2, sar.shape[1] // 2), 0, random.uniform(0.5,2.0))
                msk = cv2.warpAffine(msk, rot_mat, msk.shape[:2], flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
                #sar_scale = cv2.warpAffine(sar, rot_mat, sar.shape[:2], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                sar = cv2.warpAffine(sar, rot_mat, sar.shape[:2], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            
            x0 = random.randint(0, sar.shape[1] - self.crop_size)
            y0 = random.randint(0, sar.shape[0] - self.crop_size)
            msk = msk[y0 : y0 + self.crop_size, x0 : x0 + self.crop_size]
            
            sar = sar[y0 : y0 + self.crop_size, x0 : x0 + self.crop_size]

            sar_512 = copy.deepcopy(sar)
            msk_512 = copy.deepcopy(msk)

            if random.random() < self.fliplr_prob:
                msk = np.fliplr(msk)
                #sar_flip = np.fliplr(sar)
                sar = np.fliplr(sar)

            direction, strip, coord = parse_img_id(self.image_sar_paths[idx], self.orients)
            if direction.item():
                sar = np.fliplr(np.flipud(sar))
                msk = np.fliplr(np.flipud(msk))

            sar = torch.from_numpy(sar.transpose((2, 0, 1)).copy()).float()
            if self.reorder_bands == 1:
                sar = sar[[2,3,0,1]]
            elif self.reorder_bands == 2:
                sar = sar[[1,3,0,2]]
            elif self.reorder_bands == 3:
                sar = sar[[0,3,1,2]]
            
            weights = np.ones_like(msk[:,:,:1], dtype=float)
            regionlabels, regioncount = measure.label(msk[:,:,0], background=0, connectivity=1, return_num=True)
            regionproperties = measure.regionprops(regionlabels)


            weights_512 = np.ones_like(msk_512[:,:,:1], dtype=float)
            regionlabels_512, regioncount_512 = measure.label(msk_512[:,:,0], background=0, connectivity=1, return_num=True)
            regionproperties_512 = measure.regionprops(regionlabels_512)
            
            for bl in range(regioncount):
                if regionproperties[bl].area < self.train_min_building_size:
                    msk[:,:,0][regionlabels == bl+1] = 0
                   
                    msk[:,:,1][regionlabels == bl+1] = 0
                    
                weights[regionlabels == bl+1] = 1024.0 / regionproperties[bl].area
            for bl in range(regioncount_512):
                if regionproperties_512[bl].area < self.train_min_building_size:
                    msk_512[:,:,0][regionlabels_512 == bl+1] = 0
                    msk_512[:,:,1][regionlabels_512 == bl+1] = 0
                weights_512[regionlabels_512 == bl+1] = 1024.0 / regionproperties_512[bl].area
            
            msk_512[:, :, :3] = (msk_512[:, :, :3] > 1) * 1
            msk[:, :, :3] = (msk[:, :, :3] > 1) * 1
            
            weights = torch.from_numpy(weights.transpose((2, 0, 1)).copy()).float()
            
            msk = torch.from_numpy(msk.transpose((2, 0, 1)).copy()).float()
            sar_512 = torch.from_numpy(sar_512.transpose((2, 0, 1)).copy()).float()
            
            msk_512 = torch.from_numpy(msk_512.transpose((2, 0, 1)).copy()).float()
            
        else:
            direction, strip, coord = parse_img_id(self.image_sar_paths[idx], self.orients)
            if direction.item():
                sar = np.fliplr(np.flipud(sar))
                msk = np.fliplr(np.flipud(msk))

            sar = torch.from_numpy(sar.transpose((2, 0, 1)).copy()).float()
            if self.reorder_bands == 1:
                sar = sar[[2,3,0,1]]
            elif self.reorder_bands == 2:
                sar = sar[[1,3,0,2]]
            elif self.reorder_bands == 3:
                sar = sar[[0,3,1,2]]
            sar_512 = sar
            weights = regioncount = torch.Tensor([0])
            msk=msk_512= torch.from_numpy(msk.transpose((2, 0, 1)).copy()).float()
            x0=y0=weights=regioncount=torch.Tensor([0])
            

        return {"mask": msk, "sar": sar, "sar_512": sar_512, "msk_512": msk_512, "sar_full": sar_full, "mask_full": msk_full, "x0" : x0, "y0" : y0, 'img_name': self.image_sar_paths[idx],
                'ymin': ymin, 'xmin': xmin, 'b_count': regioncount, 'weights': weights, 'direction' : direction}



#################################### EVAL

def test_postprocess(pred_folder, pred_csv, **kwargs):
    np.seterr(over = 'ignore')
    sourcefiles = sorted(glob.glob(os.path.join(pred_folder, '*')))  #"/*.csv")
    with Pool() as pool:
        proposals = [p for p in tqdm(pool.imap_unordered(partial(test_postprocess_single, **kwargs), sourcefiles), total = len(sourcefiles))]
    pd.concat(proposals).to_csv(pred_csv, index=False)

def test_postprocess_single(sourcefile, watershed_line=True, conn = 2, polygon_buffer = 0.5, tolerance = 0.5, seed_msk_th = 0.75, area_th_for_seed = 40, pred_th = 0.5, area_th = 40,    ## area_th=80, area_th_for_seed=110, seed_msk_th=0.75##
        contact_weight = 1.0, edge_weight = 0.0, seed_contact_weight = 1.0, seed_edge_weight = 1.0):
    mask = gdal.Open(sourcefile).ReadAsArray() # logits
    mask = 1.0 / (1 + np.exp(-mask))
    mask[0] = mask[0] * (1 - contact_weight * mask[2]) * (1 - edge_weight * mask[1])

    seed_msk = mask[0] * (1 - seed_contact_weight * mask[2]) * (1 - seed_edge_weight * mask[1])
    seed_msk = measure.label((seed_msk > seed_msk_th), connectivity=conn, background=0)
    props = measure.regionprops(seed_msk)
    for i in range(len(props)):
        if props[i].area < area_th_for_seed:
            seed_msk[seed_msk == i + 1] = 0
    seed_msk = measure.label(seed_msk, connectivity=conn, background=0)

    mask = skimage.segmentation.watershed(-mask[0], seed_msk, mask=(mask[0] > pred_th), watershed_line=watershed_line)
    mask = measure.label(mask, connectivity=conn, background=0).astype('uint8')

    polygon_generator = rasterio.features.shapes(mask, mask)
    polygons = []
    for polygon, value in polygon_generator:
        p = shape(polygon).buffer(polygon_buffer)
        if p.area >= area_th:
            p = dumps(p.simplify(tolerance=tolerance), rounding_precision=0)
            polygons.append(p)

    tilename = '_'.join(os.path.splitext(os.path.basename(sourcefile))[0].split('_')[-4:])
    csvaddition = pd.DataFrame({'ImageId': tilename, 'BuildingId': range(len(polygons)), 'PolygonWKT_Pix': polygons, 'Confidence': 1 })
    return csvaddition

def evaluation(pred_csv, gt_csv):
    evaluator = base.Evaluator(gt_csv)
    evaluator.load_proposal(pred_csv, proposalCSV=True, conf_field_list=[])
    report = evaluator.eval_iou_spacenet_csv(miniou=0.5, min_area=40)
    tp = 0
    fp = 0
    fn = 0
    for entry in report:
        tp += entry['TruePos']
        fp += entry['FalsePos']
        fn += entry['FalseNeg']
    f1score = (2*tp) / ((2*tp) + fp + fn)
    if(tp!=0):
        Precision = (tp) / (tp + fp)
        Recall = (tp) / (tp + fn)
    print('Validation F1 {} tp {} fp {} fn {}'.format(f1score, tp, fp, fn))   
    return f1score

############################### TRAIN
class FocalLoss2d(torch.nn.Module):
    def __init__(self, gamma=2, ignore_index=255, eps=1e-6):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, outputs, targets, weights = 1.0):
        outputs = torch.sigmoid(outputs)
        outputs = outputs.contiguous()
        targets = targets.contiguous()
        weights = weights.contiguous()

        non_ignored = targets.view(-1) != self.ignore_index
        targets = targets.view(-1)[non_ignored].float()
        outputs = outputs.contiguous().view(-1)[non_ignored]
        weights = weights.contiguous().view(-1)[non_ignored]

        outputs = torch.clamp(outputs, self.eps, 1. - self.eps)
        targets = torch.clamp(targets, self.eps, 1. - self.eps)

        pt = (1 - targets) * (1 - outputs) + targets * outputs
        return ((-(1. - pt) ** self.gamma * torch.log(pt)) * weights).mean()


class DiceLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True, per_image=False, eps = 1e-6):
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)
        self.per_image = per_image
        self.eps = eps

    def forward(self, outputs, targets):
        outputs = torch.sigmoid(outputs)
        batch_size = outputs.size()[0]
        if not self.per_image:
            batch_size = 1
        dice_target = targets.contiguous().view(batch_size, -1).float()
        dice_output = outputs.contiguous().view(batch_size, -1)
        intersection = torch.sum(dice_output * dice_target, dim=1)
        union = torch.sum(dice_output, dim=1) + torch.sum(dice_target, dim=1) + self.eps
        loss = (1 - (2 * intersection + self.eps) / union).mean()
        return loss

def load_state_dict(model, state_dict):
    missing_keys = [] 
    unexpected_keys = []
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, [])
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')
    load(model)
    print('Unexpected key(s) in state_dict: {} '.format(', '.join('"{}"'.format(k) for k in unexpected_keys)))
    print('Missing key(s) in state_dict: {} '.format(', '.join('"{}"'.format(k) for k in missing_keys)))

rot_in_path = '/home/user/Perception/Buildingsegmentation/Sumanth/Spacenet-codes/SAR_orientations.txt'
rot_out_path = '/home/user/Perception/Buildingsegmentation/Sumanth/Spacenet-codes/SAR_orientations.csv'
models_folder = '/home/user/Perception/Buildingsegmentation/Sumanth/Spacenet-codes/wdata_pngsave/weights'
cutmix_rgb_folder = '/home/user/Perception/SN6_dataset/train_set/AOI_11_Rotterdam/c1_RGB'
cutmix_mas_folder = '/home/user/Perception/SN6_dataset/train_set/AOI_11_Rotterdam/c1_mas'




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SpaceNet 6 Baseline Algorithm')
    parser.add_argument('--split_folds', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--merge', action='store_true')
    
    parser.add_argument('--masks_csv', default='/home/user/Perception/SN6_dataset/val_set/AOI_11_Rotterdam/val_masks_csv', type=str)
    parser.add_argument('--pred_csv', default='./wdata_pngsave/pred_fold_{0}_csv', type=str)
    parser.add_argument('--pred_folder', default='./wdata_pngsave/pred_fold_{0}_0', type=str)
    parser.add_argument('--val_pngsfolder', default='./wdata_pngsave/val_pngs', type=str)
    parser.add_argument('--snapshot_last', default='snapshot_fold_{0}_last', type=str)
    parser.add_argument('--snapshot_best', default='snapshot_fold_{0}_best', type=str)
    
    parser.add_argument('--edge_width', default=3, type=int)
    parser.add_argument('--contact_width', default=9, type=int)
    parser.add_argument('--train_min_building_size', default=0, type=int)

    parser.add_argument('--start_val_epoch', default=0, type=int)

    parser.add_argument('--num_workers', default=4, type=int)                     

    parser.add_argument('--batch_size', default=8, type=int)                      
    parser.add_argument('--crop_size', default=512, type=int)                     ##512##
    parser.add_argument('--lr', default=2e-4, type=float)                        
    parser.add_argument('--warm_up_lr_scale', default=1.0, type=float)
    parser.add_argument('--warm_up_lr_epochs', default=0, type=int)
    parser.add_argument('--warm_up_dec_epochs', default=0, type=int)
    parser.add_argument('--wd', default=1e-2, type=float)
    parser.add_argument('--gamma', default=0.5, type=float)                   ###0.5###
    parser.add_argument('--pos_weight', default=0.5, type=float)
    parser.add_argument('--b_count_weight', default=0.5, type=float)
    parser.add_argument('--b_count_div', default=8, type=float)
    parser.add_argument('--b_rev_size_weight', default=0.0, type=float)
    parser.add_argument('--focal_weight', default=1.0, type=float)             ## 1.0 ##
    parser.add_argument('--edge_weight', default=0.25, type=float)
    parser.add_argument('--contact_weight', default=0.1, type=float)
    parser.add_argument('--height_scale', default=0.0, type=float)
    parser.add_argument('--rgb_weight', default=0.0, type=float)
    parser.add_argument('--loss_eps', default=1e-6, type=float)
    parser.add_argument('--clip_grad_norm_value', default=1.2, type=float)
    parser.add_argument('--focal_gamma', default=2.0, type=float)
    parser.add_argument('--rot_prob', default=0.7, type=float)              ##0.7##
    parser.add_argument('--scale_prob', default=1.0, type=float)            
    parser.add_argument('--color_aug_prob', default=0.0, type=float)        
    parser.add_argument('--fliplr_prob', default=0.5, type=float)         ##0.5##
    parser.add_argument('--input_scale', default=1.0, type=float)
    parser.add_argument('--strip_scale', default=1.0, type=float)
    parser.add_argument('--direction_scale', default=1.0, type=float)
    parser.add_argument('--coord_scale', default=1.0, type=float)
    parser.add_argument('--cutmix_epoch', default=200, type=float)#after the mentioned number cutmix images will be generated and saved
    parser.add_argument('--total_epochs', default=10, type=float)
    parser.add_argument('--sar', default=False, type=bool)#set to true if training or validating with SAR


    args = parser.parse_args(sys.argv[1:])
    if not (args.train or args.val or args.test):
        sys.exit(0)

    ############# TRAINING
    print("In training")
    if args.train:
        for f in glob.glob("/home/user/Perception/SN6_dataset/train_set/AOI_11_Rotterdam/c1_RGB/*_cutmix_aug_*"):
            os.remove(f)
        for f in glob.glob("/home/user/Perception/SN6_dataset/train_set/AOI_11_Rotterdam/c1_mas/*_cutmix_aug_*"):
            os.remove(f)
    #change below training paths if required
    train_sar_img_files = sorted([f for f in glob.glob(os.path.join('/home/user/Perception/SN6_dataset/train_set/AOI_11_Rotterdam/SAR-3ch/*.tif'))])
    

    
    train_rgb_img_files = sorted([f for f in glob.glob(os.path.join('/home/user/Perception/SN6_dataset/train_set/AOI_11_Rotterdam/PS-RGB/*.tif'))])
    
    train_label_files = sorted([f for f in glob.glob(os.path.join('/home/user/Perception/SN6_dataset/train_set/AOI_11_Rotterdam/masks/*.tif'))])
    

    train_label_index_files = sorted([f for f in glob.glob(os.path.join('/home/user/Perception/SN6_dataset/train_set/AOI_11_Rotterdam/label_index_masks/*.tif'))])
    
    
    if args.val: #change the validation paths if required
        
        val_sar_img_files = sorted([f for f in glob.glob(os.path.join('/home/user/Perception/SN6_dataset/val_set/AOI_11_Rotterdam/SAR-Intensity/*.tif'))])
        val_rgb_img_files = sorted([f for f in glob.glob(os.path.join('/home/user/Perception/SN6_dataset/val_set/AOI_11_Rotterdam/PS-RGB/*.tif'))])
        val_label_files = sorted([f for f in glob.glob(os.path.join('/home/user/Perception/SN6_dataset/val_set/AOI_11_Rotterdam/masks/*.tif'))])

    makedirs(models_folder, exist_ok=True)
    
    val_data_loader = DataLoader(MyData(val_sar_img_files, val_rgb_img_files, val_label_files, train=False, test=False), batch_size=1, num_workers=args.num_workers, pin_memory=True, shuffle=True)
    device = torch.device("cuda")


    model = Simple_Unet().cuda() #Change model definition to Simple_Unet or DeepMAO as required
    ###############
    if torch.cuda.device_count() > 1: #will enter this condition if more than 1 CUDA_VISIBLE_DEVICES is defined in train.sh file 
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model,device_ids=[0]).cuda() #device ids must always start from 0
    model.to(device)

    retrain = False
    if not args.train or retrain:
        loaded = torch.load(path.join(models_folder, args.snapshot_best))
        print("loaded checkpoint '{}' (epoch {}, f1 score {})".format(args.snapshot_best, loaded['epoch'], loaded['best_score']))
        load_state_dict(model, loaded['state_dict'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))
    def lr_comp(epoch):
        if epoch < args.warm_up_lr_epochs:
            return args.warm_up_lr_scale
        elif epoch < 60:
            return 1.0
        elif epoch < 80:
            return 0.33
        elif epoch < 90:
            return 0.1
    
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[80,100,120], gamma=args.gamma)
    dice_loss = DiceLoss(eps=args.loss_eps).cuda()
    focal_loss = FocalLoss2d(gamma = args.focal_gamma, eps=args.loss_eps).cuda()
    criterion = nn.L1Loss()
 
    q = Queue()
    best_f1 = -1.0
    for epoch in range(args.total_epochs if args.train else 1):
        
        if args.train:
            time2 = time.time()
            if(epoch<=args.cutmix_epoch):
                data_train = MyData(train_sar_img_files, train_rgb_img_files, train_label_files, train=True, test=False, crop_size=args.crop_size, 
                        rot_prob = args.rot_prob, scale_prob = args.scale_prob, color_aug_prob = args.color_aug_prob, fliplr_prob = args.fliplr_prob, train_min_building_size = args.train_min_building_size, normalize=False)
                train_data_loader = DataLoader(data_train, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)
                iterator = tqdm(train_data_loader)
                model.train()
                torch.cuda.empty_cache()

                for sample in iterator:
                    time1 = time.time()
                    load = time1 - time2

                    
                    rgb = sample["rgb"].cuda(non_blocking=True).to('cuda:0')#change the value to sample['rgb'] or sample['sar'] as required(make sure to change to the right MyData class as well)
                    target = sample["mask"].cuda(non_blocking=True).to('cuda:0')

                    b_count = sample["b_count"].cuda(non_blocking=True) / args.b_count_div
                    b_weights = b_count * args.b_count_weight + 1.0 * (1.0 - args.b_count_weight)
                    b_rev_size_weights = sample["weights"].cuda(non_blocking=True)
                    b_rev_size_weights = b_rev_size_weights * args.b_rev_size_weight + 1.0 * (1.0 - args.b_rev_size_weight)

                    weights = torch.ones(size=target.shape).cuda()
                    weights[target > 0.0] *= args.pos_weight
                    weights[:, :1] *= b_rev_size_weights
                    weights[:, 1:2] *= b_rev_size_weights
                    for i in range(weights.shape[0]):
                        weights[i] = weights[i] * b_weights[i]
                
                    outputs = model(rgb)
                    if(epoch==args.cutmix_epoch):
                        tile_h, tile_w = outputs.size(2)//4, outputs.size(3)//4

                        tiled_rgb_data_list = []
                        tiled_output_data_list = []
                        tiled_label_data_list = []   
                        for i in range(rgb.shape[0]):
                            
                            tiled_label_data_full= [target[i,:,x:x+tile_h,y:y+tile_w] for x in range(0,outputs.size(2),tile_h) for y in range(0,outputs.size(3),tile_w)]

                            only_buildings_indices = [i for i in range(len(tiled_label_data_full)) if torch.sum(tiled_label_data_full[i])>0]
                    

                            
                            tiled_rgb_data_full = [rgb[i,:,x:x+tile_h,y:y+tile_w] for x in range(0,rgb.size(2),tile_h) for y in range(0,rgb.size(3),tile_w)]
                            tiled_output_data_full = [outputs[i,:,x:x+tile_h,y:y+tile_w] for x in range(0,outputs.size(2),tile_h) for y in range(0,outputs.size(3),tile_w)]
                            

                            tiled_label_data = [tiled_label_data_full[i] for i in only_buildings_indices]
                            tiled_rgb_data = [tiled_rgb_data_full[i] for i in only_buildings_indices]
                            tiled_output_data = [tiled_output_data_full[i] for i in only_buildings_indices]
                           
                            if len(only_buildings_indices)>0:
                                tiled_label_data_list.append(tiled_label_data)
                            
                                
                                tiled_rgb_data_list.append(tiled_rgb_data)
                                
                                tiled_output_data_list.append(tiled_output_data)
                            else:
                                tiled_label_data_list.append(tiled_label_data_full)
                            
                                
                                tiled_rgb_data_list.append(tiled_rgb_data_full)
                                
                                tiled_output_data_list.append(tiled_output_data_full)
                        
                        tiled_losses_out = []
                        cutmix_patch = []
                        cutmix_label_patch = []
                        for i in range(len(tiled_output_data_list)):
                            tiled_losses = []
                            for j in range(len(tiled_output_data_list[i])):
                                tiled_losses.append(criterion(torch.relu(tiled_output_data_list[i][j]), tiled_label_data_list[i][j]))

                            maxloss_tile_index = tiled_losses.index(max(tiled_losses))
                            
                            cutmix_patch.append(tiled_rgb_data_list[i][maxloss_tile_index])
                            
                            cutmix_label_patch.append(tiled_label_data_list[i][maxloss_tile_index])


                        for k in range(len(cutmix_patch)):
                            rgb_512 = sample["rgb_512"][k]
                            target_512 = sample["msk_512"][k]

                            x01 = random.randint(0, rgb[k].shape[1] - cutmix_patch[k].shape[1])
                            y01 = random.randint(0, rgb[k].shape[1] - cutmix_patch[k].shape[1])
                            rgb_aug, target_aug = cutmix_aug(cutmix_patch[k], cutmix_label_patch[k], rgb_512, target_512, x01, y01)
                            rgb_aug = rgb_aug.cpu().detach().numpy()
                            rgb_aug = np.moveaxis(rgb_aug,0,2)
                            
                            rgb_full = sample["rgb_full"][k]
                            rgb_full = rgb_full.cpu().detach().numpy()
                            
                            target_aug = target_aug.cpu().detach().numpy()
                            target_aug = np.moveaxis(target_aug,0,2)
                            target_full = sample["mask_full"][k]
                            target_full = target_full.cpu().detach().numpy()

                            x0 = sample["x0"][k].cpu().numpy()
                            y0 = sample["y0"][k].cpu().numpy()
                            ymin = sample["ymin"][k].cpu().numpy()
                            xmin = sample["xmin"][k].cpu().numpy()
                        
                            try:
                                rgb_full[y0+ymin:y0+ymin+args.crop_size, x0:x0+args.crop_size] = rgb_aug
                                
                                target_full[y0+ymin:y0+ymin+args.crop_size, x0:x0+args.crop_size] = target_aug*255
                            except:
                                y0 = y0-((y0+ymin+args.crop_size)-900)

                                rgb_full[y0+ymin:y0+ymin+args.crop_size, x0:x0+args.crop_size] = rgb_aug
                                
                                target_full[y0+ymin:y0+ymin+args.crop_size, x0:x0+args.crop_size] = target_aug*255

                            skimage.io.imsave(os.path.join('/home/user/Perception/SN6_dataset/train_set/AOI_11_Rotterdam/c1_RGB/', (sample['img_name'][k][68:-4])+'_cutmix_aug_'+str(args.cutmix_epoch)+'.tif'), rgb_full)
                            target_full_name = os.path.join('/home/user/Perception/SN6_dataset/train_set/AOI_11_Rotterdam/c1_mas', os.path.basename(sample["img_name"][k]).replace('PS-RGB', 'SAR-Intensity'))
                            skimage.io.imsave(os.path.join('/home/user/Perception/SN6_dataset/train_set/AOI_11_Rotterdam/c1_mas/', (target_full_name[68:-4])+'_cutmix_aug_'+str(args.cutmix_epoch)+'.tif'), target_full)
                    
                    l0 = args.focal_weight * focal_loss(outputs[:, 0], target[:, 0], weights[:, 0]) + dice_loss(outputs[:, 0], target[:, 0])
                    l1 = args.edge_weight * (args.focal_weight * focal_loss(outputs[:, 1], target[:, 1], weights[:, 1]) + dice_loss(outputs[:, 1], target[:, 1]))
                    l2 = args.contact_weight * (args.focal_weight * focal_loss(outputs[:, 2], target[:, 2], weights[:, 2]) + dice_loss(outputs[:, 2], target[:, 2]))
                    loss = l0+l1+l2
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm_value)
                    optimizer.step()
                    time2 = time.time()
                    proc = time2 - time1
                    iterator.set_description("epoch: {}; lr {:.7f}; Loss {:.4f} l0 {:.4f}; l1 {:.4f};l2 {:.4f};load time {:.3f} proc time {:.3f}".format(
                        epoch, scheduler.get_lr()[-1], loss, l0, l1, l2, load, proc))

                scheduler.step()
                torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, path.join(models_folder, args.snapshot_last))
                torch.cuda.empty_cache()
            else:
                
                train_rgb_img_files_for_cutmix = sorted([f for f in glob.glob(os.path.join('/home/user/Perception/SN6_dataset/train_set/AOI_11_Rotterdam/c1_RGB/*.tif'))])
                train_label_files_for_cutmix = sorted([f for f in glob.glob(os.path.join('/home/user/Perception/SN6_dataset/train_set/AOI_11_Rotterdam/c1_mas/*.tif'))])
               
                train_sar_img_files_for_cutmix = sorted([f for f in glob.glob(os.path.join('/home/user/Perception/SN6_dataset/train_set/AOI_11_Rotterdam/SAR-3ch_for_cutmix/*.tif'))])

                data_train_2 = MyData(train_sar_img_files_for_cutmix, train_rgb_img_files_for_cutmix, train_label_files_for_cutmix, train_label_files_for_cutmix, train=True, test=False, crop_size=args.crop_size, 
                            rot_prob = args.rot_prob, scale_prob = args.scale_prob, color_aug_prob = args.color_aug_prob, fliplr_prob = args.fliplr_prob, train_min_building_size = args.train_min_building_size, normalize=False)
                train_data_loader_2 = DataLoader(data_train_2, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)
                
                iterator_2 = tqdm(train_data_loader_2)
                model.train()
                torch.cuda.empty_cache()

                for sample in iterator_2:
                    time1 = time.time()
                    load = time1 - time2

                    #sar = sample["sar"].cuda(non_blocking=True).to('cuda:0')
                    rgb = sample["rgb"].cuda(non_blocking=True).to('cuda:0')#change the value to sample['rgb'] or sample['sar'] as required(make sure to change to the right MyData class as well)
                    target = sample["mask"].cuda(non_blocking=True).to('cuda:0')

                    b_count = sample["b_count"].cuda(non_blocking=True) / args.b_count_div
                    b_weights = b_count * args.b_count_weight + 1.0 * (1.0 - args.b_count_weight)
                    b_rev_size_weights = sample["weights"].cuda(non_blocking=True)
                    b_rev_size_weights = b_rev_size_weights * args.b_rev_size_weight + 1.0 * (1.0 - args.b_rev_size_weight)

                    weights = torch.ones(size=target.shape).cuda()
                    weights[target > 0.0] *= args.pos_weight
                    weights[:, :1] *= b_rev_size_weights
                    weights[:, 1:2] *= b_rev_size_weights
                    for i in range(weights.shape[0]):
                        weights[i] = weights[i] * b_weights[i]
                
                    outputs = model(rgb)

                    tile_h, tile_w = outputs.size(2)//4, outputs.size(3)//4

                    tiled_rgb_data_list = []
                    tiled_output_data_list = []
                    tiled_label_data_list = []   
                    for i in range(rgb.shape[0]):
                            tiled_label_data_full= [target[i,:,x:x+tile_h,y:y+tile_w] for x in range(0,outputs.size(2),tile_h) for y in range(0,outputs.size(3),tile_w)]

                            only_buildings_indices = [i for i in range(len(tiled_label_data_full)) if torch.sum(tiled_label_data_full[i])>0]

                            tiled_rgb_data_full = [rgb[i,:,x:x+tile_h,y:y+tile_w] for x in range(0,rgb.size(2),tile_h) for y in range(0,rgb.size(3),tile_w)]
                            tiled_output_data_full = [outputs[i,:,x:x+tile_h,y:y+tile_w] for x in range(0,outputs.size(2),tile_h) for y in range(0,outputs.size(3),tile_w)]
                            

                            tiled_label_data = [tiled_label_data_full[i] for i in only_buildings_indices]
                            tiled_rgb_data = [tiled_rgb_data_full[i] for i in only_buildings_indices]
                            tiled_output_data = [tiled_output_data_full[i] for i in only_buildings_indices]
                           
                            if len(only_buildings_indices)>0:
                                tiled_label_data_list.append(tiled_label_data)
                               
                                
                                tiled_rgb_data_list.append(tiled_rgb_data)
                                
                                tiled_output_data_list.append(tiled_output_data)
                            else:
                                tiled_label_data_list.append(tiled_label_data_full)
                                
                                
                                tiled_rgb_data_list.append(tiled_rgb_data_full)
                                
                                tiled_output_data_list.append(tiled_output_data_full)
                        
                    
                    tiled_losses_out = []
                    cutmix_patch = []
                    cutmix_label_patch = []
                    for i in range(len(tiled_output_data_list)):
                        tiled_losses = []
                        for j in range(len(tiled_output_data_list[i])):
                            tiled_losses.append(criterion(torch.relu(tiled_output_data_list[i][j]), tiled_label_data_list[i][j]))
                

                        maxloss_tile_index = tiled_losses.index(max(tiled_losses))
                        
                        cutmix_patch.append(tiled_rgb_data_list[i][maxloss_tile_index])
                        
                        cutmix_label_patch.append(tiled_label_data_list[i][maxloss_tile_index])

                    for k in range(len(cutmix_patch)):
                        rgb_512 = sample["rgb_512"][k]
                        target_512 = sample["msk_512"][k]

                        x01 = random.randint(0, rgb[k].shape[1] - cutmix_patch[k].shape[1])
                        y01 = random.randint(0, rgb[k].shape[1] - cutmix_patch[k].shape[1])
                        rgb_aug, target_aug = cutmix_aug(cutmix_patch[k], cutmix_label_patch[k], rgb_512, target_512, x01, y01)
                        rgb_aug = rgb_aug.cpu().detach().numpy()
                        rgb_aug = np.moveaxis(rgb_aug,0,2)
                        
                        rgb_full = sample["rgb_full"][k]
                        rgb_full = rgb_full.cpu().detach().numpy()
                        
                        target_aug = target_aug.cpu().detach().numpy()
                        target_aug = np.moveaxis(target_aug,0,2)
                        target_full = sample["mask_full"][k]
                        target_full = target_full.cpu().detach().numpy()

                        x0 = sample["x0"][k].cpu().numpy()
                        y0 = sample["y0"][k].cpu().numpy()
                        ymin = sample["ymin"][k].cpu().numpy()
                        xmin = sample["xmin"][k].cpu().numpy()
                    
                        try:
                            rgb_full[y0+ymin:y0+ymin+args.crop_size, x0:x0+args.crop_size] = rgb_aug
                            
                            target_full[y0+ymin:y0+ymin+args.crop_size, x0:x0+args.crop_size] = target_aug*255
                        except:
                            y0 = y0-((y0+ymin+args.crop_size)-900)

                            rgb_full[y0+ymin:y0+ymin+args.crop_size, x0:x0+args.crop_size] = rgb_aug
                            
                            target_full[y0+ymin:y0+ymin+args.crop_size, x0:x0+args.crop_size] = target_aug*255

                        if('_cutmix_aug_' not in sample["img_name"][k]):
                        
                            skimage.io.imsave(os.path.join('/home/user/Perception/SN6_dataset/train_set/AOI_11_Rotterdam/c1_RGB/', (sample['img_name'][k][68:-4])+'_cutmix_aug_'+str(epoch)+'.tif'), rgb_full)
                            target_full_name = os.path.join('/home/user/Perception/SN6_dataset/train_set/AOI_11_Rotterdam/c1_mas', os.path.basename(sample["img_name"][k]).replace('PS-RGB', 'SAR-Intensity'))
                            skimage.io.imsave(os.path.join('/home/user/Perception/SN6_dataset/train_set/AOI_11_Rotterdam/c1_mas/', (target_full_name[68:-4])+'_cutmix_aug_'+str(epoch)+'.tif'), target_full)
                        else:
                            pass
                    l0 = args.focal_weight * focal_loss(outputs[:, 0], target[:, 0], weights[:, 0]) + dice_loss(outputs[:, 0], target[:, 0])
                    l1 = args.edge_weight * (args.focal_weight * focal_loss(outputs[:, 1], target[:, 1], weights[:, 1]) + dice_loss(outputs[:, 1], target[:, 1]))
                    l2 = args.contact_weight * (args.focal_weight * focal_loss(outputs[:, 2], target[:, 2], weights[:, 2]) + dice_loss(outputs[:, 2], target[:, 2]))
                    
                    
                    loss = l0+l1+l2
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm_value)
                    optimizer.step()
                    time2 = time.time()
                    proc = time2 - time1
                    iterator_2.set_description("epoch: {}; lr {:.7f}; Loss {:.4f} l0 {:.4f}; l1 {:.4f};l2 {:.4f};load time {:.3f} proc time {:.3f}".format(
                        epoch, scheduler.get_lr()[-1], loss, l0, l1, l2, load, proc))

                scheduler.step()
                torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, path.join(models_folder, args.snapshot_last))
                torch.cuda.empty_cache()
            
                for f in glob.glob("/home/user/Perception/SN6_dataset/train_set/AOI_11_Rotterdam/c1_RGB/*_cutmix_aug_"+str(epoch-1)+".tif"):
                    os.remove(f)
                for f in glob.glob("/home/user/Perception/SN6_dataset/train_set/AOI_11_Rotterdam/c1_mas/*_cutmix_aug_"+str(epoch-1)+".tif"):
                    os.remove(f)

        if args.val and epoch > args.start_val_epoch: #comment this block and change below if function 'and' functions when running only validation/inference
            
            t.join()
            best_f1 = max(best_f1, q.get())

        if (args.val and epoch >= args.start_val_epoch) or args.test:
            
            print('Validation starts')

            shutil.rmtree(args.pred_folder, ignore_errors=True)
            makedirs(args.pred_folder, exist_ok=True)
            model.eval()
            torch.cuda.empty_cache()
            with torch.no_grad():
                for _, sample in enumerate(tqdm(val_data_loader)):
                    
                    rgb = sample["rgb"].cuda(non_blocking=True)#change the value to sample['rgb'] or sample['sar'] as required(make sure to change to the right MyData class as well)
                    #print(sample['img_name'])
                    #print(hey)
                    ymin, xmin = sample['ymin'].item(), sample['xmin'].item()
                    direction = sample["direction"].cuda(non_blocking=True)
                    _, _, h, w = rgb.shape
                    scales = [1.5]
                    oos = torch.zeros((rgb.shape[0], 3, rgb.shape[2], rgb.shape[3])).cuda()
                    for sc in scales:
                        rgb = F.interpolate(rgb, size=(ceil(h*sc/32)*32, ceil(w*sc/32)*32), mode = 'bilinear', align_corners=True)
                        o = model(rgb)
                        oos += F.interpolate(o, size=(h,w), mode = 'bilinear', align_corners=True)

                    o = np.moveaxis(oos.cpu().data.numpy(), 1, 3)

                    
                    
                    for i in range(len(o)):
                        
                        img = o[i][:,:,:3]
                        if(args.sar):
                            if direction[i].item():
                                img = np.fliplr(np.flipud(img))
                        img = cv2.copyMakeBorder(img, ymin, 900 - h - ymin, xmin, 900 - w - xmin, cv2.BORDER_CONSTANT, 0.0)

                        skimage.io.imsave(os.path.join(args.pred_folder, os.path.split(sample['img_name'][i])[1]), img)
                        

            torch.cuda.empty_cache()

        if (args.val and epoch >= args.start_val_epoch) or args.test:
           
            to_save = {k: copy.deepcopy(v.cpu()) for k, v in model.state_dict().items()}
               
            def new_thread():
                
                test_postprocess(args.pred_folder, args.pred_csv)

                calculate_acc_iou(args.pred_folder) #comment this line if you want validation to be faster
                
                val_f1 = evaluation(args.pred_csv,args.masks_csv)              
                
                print()
                print('    Validation F1 score at epoch {}: {:.5f}, best {}'.format(epoch, val_f1, max(val_f1, best_f1)))
                print()
                if best_f1 < val_f1 and args.train:
                    torch.save({'epoch': epoch, 'state_dict': to_save, 'best_score': val_f1}, path.join(models_folder, args.snapshot_best))
                q.put(val_f1)
            t = Process(target = new_thread)
            t.start()

    if args.val:
        t.join()
