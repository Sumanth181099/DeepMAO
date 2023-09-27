import gdal
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import shutil
import skimage
from skimage import io
from PIL import Image
import torch.nn.functional as F
import random
import math

seed =99
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

##########TGRS DISOPTNET SCHEDULER##############

class LR_Scheduler(object):
    """Learning Rate Scheduler
    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``
    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``
    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``
    Args:
        args:  :attr:`args.lr_scheduler` lr scheduler mode (`cos`, `poly`),
          :attr:`args.lr` base learning rate, :attr:`args.epochs` number of epochs,
          :attr:`args.lr_step`
        iters_per_epoch: number of iterations per epoch
    """
    def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0,
                 lr_step=0, warmup_epochs=0):
        self.mode = mode
        print('Using {} LR Scheduler!'.format(self.mode))
        self.lr = base_lr
        if mode == 'step':
            assert lr_step
        self.lr_step = lr_step
        self.iters_per_epoch = iters_per_epoch
        self.N = num_epochs * iters_per_epoch
        self.epoch = -1
        self.warmup_iters = warmup_epochs * iters_per_epoch

    def __call__(self, optimizer, i, epoch):
        T = epoch * self.iters_per_epoch + i
        if self.mode == 'cos':
            lr = 0.5 * self.lr * (1 + math.cos(1.0 * T / self.N * math.pi))
        elif self.mode == 'poly':
            lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)
        elif self.mode == 'step':
            lr = self.lr * (0.1 ** (epoch // self.lr_step))
        else:
            raise NotImplemented
        # warm up lr schedule
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = lr * 1.0 * T / self.warmup_iters
        if epoch > self.epoch:
            print('\n=>Epoches %i, learning rate = %.7f' % (epoch, lr))
            self.epoch = epoch
        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            # enlarge the lr at the head
            for i in range(len(optimizer.param_groups)):
                if optimizer.param_groups[i]['lr'] > 0: optimizer.param_groups[i]['lr'] = lr
            # optimizer.param_groups[0]['lr'] = lr
            # for i in range(1, len(optimizer.param_groups)):
            #     optimizer.param_groups[i]['lr'] = lr * 10

#########################################

def plot(imgs):
    #fig = plt.figure(figsize=(25,25))
    #rows = 2
    #columns = 8
    fig = plt.figure(figsize=(25, 25))
    rows = 1
    columns = 1
    for i in range(len(imgs)):
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(imgs[i])
        #print(imgs[i])
        plt.axis('off')
        #plt.subplots_adjust(wspace=0.05, hspace=-0.3)
    plt.tight_layout()
    plt.savefig('label_index.jpg')

def span_image(img):
    img_sp = img[:,:,0]**2 + 2*abs(img[:,:,1]) + img[:,:,3]**2
    return img_sp
def generate_span():
    sar_imgs = sorted([f for f in glob.glob(os.path.join('/home/airl-gpu4/Aniruddh/SN6_dataset/val_10percent/AOI_11_Rotterdam/SAR-Intensity/*.tif'))])
    for i in range(len(sar_imgs)):
        tilename = '_'.join(os.path.splitext(os.path.basename(sar_imgs[i]))[0].split('/')[-4:])
        print(i)
        sarimg = gdal.Open(sar_imgs[i])
        sarimg = sarimg.ReadAsArray()
        sarimg = np.swapaxes(sarimg,0,2)
        sarimg = np.swapaxes(sarimg,0,1)
        sar_single_chnl = span_image(sarimg)
        #with open(f'/home/airl-gpu4/Aniruddh/SN6_dataset/train_10percent/AOI_11_Rotterdam/SAR_span/{tilename}.npy', 'wb') as f:
        #    np.save(f, sar_single_chnl)
        plt.imshow(sar_single_chnl, cmap='gray')
        plt.savefig(f'/home/airl-gpu4/Aniruddh/SN6_dataset/val_10percent/AOI_11_Rotterdam/Span_imgs/{tilename}.tif')

#generate_span()

def denoised_imgs(savepath):
    path = '/home/airl-gpu4/Aniruddh/test'
    for file in os.listdir(path):
            if file.endswith(".tif"):
                shutil.copy(f'/{path}/{file}',savepath)
#denoised_imgs('/home/airl-gpu4/Aniruddh/SN6_dataset/val_10percent/AOI_11_Rotterdam/despeckled_SAR/')
def rename_imgs(path):
    imgs = sorted([f for f in glob.glob(os.path.join('/home/wirin/Aniruddh/SN6_dataset/val_10percent/AOI_11_Rotterdam/despeckled_SAR/*.tif'))])
    for i in range(len(imgs)):
        tilename = '_'.join(os.path.splitext(os.path.basename(imgs[i]))[0].split('/')[-4:])
        old_file = os.path.join(path,f'{tilename}.tif')
        tilename = ''.join(os.path.splitext(os.path.basename(imgs[i]))[0].split('denoised_')[-4:])
        new_file = os.path.join(path,f'{tilename}.tif')
        os.rename(old_file, new_file)
#rename_imgs('/home/airl-gpu4/Aniruddh/SN6_dataset/val_10percent/AOI_11_Rotterdam/despeckled_SAR/')
def convert_imgs(path):
    imgs = sorted([f for f in glob.glob(os.path.join('/home/airl-gpu4/Aniruddh/SN6_dataset/val_10percent/AOI_11_Rotterdam/despeckled_SAR/*.tif'))])
    for i in range(len(imgs)):
        tilename = '_'.join(os.path.splitext(os.path.basename(imgs[i]))[0].split('/')[-4:])
        old_file = os.path.join(path,f'{tilename}.tif')
        tilename = '_'.join(os.path.splitext(os.path.basename(imgs[i]))[0].split('/')[-4:])
        new_file = os.path.join(path,f'{tilename}.png')
        os.rename(old_file, new_file)
#convert_imgs('/home/airl-gpu4/Aniruddh/SN6_dataset/val_10percent/AOI_11_Rotterdam/despeckled_SAR')
def tiling_imgs(data_path, mask_path, tiled_data_path, tiled_mask_path):
    rgb_imgs = sorted([f for f in glob.glob(os.path.join(data_path))])
    masks_imgs = sorted([f for f in glob.glob(os.path.join(mask_path))])
    for i in range(len(rgb_imgs)):
        tilename = '_'.join(os.path.splitext(os.path.basename(rgb_imgs[i]))[0].split('_')[-10:])

        imgs = gdal.Open(rgb_imgs[i])
        imgs = imgs.ReadAsArray()
        imgs = np.swapaxes(imgs,0,2)
        imgs = np.swapaxes(imgs,0,1)
        tile_h, tile_w = imgs.shape[0]//4, imgs.shape[1]//4

        tiled_data = [imgs[x:x+tile_h,y:y+tile_w] for x in range(0,imgs.shape[0],tile_h) for y in range(0,imgs.shape[1],tile_w)]

        '''imgs = gdal.Open(masks_imgs[i])
        imgs = imgs.ReadAsArray()
        imgs = np.swapaxes(imgs,0,2)
        imgs = np.swapaxes(imgs,0,1)
        tile_h, tile_w = imgs.shape[0]//4, imgs.shape[1]//4
        
        tiled_masks = [imgs[x:x+tile_h,y:y+tile_w] for x in range(0,imgs.shape[0],tile_h) for y in range(0,imgs.shape[1],tile_w)]'''

        for i in range(len(tiled_data)):
            img_name = tilename+'_'+f'{i}.tif'
            tiled_data_file = os.path.join(tiled_data_path,img_name)
            skimage.io.imsave(tiled_data_file,arr=tiled_data[i])
            #tiled_mask_file = os.path.join(tiled_mask_path,img_name)
            #skimage.io.imsave(tiled_mask_file,arr=tiled_masks[i])

#data_path = '/home/wirin/Aniruddh/SN6_dataset/train_10percent/AOI_11_Rotterdam/PS-RGB/*.tif'
#mask_path = '/home/wirin/Aniruddh/SN6_dataset/train_10percent/AOI_11_Rotterdam/Small_buildings_masks/*.tif'
#tiled_data_path = '/home/wirin/Aniruddh/SN6_dataset/train_10percent/AOI_11_Rotterdam/variations/Tiled_RGB'
#tiled_mask_path = '/home/wirin/Aniruddh/SN6_dataset/train_10percent/AOI_11_Rotterdam/variations/Tiled_Small_masks'

#tiling_imgs(data_path,mask_path,tiled_data_path,tiled_mask_path)

'''Feats.append(features['feats'].cpu().numpy())
dist1 = features['feats'][:,18,:,:]
#print(dist1.flatten().shape)
plt.hist(dist1.flatten().cpu().numpy(),bins=500)
plt.savefig('18th_chnl_rgb.png')
print(hey)'''

'''class MyData(Dataset):
    def __init__(self, image_sar_paths, image_rgb_paths, label_paths, train, test, crop_size = None, reorder_bands = 0,
        rot_prob = 0.3, scale_prob = 0.5, color_aug_prob = 0.0, gauss_aug_prob = 0.0, flipud_prob=0.0, fliplr_prob = 0.0, rot90_prob = 0.0, gamma_aug_prob = 0.0, elastic_aug_prob = 0.0, 
            channel_swap_prob = 0.0, train_min_building_size=0,normalize=False):
        super().__init__()
        self.image_sar_paths = image_sar_paths
        self.image_rgb_paths = image_rgb_paths
        self.label_paths = label_paths
        self.train = train
        self.test = test
        self.crop_size = crop_size
        self.reorder_bands = reorder_bands
        self.rot_prob = rot_prob
        self.scale_prob = scale_prob
        self.color_aug_prob = color_aug_prob
        self.gamma_aug_prob = gamma_aug_prob
        self.gauss_aug_prob = gauss_aug_prob
        self.elastic_aug_prob = elastic_aug_prob
        self.flipud_prob = flipud_prob
        self.fliplr_prob = fliplr_prob
        self.rot90_prob = rot90_prob
        self.channel_swap_prob = channel_swap_prob
        self.train_min_building_size = train_min_building_size
        self.normalize = normalize
        #self.elastic = iaa.ElasticTransformation(alpha=(0.25, 1.2), sigma=0.2)
        self.orients = pd.read_csv(rot_out_path, index_col = 0)
        #print(self.orients)
        #print(hello)
        self.orients['val'] = list(range(len(self.orients.index)))

    def __len__(self):
        return len(self.image_rgb_paths)

    def __getitem__(self,idx):
        
        img = skimage.io.imread(self.image_sar_paths[idx]) #, cv2.IMREAD_UNCHANGED) #cv2.IMREAD_COLOR
        #img = np.load(self.image_sar_paths[idx])
        #img = img.reshape((img.shape[0],img.shape[1],1))  ## for despeckled sar image ##
        #img = cv2.imread(self.image_sar_paths[idx])
        ################################### Small temp change #####################################
        if not self.test:
            rgb = skimage.io.imread(self.image_rgb_paths[idx])
        
        #rgb = skimage.io.imread(os.path.join('/home/airl-gpu1/Sumanth/Spacenet6_Baseline/Datasets/train/AOI_11_Rotterdam/PS-RGB', os.path.basename(self.label_paths[idx]).replace('SAR-Intensity', 'PS-RGB') )) 
        #rgbir = skimage.io.imread(os.path.join('/home/airl-gpu1/Sumanth/Spacenet6_Baseline/Datasets/train/AOI_11_Rotterdam/PS-RGBNIR', os.path.basename(self.label_paths[idx]).replace('SAR-Intensity', 'PS-RGBNIR') ))
        #######################################################################################################
        #m = np.where((img.sum(axis=2) > 0).any(1))                     ###### for sar image ####################
        m = np.where((rgb.sum(axis=2) > 0).any(1))                   ######## for tiled rgb images ###########
        ymin, ymax = np.amin(m), np.amax(m) + 1
        #m = np.where((img.sum(axis=2) > 0).any(0))
        m = np.where((rgb.sum(axis=2) > 0).any(0))
        xmin, xmax = np.amin(m), np.amax(m) + 1
        img = img[ymin:ymax, xmin:xmax]
        if not self.test:
            rgb = rgb[ymin:ymax, xmin:xmax]
        #rgbir = rgbir[ymin:ymax, xmin:xmax]
    
        if self.train:
            msk = skimage.io.imread(self.label_paths[idx]) #, cv2.IMREAD_UNCHANGED)
            #pan = skimage.io.imread(os.path.join('/data/train/AOI_11_Rotterdam/PAN', os.path.basename(self.label_paths[idx]).replace('SAR-Intensity', 'PAN') )) 
            #rgb = skimage.io.imread(os.path.join('/home/airl-gpu1/Sumanth/Spacenet6_Baseline/Datasets/train/AOI_11_Rotterdam/PS-RGB', os.path.basename(self.label_paths[idx]).replace('SAR-Intensity', 'PS-RGB') )) 
            #rgbir = skimage.io.imread(os.path.join('/home/airl-gpu1/Sumanth/Spacenet6_Baseline/Datasets/train/AOI_11_Rotterdam/PS-RGBNIR', os.path.basename(self.label_paths[idx]).replace('SAR-Intensity', 'PS-RGBNIR') ))
            #rgb = np.concatenate([rgb, pan], axis=2)

            msk = msk[ymin:ymax, xmin:xmax]             

            #rgb = rgb[ymin:ymax, xmin:xmax]
            #rgbir = rgbir[ymin:ymax, xmin:xmax]

            #pad = max(0, self.crop_size - img.shape[0])       ### for sar images #####
            pad = max(0, self.crop_size - rgb.shape[0])       #### for tiled rgb images ####
            img = cv2.copyMakeBorder(img, 0, pad, 0, 0, cv2.BORDER_CONSTANT, 0.0)
            msk = cv2.copyMakeBorder(msk, 0, pad, 0, 0, cv2.BORDER_CONSTANT, 0.0)
            rgb = cv2.copyMakeBorder(rgb, 0, pad, 0, 0, cv2.BORDER_CONSTANT, 0.0)
            #rgbir = cv2.copyMakeBorder(rgbir, 0, pad, 0, 0, cv2.BORDER_CONSTANT, 0.0)

            if random.random() < args.rot_prob:
                #rot_mat = cv2.getRotationMatrix2D((img.shape[0] // 2, img.shape[1] // 2), random.randint(0, 10) - 5, 1.0)           #### for sar images #############
                rot_mat = cv2.getRotationMatrix2D((rgb.shape[0] // 2, rgb.shape[1] // 2), random.randint(0, 10) - 5, 1.0)          ## for tiled rgb images #########
                img = cv2.warpAffine(img, rot_mat, img.shape[:2], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                msk = cv2.warpAffine(msk, rot_mat, msk.shape[:2], flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
                rgb = cv2.warpAffine(rgb, rot_mat, rgb.shape[:2], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                #rgbir = cv2.warpAffine(rgbir, rot_mat, rgbir.shape[:2], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

            if random.random() < args.scale_prob:
                #rot_mat = cv2.getRotationMatrix2D((img.shape[0] // 2, img.shape[1] // 2), 0, random.uniform(0.5,2.0))        ## for sar images######
                rot_mat = cv2.getRotationMatrix2D((rgb.shape[0] // 2, rgb.shape[1] // 2), 0, random.uniform(0.5,2.0))    ### for tiled rgb ###
                img = cv2.warpAffine(img, rot_mat, img.shape[:2], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                msk = cv2.warpAffine(msk, rot_mat, msk.shape[:2], flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
                rgb = cv2.warpAffine(rgb, rot_mat, rgb.shape[:2], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                #rgbir = cv2.warpAffine(rgbir, rot_mat, rgbir.shape[:2], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

            x0 = random.randint(0, img.shape[1] - self.crop_size)  ##for sar images##
            y0 = random.randint(0, img.shape[0] - self.crop_size)  ##for sar images##
            
            img = img[y0 : y0 + self.crop_size, x0 : x0 + self.crop_size]
            msk = msk[y0 : y0 + self.crop_size, x0 : x0 + self.crop_size]
            rgb = rgb[y0 : y0 + self.crop_size, x0 : x0 + self.crop_size]
            #msk = cv2.resize(msk, dsize=(512,512), interpolation=cv2.INTER_LINEAR)
            #rgb = cv2.resize(rgb, dsize=(512,512), interpolation=cv2.INTER_LINEAR)
            ######################################################################### Comment only for SAR 3 channel/SAR span image ##################################################
            if random.random() < self.color_aug_prob:
                img = saturation(img, 0.8 + random.random() * 0.4)
            if random.random() < self.color_aug_prob:
                img = brightness(img, 0.8 + random.random() * 0.4)
            if random.random() < self.color_aug_prob:
                img = contrast(img, 0.8 + random.random() * 0.4)
            if random.random() < self.gamma_aug_prob:
                gamma = 0.8  + 0.4 * random.random()
                img = np.clip(img, a_min = 0.0, a_max = None)
                img = np.power(img, gamma)
            if random.random() < args.gauss_aug_prob:
                gauss = np.random.normal(10.0, 10.0**0.5 , img.shape)
                img += gauss - np.min(gauss)
            if random.random() < args.elastic_aug_prob:
                el_det = self.elastic.to_deterministic()
                img = el_det.augment_image(img)
            if random.random() < self.flipud_prob:
                img = np.flipud(img)
                msk = np.flipud(msk)
                rgb = np.flipud(rgb)
                
            if random.random() < self.fliplr_prob:
                img = np.fliplr(img)
                msk = np.fliplr(msk)
                rgb = np.fliplr(rgb)
                
            if random.random() < self.rot90_prob:
                k = random.randint(0,3)
                img = np.rot90(img, k)
                msk = np.rot90(msk, k)
                rgb = np.rot90(rgb, k)
                
            if random.random() < self.channel_swap_prob:
                c1 = random.randint(0,3)
                c2 = random.randint(0,3)
                img[:, :, [c1, c2]] = img[:, :, [c2, c1]]

        direction, strip, coord = parse_img_id(self.image_sar_paths[idx], self.orients)
        if direction.item():
            img = np.fliplr(np.flipud(img))
            if self.train:
                msk = np.fliplr(np.flipud(msk))
                rgb = np.fliplr(np.flipud(rgb))
        
        img = (img - np.array([28.62501827, 36.09922463, 33.84483687, 26.21196667])) / np.array([8.41487376, 8.26645475, 8.32328472, 8.63668993])           #### comment only for sar 3 ch/span image ##########
        img = torch.from_numpy(img.transpose((2, 0, 1)).copy()).float()
        
        if self.normalize:
            tn = torchvision.transforms.Normalize(
            [0],
            [1] 
            )
            img = tn(img)

        if self.reorder_bands == 1:
            img = img[[2,3,0,1]]
        elif self.reorder_bands == 2:
            img = img[[1,3,0,2]]
        elif self.reorder_bands == 3:
            img = img[[0,3,1,2]]

        if self.train:
            weights = np.ones_like(msk[:,:,:1], dtype=float)
            regionlabels, regioncount = measure.label(msk[:,:,0], background=0, connectivity=1, return_num=True)
            regionproperties = measure.regionprops(regionlabels)
            for bl in range(regioncount):
                if regionproperties[bl].area < self.train_min_building_size:
                    msk[:,:,0][regionlabels == bl+1] = 0
                    msk[:,:,1][regionlabels == bl+1] = 0
                weights[regionlabels == bl+1] = 1024.0 / regionproperties[bl].area


            msk[:, :, :3] = (msk[:, :, :3] > 1) * 1
            weights = torch.from_numpy(weights.transpose((2, 0, 1)).copy()).float()
            msk = torch.from_numpy(msk.transpose((2, 0, 1)).copy()).float()
            rgb = torch.from_numpy(rgb.transpose((2, 0, 1)).copy()).float()

        else:
            rgb = torch.from_numpy(rgb.transpose((2, 0, 1)).copy()).float()
            if self.normalize:
                tn = torchvision.transforms.Normalize(
                [0, 0, 0],
                [1, 1, 1] 
                )
                rgb = tn(rgb)
            msk=weights = regioncount = torch.Tensor([0])

        return {"img": img, "mask": msk, "rgb": rgb, 'strip': strip, 'direction': direction, 'coord': coord, 'img_name': self.image_sar_paths[idx],
                'ymin': ymin, 'xmin': xmin, 'b_count': regioncount, 'weights': weights}'''

"""import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

import cv2
#rgb_imgs = sorted([f for f in glob.glob(os.path.join('/home/wirin/Aniruddh/SN6_dataset/val_10percent/AOI_11_Rotterdam/3_2_tiled_imgs/*.tif'))],key=numericalSort)
#tiled_data = []
himgs = []
rgb = skimage.io.imread('/home/wirin/Aniruddh/SN6_dataset/val_10percent/AOI_11_Rotterdam/PS-RGB/SN6_Train_AOI_11_Rotterdam_PS-RGB_20190804111224_20190804111453_tile_8693.tif')
tile_h, tile_w = rgb.shape[0]//3, rgb.shape[1]//3
tiled_data = [rgb[x:x+tile_h,y:y+tile_w,:] for x in range(0,rgb.shape[0],tile_h) for y in range(0,rgb.shape[1],tile_w)]
'''for i in range(len(rgb_imgs)):
    print(rgb_imgs[i])
    tiled_imgs = gdal.Open(rgb_imgs[i])
    band1 = tiled_imgs.GetRasterBand(1) 
    band2 = tiled_imgs.GetRasterBand(2) 
    band3 = tiled_imgs.GetRasterBand(3)
    b1 = band1.ReadAsArray()
    b2 = band2.ReadAsArray()
    b3 = band3.ReadAsArray()
    rgb_img = np.dstack((b1,b2,b3))
    tiled_data.append(rgb_img)'''
x=0
for i in range(3):
    img = tiled_data[x]
    for j in range(2):
        img = np.concatenate((img,tiled_data[x+j+1]),axis=1)
    himgs.append(img)
    x = x+3
img = himgs[0]
for j in range(2):
    img = np.concatenate((img,himgs[j+1]),axis=0)
img = Image.fromarray(img, 'RGB')
img.save('numpy_merge_tiles3x3.tif')"""


#tiled_sr_mask =  sorted([f for f in glob.glob(os.path.join('/home/wirin/Aniruddh/SN6_dataset/train_10percent/AOI_11_Rotterdam/Tiled-SR-masks/*.tif'))])
'''tiled_sr_mask = '/home/wirin/Aniruddh/SN6_dataset/train_10percent/AOI_11_Rotterdam/SN6_Train_AOI_11_Rotterdam_PS-RGB_20190804111224_20190804111453_tile_8679_0.tif'
tiled_sr_rgb =  sorted([f for f in glob.glob(os.path.join('/home/wirin/Aniruddh/SN6_dataset/train_10percent/AOI_11_Rotterdam/Tiled-SR-RGB/*.tif'))])
tiled_bilin_rgb = sorted([f for f in glob.glob(os.path.join('/home/wirin/Aniruddh/SN6_dataset/train_10percent/AOI_11_Rotterdam/Tiled-Bilin-RGB/*.tif'))])
dest_trash_rgb = '/home/wirin/Aniruddh/SN6_dataset/semi-trash/Bad_sr_imgs'
dest_trash_mask = '/home/wirin/Aniruddh/SN6_dataset/semi-trash/Bad_sr_masks'
dest_trash_bilin_rgb = '/home/wirin/Aniruddh/SN6_dataset/semi-trash/Bad_bilin_imgs'
for i in range(len(tiled_sr_rgb)):
    msk = skimage.io.imread(tiled_sr_mask)
    if(np.any(msk)>0):
        msk = msk*255
        plt.imshow(msk)
        plt.savefig('what.jpg')
        plt.close()
    print(np.sum(msk))
    print(hey)
    #print(np.unique(msk))
    if(np.sum(msk==175)):
        #print(i)
        shutil.move(tiled_sr_rgb[i],dest_trash_rgb)
        shutil.move(tiled_sr_mask[i],dest_trash_mask)
        shutil.move(tiled_bilin_rgb[i],dest_trash_bilin_rgb)

print("done")'''
"""import cv2

imgs = []
mode = []

def plot(imgs,mode):
    fig = plt.figure(figsize=(25, 25))
    rows = 1
    columns = 1
    for i in range(len(imgs)):
        mode_i = mode[i]
        #fig.add_subplot(rows, columns, i+1)
        plt.imshow(imgs[i])
        #print(imgs[i])
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'SAR_chnl_{mode_i}.jpg')

sar_imgs = sorted([f for f in glob.glob(os.path.join('/home/wirin/Aniruddh/SN6_dataset/train_10percent/AOI_11_Rotterdam/SAR-Intensity/*.tif'))])
sar_masks = sorted([f for f in glob.glob(os.path.join('/home/wirin/Aniruddh/SN6_dataset/train_10percent/AOI_11_Rotterdam/masks/*.tif'))])
 
for i in range(1):
    mean = []
    var = []
    inv_mean = []
    inv_var = []

    msk = skimage.io.imread(sar_masks[i])
    msk = msk/255.0
    inv_msk = cv2.bitwise_not(msk.astype(np.uint8))
    inv_msk = inv_msk/255.0
    
    sar_img = skimage.io.imread(sar_imgs[i])

    sar_view = gdal.Open(sar_imgs[i])
    band1 = sar_view.GetRasterBand(1) 
    band2 = sar_view.GetRasterBand(2) 
    band3 = sar_view.GetRasterBand(3)
    band4 = sar_view.GetRasterBand(4)
    b1 = band1.ReadAsArray()
    b2 = band2.ReadAsArray()
    b3 = band3.ReadAsArray()
    b4 = band4.ReadAsArray()
    img_sar = np.dstack((b1,b2,b3,b4))
    #img_sar = np.fliplr(np.flipud(img_sar))
    '''img_sar = np.rot90(img_sar,4)
    img_sarch1 = img_sar[:,:,0]
    imgs.append(img_sarch1)
    mode.append(1)
    img_sarch2 = img_sar[:,:,1]
    imgs.append(img_sarch2)
    mode.append(2)
    img_sarch3 = img_sar[:,:,2]
    imgs.append(img_sarch3)
    mode.append(3)
    img_sarch4 = img_sar[:,:,3]
    imgs.append(img_sarch4)
    mode.append(4)'''
    #plot(imgs,mode)


    msk = msk[:,:,0]
    msk_4d = np.stack((msk,msk,msk,msk),axis=2)
    img = img_sar*msk_4d

    '''img_sarch1 = img[:,:,0]
    imgs.append(img_sarch1)
    mode.append(1)
    img_sarch2 = img[:,:,1]
    imgs.append(img_sarch2)
    mode.append(2)
    img_sarch3 = img[:,:,2]
    imgs.append(img_sarch3)
    mode.append(3)
    img_sarch4 = img[:,:,3]
    imgs.append(img_sarch4)
    mode.append(4)
    #plot(imgs,mode)'''
    
    inv_msk = inv_msk[:,:,0]
    inv_msk_4d = np.stack((inv_msk,inv_msk,inv_msk,inv_msk),axis=2)
    inv_img = img_sar*inv_msk_4d

    '''img_sarch1 = inv_img[:,:,0]
    imgs.append(img_sarch1)
    mode.append(1)
    img_sarch2 = inv_img[:,:,1]
    imgs.append(img_sarch2)
    mode.append(2)
    img_sarch3 = inv_img[:,:,2]
    imgs.append(img_sarch3)
    mode.append(3)
    img_sarch4 = inv_img[:,:,3]
    imgs.append(img_sarch4)
    mode.append(4)
    plot(imgs,mode)'''

    mean.append(np.mean(img[:,:,0]))
    mean.append(np.mean(img[:,:,1]))
    mean.append(np.mean(img[:,:,2]))
    mean.append(np.mean(img[:,:,3]))

    var.append(np.var(img[:,:,0]))
    var.append(np.var(img[:,:,1]))
    var.append(np.var(img[:,:,2]))
    var.append(np.var(img[:,:,3]))

    inv_mean.append(np.mean(inv_img[:,:,0]))
    inv_mean.append(np.mean(inv_img[:,:,1]))
    inv_mean.append(np.mean(inv_img[:,:,2]))
    inv_mean.append(np.mean(inv_img[:,:,3]))

    inv_var.append(np.var(inv_img[:,:,0]))
    inv_var.append(np.var(inv_img[:,:,1]))
    inv_var.append(np.var(inv_img[:,:,2]))
    inv_var.append(np.var(inv_img[:,:,3]))
    
    print("Mean:", mean)
    print("Var:", var)
    print("\n")
    print("Inv_Mean:", inv_mean)
    print("Inv_Var:", inv_var)"""

def rgb_mask2label_index(rgb_mask_path,dest_path):
    rgb = sorted([f for f in glob.glob(os.path.join(rgb_mask_path))])
    for i in range(len(rgb)):
        rgb_mask = skimage.io.imread(rgb[i])
        img_name = '_'.join(os.path.splitext(os.path.basename(rgb[i]))[0].split('_')[-10:])+'.tif'
        target = torch.from_numpy(rgb_mask)
        colors = torch.unique(target.view(-1, target.size(2)), dim=0).numpy()
        target = target.permute(2, 0, 1).contiguous()

        mapping = {tuple(c): t for c, t in zip(colors.tolist(), range(len(colors)))}

        mask = torch.empty(900, 900, dtype=torch.long)
        for k in mapping:
            # Get all indices for current class
            idx = (target==torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
            validx = (idx.sum(0) == 3)  # Check that all channels match
            mask[validx] = torch.tensor(mapping[k], dtype=torch.long)
        mask = mask.float().numpy()
        #mask = np.expand_dims(mask, axis=2)
        #mask = torch.from_numpy(mask)
        #print((mask == 0).nonzero(as_tuple=False))
        #print(hey)
        label_index_mask = os.path.join(dest_path,img_name)
        skimage.io.imsave(label_index_mask,arr=mask)
    print("done")

#rgb_mask2label_index('/home/wirin/Aniruddh/SN6_dataset/val_set/AOI_11_Rotterdam/masks/*.tif','/home/wirin/Aniruddh/SN6_dataset/val_set/AOI_11_Rotterdam/label_index_masks')

def SAR_3ch(img_path,dest_path):
    sar = sorted([f for f in glob.glob(os.path.join(img_path))])
    for i in range(len(sar)):
        sar_img = skimage.io.imread(sar[i])
        img_name = '_'.join(os.path.splitext(os.path.basename(sar[i]))[0].split('_')[-10:])+'.tif'
        print(img_name)
        sar_img = torch.from_numpy(sar_img.transpose((2, 0, 1)).copy()).float()
        _,hs,ws = sar_img.shape
        
        sar_img = sar_img[[0,3,1,2]]
        sar_img = sar_img[:3,:,:]
        sar_img = sar_img.numpy()
        sar_img = np.moveaxis(sar_img,0,2)
        sar_img_loc = os.path.join(dest_path,img_name)
        skimage.io.imsave(sar_img_loc,arr=sar_img)

#SAR_3ch('/home/wirin/Aniruddh/SN6_dataset/val_set/AOI_11_Rotterdam/SAR-Intensity/*.tif', '/home/wirin/Aniruddh/SN6_dataset/val_set/AOI_11_Rotterdam/SAR-3ch')

import torchvision.transforms as transforms

def six_chnl_sar(sar_path,dest_path,reorder_bands):
    sar = sorted([f for f in glob.glob(os.path.join(sar_path))])
    for i in range(len(sar)):
        sar_img = skimage.io.imread(sar[i])
        img_name = '_'.join(os.path.splitext(os.path.basename(sar[i]))[0].split('_')[-10:])+'.tif'
        print(img_name)
        sar_img = torch.from_numpy(sar_img.transpose((2, 0, 1)).copy()).float()
        _,hs,ws = sar_img.shape

        if(reorder_bands == 1):
            sar_img = sar_img[[2,3,0,1]]
        elif(reorder_bands == 2):
            sar_img = sar_img[[1,3,0,2]]
        elif(reorder_bands == 3):
            sar_img = sar_img[[0,3,1,2]]

        #mean_img = torch.zeros(hs,ws)
        #var_img = torch.zeros(hs,ws)

        transform = transforms.Pad(1)
        padded_sar = transform(sar_img)
        _,h,w = padded_sar.shape
        mean_img_chnl0 = torch.zeros(hs,ws)
        mean_img_chnl1 = torch.zeros(hs,ws)
        mean_img_chnl2 = torch.zeros(hs,ws)
        mean_img_chnl3 = torch.zeros(hs,ws)

        var_img_chnl0 = torch.zeros(hs,ws)
        var_img_chnl1 = torch.zeros(hs,ws)
        var_img_chnl2 = torch.zeros(hs,ws)
        var_img_chnl3 = torch.zeros(hs,ws)
        
        for i in range(1,h-1):
            for j in range(1, w-1):
                    neigh_arr_chnl0 = np.asarray([padded_sar[0,i-1,j-1],padded_sar[0,i-1,j],padded_sar[0,i-1,j+1],padded_sar[0,i,j-1],padded_sar[0,i,j],padded_sar[0,i,j+1],padded_sar[0,i+1,j-1],padded_sar[0,i+1,j],padded_sar[0,i+1,j+1]])
                    neigh_arr_chnl1 = np.asarray([padded_sar[1,i-1,j-1],padded_sar[1,i-1,j],padded_sar[1,i-1,j+1],padded_sar[1,i,j-1],padded_sar[1,i,j],padded_sar[1,i,j+1],padded_sar[1,i+1,j-1],padded_sar[1,i+1,j],padded_sar[1,i+1,j+1]])
                    neigh_arr_chnl2 = np.asarray([padded_sar[2,i-1,j-1],padded_sar[2,i-1,j],padded_sar[2,i-1,j+1],padded_sar[2,i,j-1],padded_sar[2,i,j],padded_sar[2,i,j+1],padded_sar[2,i+1,j-1],padded_sar[2,i+1,j],padded_sar[2,i+1,j+1]])
                    neigh_arr_chnl3 = np.asarray([padded_sar[3,i-1,j-1],padded_sar[3,i-1,j],padded_sar[3,i-1,j+1],padded_sar[3,i,j-1],padded_sar[3,i,j],padded_sar[3,i,j+1],padded_sar[3,i+1,j-1],padded_sar[3,i+1,j],padded_sar[3,i+1,j+1]])
                                            
                    neigh_tensor_chnl0 = torch.from_numpy(neigh_arr_chnl0)
                    neigh_tensor_chnl1 = torch.from_numpy(neigh_arr_chnl1)
                    neigh_tensor_chnl2 = torch.from_numpy(neigh_arr_chnl2)
                    neigh_tensor_chnl3 = torch.from_numpy(neigh_arr_chnl3)

                    var0, mean0 = torch.var_mean(neigh_tensor_chnl0)
                    var1, mean1 = torch.var_mean(neigh_tensor_chnl1)
                    var2, mean2 = torch.var_mean(neigh_tensor_chnl2)
                    var3, mean3 = torch.var_mean(neigh_tensor_chnl3)

                    mean_img_chnl0[(i-1,j-1)] = mean0.item()
                    var_img_chnl0[(i-1,j-1)] = var0.item()
                    mean_img_chnl1[(i-1,j-1)] = mean1.item()
                    var_img_chnl1[(i-1,j-1)] = var1.item()
                    mean_img_chnl2[(i-1,j-1)] = mean2.item()
                    var_img_chnl2[(i-1,j-1)] = var2.item()
                    mean_img_chnl3[(i-1,j-1)] = mean3.item()
                    var_img_chnl3[(i-1,j-1)] = var3.item()

        enhanced_sar = torch.stack([sar_img[0,:,:],sar_img[1,:,:],sar_img[2,:,:],sar_img[3,:,:],mean_img_chnl0,mean_img_chnl1,mean_img_chnl2,mean_img_chnl3,var_img_chnl0,var_img_chnl1,var_img_chnl2,var_img_chnl3],axis=0)
        enhanced_sar = enhanced_sar.numpy()
        enhanced_sar = np.moveaxis(enhanced_sar, 0, 2)
        enh_sar = os.path.join(dest_path,img_name)
        skimage.io.imsave(enh_sar, arr=enhanced_sar)
    print("done")

#six_chnl_sar('/home/wirin/Aniruddh/SN6_dataset/train_10percent/AOI_11_Rotterdam/SAR-Intensity/*.tif', '/home/wirin/Aniruddh/SN6_dataset/train_10percent/AOI_11_Rotterdam/12_chnl_sar',reorder_bands=3)

def tiff_to_pt(tiff_path):
    tiff = sorted([f for f in glob.glob(os.path.join(tiff_path))])
    for i in range(len(tiff)):
        tiff_img = skimage.io.imread(tiff[i])
        img_name = '_'.join(os.path.splitext(os.path.basename(tiff[i]))[0].split('_')[-10:])
        print(img_name)
        tiff_img = tiff_img[:,:,0]
        tiff_img = torch.from_numpy(tiff_img).float()
        tiff_img = tiff_img.unsqueeze(0)
        tiff_img = tiff_img.view(tiff_img.size(0),-1)
        tiff_img = tiff_img.squeeze(0)
        #tiff_img = tiff_img.permute(1,0)
        print(tiff_img.shape)
        torch.save(tiff_img,'/home/wirin/Aniruddh/SN6_dataset/train_10percent/AOI_11_Rotterdam/tiff_n_pt_mask/'+img_name+'.pt')

#tiff_to_pt('/home/wirin/Aniruddh/SN6_dataset/train_10percent/AOI_11_Rotterdam/tiff_n_pt_mask/*.tif')


import torch
import torch.nn as  nn
import pdb

class PixelwiseContrastiveLoss(torch.nn.Module):
    '''
    The Pixel wise Contrastive Loss
    '''
    def __init__(self,
                 neg_multiplier=6,
                 n_max_pos=128,
                 boundary_aware=False,
                 boundary_loc='both',
                 sampling_type='full',
                 temperature=0.1):
        super(PixelwiseContrastiveLoss, self).__init__()
        self.cosine = nn.CosineSimilarity(dim=-1, eps=1e-8)
        self.n_max_pos = n_max_pos
        self.n_max_neg = n_max_pos * neg_multiplier
        self.boundary_aware = boundary_aware
        self.boundary_loc = boundary_loc
        self.sampling_type = sampling_type # 'full', 'random', 'linear'
        self.temperature = temperature

    def extract_boundary(self, real_label, is_pos=True):
        if not is_pos:
            real_label = 1 - real_label

        gt_b = F.max_pool2d(1 - real_label, kernel_size=5, stride=1, padding=2)
        gt_b_in = 1 - gt_b
        gt_b -= 1 - real_label
        return gt_b, gt_b_in

    def sample_pixels(self, label, n):
        labels = label>0
        cand_pixels = labels.nonzero()
        #cand_pixels = torch.nonzero(label)
        
        sample_idx = torch.randperm(cand_pixels.shape[0])[:n]
        
        sample_pixels = cand_pixels[sample_idx]
       
        return sample_pixels

    def _sample_balance(self, cand_pixels, n):
        batch_idx = cand_pixels[:,0]
        bs = batch_idx.max() + 1
        n_per_sample = n // bs
        sample_idx = []
        accum = 0
        for b in range(bs):
            n_features = int((batch_idx == b).sum().cpu())
            temp_idx = np.random.permutation(n_features)[:n_per_sample] + accum
            sample_idx += temp_idx.tolist()
            accum += n_features
        return sample_idx

    def split_n(self, n, boundary_type, limit, split_param=None):
        if n < limit:
            valid_n = n
        else:
            valid_n = limit

        if boundary_type == 'full':
            return valid_n, n-valid_n
        elif boundary_type == 'exclude':
            return 0, valid_n
        elif boundary_type == 'random':
            n_bd = int(torch.rand(1) * valid_n)
            n_not_bd = valid_n - n_bd
            return n_bd, n_not_bd
        elif boundary_type == 'linear':
            current_epoch, max_epoch = split_param
            n_bd = int(current_epoch/max_epoch * valid_n)
            n_not_bd = valid_n - n_bd
            return n_bd, n_not_bd
        elif boundary_type == 'fixed':
            n_bd = int(0.2 * valid_n)
            n_not_bd = valid_n - n_bd
            return n_bd, n_not_bd

    def forward(self, predict_seg_map, real_label, split_param=None, vector="embedding"):

        contrast_feature = torch.cat(torch.unbind(predict_seg_map, dim=1), dim=0)  # of size (bsz*v, c, h, w)
        predict_seg_map = contrast_feature
        
        real_label = torch.cat(torch.unbind(real_label, dim=1), dim=0).unsqueeze(dim=1)

        if self.boundary_aware:
            if self.boundary_loc == 'pos':
                pos_b, pos_b_in = self.extract_boundary(real_label)
                n_pos_bd, n_pos_not_bd = self.split_n(self.n_max_pos,
                                                      self.sampling_type,
                                                      limit=pos_b.sum(),
                                                      split_param=split_param)
                neg_b, neg_b_in = 1-real_label, 1-real_label
                n_neg_bd, n_neg_not_bd = 0, self.n_max_neg
            elif self.boundary_loc == 'neg':
                neg_b, neg_b_in = self.extract_boundary(real_label, is_pos=False)
                n_neg_bd, n_neg_not_bd = self.split_n(self.n_max_neg,
                                                      self.sampling_type,
                                                      limit=neg_b.sum(),
                                                      split_param=split_param)
                pos_b, pos_b_in = real_label, real_label
                n_pos_bd, n_pos_not_bd = 0, self.n_max_pos
            elif self.boundary_loc == 'both':
                pos_b, pos_b_in = self.extract_boundary(real_label)
                neg_b, neg_b_in = self.extract_boundary(real_label, is_pos=False)
                n_pos_bd, n_pos_not_bd = self.split_n(self.n_max_pos,
                                                      self.sampling_type,
                                                      limit=pos_b.sum(),
                                                      split_param=split_param)
                n_neg_bd = n_pos_bd
                n_neg_not_bd = self.n_max_neg - n_neg_bd
        else:
            pos_b, pos_b_in = real_label, real_label
            neg_b, neg_b_in = 1-real_label, 1-real_label

            #print((pos_b_in == 2).nonzero(as_tuple=False))
            
            n_pos_bd, n_pos_not_bd = 0, self.n_max_pos
            n_neg_bd, n_neg_not_bd = 0, self.n_max_neg

        # sample positive pixels
        pos_b_pixels = self.sample_pixels(pos_b, n_pos_bd)
        pos_b_in_pixels = self.sample_pixels(pos_b_in, n_pos_not_bd)
        pos_pixels = torch.cat((pos_b_pixels, pos_b_in_pixels), dim=0).detach()
        #print('pos_pixels:',pos_pixels)
        #print('\n')
        #print(pos_pixels.size())
        pos_pixels = tuple(pos_pixels.t())
        #print(len(pos_pixels))
        

        # sample negative pixels
        neg_b_pixels = self.sample_pixels(neg_b, n_neg_bd)
        neg_b_in_pixels = self.sample_pixels(neg_b_in, n_neg_not_bd)
        neg_pixels = torch.cat((neg_b_pixels, neg_b_in_pixels), dim=0).detach()
        neg_pixels = tuple(neg_pixels.t())
        #print(hey)

        if vector == "embedding" or vector == "first"  :
            positive_logits = predict_seg_map[pos_pixels[0], :, pos_pixels[2], pos_pixels[3]]
            #print('positive_logits:',positive_logits)
            #print('\n')
            negative_logits = predict_seg_map[neg_pixels[0], :, neg_pixels[2], neg_pixels[3]]
            #print('negative_logits:',negative_logits)
            #print('\n')
            
        elif vector == "second" :
            positive_logits = predict_seg_map[pos_pixels[0], :, pos_pixels[2], pos_pixels[3]]
            positive_logits = positive_logits[:int(len(positive_logits)/4)]

            negative_logits = predict_seg_map[neg_pixels[0], :, neg_pixels[2], neg_pixels[3]]
            negative_logits = negative_logits[:int(len(negative_logits)/4)]
        elif vector == "third" :
            positive_logits = predict_seg_map[pos_pixels[0], :, pos_pixels[2], pos_pixels[3]]
            positive_logits = positive_logits[:int(len(positive_logits)/16)]

            negative_logits = predict_seg_map[neg_pixels[0], :, neg_pixels[2], neg_pixels[3]]
            negative_logits = negative_logits[:int(len(negative_logits)/16)]
        elif vector == "four" :
            positive_logits = predict_seg_map[pos_pixels[0], :, pos_pixels[2], pos_pixels[3]]
            positive_logits = positive_logits[:int(len(positive_logits)/64)]

            negative_logits = predict_seg_map[neg_pixels[0], :, neg_pixels[2], neg_pixels[3]]
            negative_logits = negative_logits[:int(len(negative_logits)/64)]

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            all_positive_logits = SyncFunction.apply(positive_logits)
            all_negative_logits = SyncFunction.apply(negative_logits)
        else:
            all_positive_logits = positive_logits
            all_negative_logits = negative_logits

        pos_nll = self._compute_loss(positive_logits,
                                     all_positive_logits,
                                     all_negative_logits)

        return pos_nll

    def _compute_loss(self, pos, all_pos, all_negs):
        
        positive_sim = self.cosine(pos.unsqueeze(1),
                                   all_pos.unsqueeze(0))
        
        exp_positive_sim = torch.exp(positive_sim/self.temperature)
        #print('exp_positive_sim:',exp_positive_sim)
        #print('\n')
        off_diagonal = torch.ones(exp_positive_sim.shape).type_as(exp_positive_sim)
        off_diagonal = off_diagonal.fill_diagonal_(0.0)
        exp_positive_sim = exp_positive_sim * off_diagonal
        positive_row_sum = torch.sum(exp_positive_sim, dim=1)

        negative_sim = self.cosine(pos.unsqueeze(1),
                                   all_negs.unsqueeze(0))
        exp_negative_sim = torch.exp(negative_sim/self.temperature)
        #print('exp_negative_sim:',exp_negative_sim)
        #print('\n')
        negative_row_sum = torch.sum(exp_negative_sim, dim=1)

        likelihood = positive_row_sum / (positive_row_sum + negative_row_sum)
        
        #print('\n')
        if(likelihood.nelement()==0):
            nll = torch.tensor(0.0).to('cuda:0')
        else:
            nll = -torch.log(likelihood).mean().to('cuda:0')
        #print('nll:',nll)
        #print('\n')
        if(torch.isnan(nll)):
            print(hey)

        return nll

criterion_MLP = nn.BCELoss()

def error_MLP(out,g_t):
    #return F.binary_cross_entropy_with_logits(out,g_t)
    return criterion_MLP(out,g_t)

def reg_MLP(out,g_t):
    err = []
    e_thresh = 0.1
    correct_prediction = torch.eq(torch.argmax(g_t, 1), torch.argmax(out, 1))
    for i in range(len(correct_prediction)):
        e = torch.max(abs(error_MLP(out[i],g_t[i])))
        if((correct_prediction[i]==True) and (e < e_thresh)):
            loss_e = torch.tensor(0.0).to('cuda:3')
        else:
            loss_e = error_MLP(out[i],g_t[i]).to('cuda:3')
        
        err.append(loss_e)

    loss_ = sum(err)/len(err)
    return loss_

'''imgs = '/home/wirin/Aniruddh/SN6_dataset/val_10percent/AOI_11_Rotterdam/PS-RGB/SN6_Train_AOI_11_Rotterdam_PS-RGB_20190804111224_20190804111453_tile_8693.tif'
rgb = skimage.io.imread(imgs)
rgb = torch.from_numpy(rgb.transpose((2, 0, 1)).copy()).float()
tile_h, tile_w = rgb.size(1)//4, rgb.size(2)//4
o = [rgb[:,x:x+tile_h,y:y+tile_w] for x in range(0,rgb.size(1),tile_h) for y in range(0,rgb.size(2),tile_w)]
himgs = []
x=0
for i in range(4):
    img = o[x]
    for j in range(3):
        img = torch.cat((img,o[x+j+1]),dim=2)
    himgs.append(img)
    x = x+4
img = himgs[0]
for j in range(3):
    img = torch.cat((img,himgs[j+1]),dim=1)

img = np.moveaxis(img.data.numpy(), 0, 2)

plt.imshow(img/255)
plt.savefig('Tiling_test.jpg')'''

################################################ CutMix ###################################

def cutmix_aug(img_patch, label_patch, rgb_bg_img, label_bg_img, x0, y0):
    #lam = np.random.beta(1, 1)
    #rand_index = img_patch
    #x0 = random.randint(0, rgb_bg_img.shape[1] - img_patch.shape[1])
    #y0 = random.randint(0, rgb_bg_img.shape[1] - img_patch.shape[1])
    label_bg_img[:,y0 : y0 + label_patch.shape[1], x0 : x0 + label_patch.shape[1]] = label_patch
    #rgb_bg_img[:,:,:] = 0
    rgb_bg_img2 = rgb_bg_img.detach().clone()
    label_bg_img2 = label_bg_img.detach().clone()
    rgb_bg_img2[:,y0 : y0 + img_patch.shape[1], x0 : x0 + img_patch.shape[1]] = img_patch
    #bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
    #input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
    #print("In cutmix before return-----------",torch.unique(label_bg_img2))
    return rgb_bg_img2, label_bg_img2
############################################ Dense Multi-Scale and Cross-Scale Contrastive Loss ECCV 2022 #########################################
import torch
import torch.nn as nn
from torch.nn.functional import one_hot

def has_inf_or_nan(x):
    return torch.isinf(x).max().item(), torch.isnan(x).max().item()


class DenseContrastiveLossV2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_all_classes = 3
        self.num_real_classes = self.num_all_classes
        self.temperature = 0.1
        self.base_temperature = 1.0
        self.min_views_per_class = 5
        self.label_scaling_mode = 'nn'
        self.cross_scale_contrast = config['cross_scale_contrast'] if 'cross_scale_contrast' in config else False
        self.dominant_mode = 'all'
        self.eps = torch.tensor(1e-10)
        self.metadata = {}
        self.max_views_per_class = config['max_views_per_class'] if 'max_views_per_class' in config else 2500
        self.max_features_total = config['max_features_total'] if 'max_features_total' in config else 10000
        self.log_this_step = False
        self._scale = None

        for class_name in DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][1]:
            self.metadata[class_name] = (0.0, 0.0)  # pos-neg per class
        # self.anchors_per_image = config['anchors_per_image'] if 'anchors_per_image' in config else 50
        # sanity checks
        if self.label_scaling_mode == 'nn':
            # when using nn interpolation to get dominant_class in feature space
            # dominant_mode can only be in 'all' mode
            # there is no notion of entropy or cross entropy in the class_distr so these weighting must be set to False
            # dominant
            assert(self.dominant_mode == 'all'), \
                'cannot use label_scaling_mode: "{}" with dominant_mode: "{}" -' \
                ' only "all" is allowed'.format(self.label_scaling_mode, self.dominant_mode)

    def forward(self, label: torch.Tensor, features: torch.Tensor):
        flag_error = False
        with torch.no_grad():  # Not sure if necessary, but these steps neither are nor need to be differentiable
            scale = int(label.shape[-1] // features.shape[-1])
            class_distribution, dominant_classes = self.get_dist_and_classes(label, scale)
            # example_identification = self.identify_examples(dominant_classes, non_ignored_anchors)
        sampled_features, sampled_labels, flag_error = self.sample_anchors_fast(dominant_classes, features)

        # old
        # sampled_features, sampled_labels, flag_error = self.sample_anchors(dominant_classes, features)
        if flag_error:
            loss = features-features
            loss = loss.mean()
        else:
            loss = self.contrastive_loss(sampled_features, sampled_labels)
        # feature_correlations = self.correlate_features(features)
        # loss = self.calculate_loss(feature_correlations, dominant_classes, example_identification,
        #                            class_distribution, non_ignored_anchors)
        # return loss.mean()
        if self.cross_scale_contrast:
            return loss, sampled_features, sampled_labels, flag_error
        return loss

    def _select_views_per_class(self, min_views, total_cls, cls_in_batch, cls_counts_in_batch):
        if self.max_views_per_class == 1:
            # no cappping to views_per_class
            views_per_class = min_views
        else:
            # capping views_per_class to avoid OOM
            views_per_class = min(min_views, self.max_views_per_class)
            if views_per_class == self.max_views_per_class:
                Log.info(
                    f'\n rank {get_rank()} capping views per class to {self.max_views_per_class},'
                    f' cls_and_counts: {cls_in_batch} {cls_counts_in_batch} ')
                self.log_this_step = True
        if views_per_class * total_cls > self.max_features_total:
            views_per_class = self.max_features_total // total_cls
            printlog(
                f'\n rank {get_rank()}'
                f' capping total features  to {self.max_features_total} total_cls:  {total_cls} '
                f'--> views_per_class:  {views_per_class} ,'
                f'  cls_and_counts: {cls_in_batch} {cls_counts_in_batch}')
            self.log_this_step = True
        return views_per_class

    def sample_anchors_fast(self, dominant_classes, features):
        """
        self.anchors_per_image =
        :param dominant_classes: N-H-W-1
        :param features:  N-C-H-W
        :return: sampled_features: (classes_in_batch, C, views)
                 sampled_labels : (classes_in_batch)
        """
        flag_error = False
        n = dominant_classes.shape[0]  # batch size
        c = features.shape[1]  # feature space dimensionality
        features = features.view(n, c, -1)
        dominant_classes = dominant_classes.view(n, -1)  # flatten   # flatten n,1,h,w --> n,h*w
        skip_ids = []
        cls_in_batch = []  # list of lists each containing classes in an image of the batch
        cls_counts_in_batch = []  # list of lists each containing classes in an image of the batch

        classes_ids = torch.arange(start=0, end=self.num_all_classes, step=1, device=dominant_classes.device)
        compare = dominant_classes.unsqueeze(-1) == classes_ids.unsqueeze(0).unsqueeze(0)# n, hw, 1 == 1, 1, n_c => n,hw,n_c
        cls_counts = compare.sum(1) # n, n_c

        present_inds = torch.where(cls_counts[:, :-1] >= self.min_views_per_class) # ([0,...,n-1], [prese   nt class ids])
        batch_inds, cls_in_batch = present_inds

        min_views = torch.min(cls_counts[present_inds])
        total_cls = cls_in_batch.shape[0]

        views_per_class = self._select_views_per_class(min_views, total_cls, cls_in_batch, cls_counts_in_batch)
        sampled_features = torch.zeros((total_cls, c, views_per_class), dtype=torch.float).cuda()
        sampled_labels = torch.zeros(total_cls, dtype=torch.float).cuda()

        for i in range(total_cls):
            # print(batch_inds[i], cls_in_batch[i])
            indices_from_cl_fast = compare[batch_inds[i], :, cls_in_batch[i]].nonzero().squeeze()
            # indices_from_cl = (dominant_classes[batch_inds[i]] == cls_in_batch[i]).nonzero().squeeze()
            random_permutation = torch.randperm(indices_from_cl_fast.shape[0]).cuda()
            sampled_indices_from_cl = indices_from_cl_fast[random_permutation[:views_per_class]]
            sampled_features[i] = features[batch_inds[i], :, sampled_indices_from_cl]
            sampled_labels[i] = cls_in_batch[i]

        return sampled_features, sampled_labels, flag_error


    def sample_anchors(self, dominant_classes, features):
        """
        self.anchors_per_image =
        :param dominant_classes: N-H-W-1
        :param features:  N-C-H-W
        :return: sampled_features: (classes_in_batch, C, views)
                 sampled_labels : (classes_in_batch)
        """
        flag_error = False
        n = dominant_classes.shape[0]  # batch size
        c = features.shape[1]  # feature space dimensionality
        features = features.view(n, c, -1)
        dominant_classes = dominant_classes.view(n, -1)
        skip_ids = []
        cls_in_batch = []  # list of lists each containing classes in an image of the batch
        cls_counts_in_batch = []  # list of lists each containing classes in an image of the batch
        total_cls = 0  # classes in batch (non-unique)
        min_views = 10000
        for i in range(n):
            y_i = dominant_classes[i].squeeze()
            # cls_in_y_i = torch.unique(y_i, return_counts=True)
            # classes in i-th image of the batch
            cls_in_y_i, cls_counts_in_y_i = torch.unique(y_i, return_counts=True)
            # filter out ignore_class and classes with few views
            cls_and_counts = [(cl.item(), cl_count.item()) for cl, cl_count in zip(cls_in_y_i, cls_counts_in_y_i)
                              if cl != self.ignore_class
                              and cl_count.item() >= self.min_views_per_class]
            if len(cls_and_counts) == 0:
                # only ignore class in labels
                skip_ids.append(i)
            else:
                cls_and_counts = [x for x in zip(*cls_and_counts)]
                cls_in_y_i = list(cls_and_counts[0])
                cls_counts_in_y_i = list(cls_and_counts[1])
                # keep track of smallest class count
                min_views_current = min(cls_counts_in_y_i)
                if min_views_current < min_views:
                    min_views = min_views_current
                total_cls += len(cls_in_y_i)


            cls_counts_in_batch.append(cls_counts_in_y_i)
            cls_in_batch.append(cls_in_y_i)

        if len(skip_ids) == n:
            flag_error = True
            Log.info(f'\n rank {get_rank()} cls_and_counts : {cls_in_batch} skipping this batch')

        # select how many samples per class (with repetition) will be sampled
        views_per_class = self._select_views_per_class(min_views, total_cls, cls_in_batch, cls_counts_in_batch)

        # tensors to be populated with anchors
        sampled_features = torch.zeros((total_cls, c, views_per_class), dtype=torch.float).cuda()
        sampled_labels = torch.zeros(total_cls, dtype=torch.float).cuda()
        ind = 0
        for i in range(n):
            if i in skip_ids:
                continue
            cls_in_y_i = cls_in_batch[i]  # classes in image
            y_i = dominant_classes[i].squeeze()
            for cl in cls_in_y_i:
                indices_from_cl = (y_i == cl).nonzero().squeeze()
                random_permutation = torch.randperm(indices_from_cl.shape[0])
                sampled_indices_from_cl = indices_from_cl[random_permutation[:views_per_class]]
                sampled_features[ind] = features[i, :, sampled_indices_from_cl]
                sampled_labels[ind] = cl
                # print(ind, cl, indices_from_cl.shape[0], sampled_indices_from_cl.shape[0], views_per_class)
                ind += 1
        return sampled_features, sampled_labels, flag_error

    def contrastive_loss(self, feats, labels):
        """
        :param feats: T-C-V
                      T: classes in batch (with repetition), which can be thought of as the number of anchors
                      C: feature space dimensionality
                      V: views per class (i.e samples from each class),
                       which can be thought of as the number of views per anchor
        :param labels: T
        :return: loss
        """
        # prepare feats
        feats = torch.nn.functional.normalize(feats, p=2, dim=1)  # L2 normalization
        feats = feats.transpose(dim0=1, dim1=2)  # feats are T-V-C
        num_anchors, views_per_anchor, c = feats.shape  # get T, V, C
        labels = labels.contiguous().view(-1, 1)  # labels are T-1

        # print( f'rank: {get_rank()} -- classes {num_anchors} v_per_class {views_per_anchor} total_anchors = {num_anchors * views_per_anchor}')
        # feats_flat = torch.cat(torch.unbind(feats, dim=1), dim=0)  # feats_flat is V*T-C
        # dot_product = torch.div(torch.matmul(feats_flat, torch.transpose(feats_flat, 0, 1)), self.temperature)
        # # dot_product # V*T-C @ C-V*T = V*T-V*T
        #
        # mask, pos_mask, neg_mask = self.get_masks(labels, num_anchors, views_per_anchor)
        # loss = self.compute(pos_mask, neg_mask, dot_product)
        # print(loss)

        # modifying to more intuitive version
        labels_ = labels.repeat(1, views_per_anchor)  # labels are T-V
        labels_ = labels_.view(-1, 1)  # labels are T*V-1
        pos_mask2, neg_mask2 = self.get_masks2(labels_, num_anchors, views_per_anchor)
        feats_flat = feats.contiguous().view(-1, c)  # feats_flat is T*V-C
        dot_product = torch.div(torch.matmul(feats_flat, torch.transpose(feats_flat, 0, 1)), self.temperature)
        loss2 = self.get_loss(pos_mask2, neg_mask2, dot_product)
        # print(loss2)
        return loss2

    @staticmethod
    def get_masks(labels, num_anchors, views_per_anchor):
        """
        :param labels: T*V-1
        :param num_anchors: T
        :param views_per_anchor: V
        :return: mask, pos_maks,
        """
        # extract mask indicating same class samples
        mask = torch.eq(labels, torch.transpose(labels, 0, 1)).float()  # mask T-T
        mask = mask.repeat(views_per_anchor, views_per_anchor)  # mask V*T-V*T
        neg_mask = 1 - mask  # indicator of negatives
        # set diagonal mask elements to zero -- self-similarities
        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(num_anchors * views_per_anchor).view(-1, 1).cuda(),
                                                     0)
        pos_mask = mask * logits_mask  # indicator of positives
        return pos_mask, neg_mask

    @staticmethod
    def get_masks2(labels, num_anchors, views_per_anchor):
        """
        takes flattened labels and identifies pos/neg of each anchor
        :param labels: T*V-1
        :param num_anchors: T
        :param views_per_anchor: V
        :return: mask, pos_maks,
        """
        # extract mask indicating same class samples
        mask = torch.eq(labels, torch.transpose(labels, 0, 1)).float()  # mask T-T
        neg_mask = 1 - mask  # indicator of negatives
        # set diagonal mask elements to zero -- self-similarities
        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(num_anchors * views_per_anchor).view(-1, 1).cuda(),
                                                     0)
        pos_mask = mask * logits_mask  # indicator of positives
        return pos_mask, neg_mask

    def get_loss(self, pos, neg, dot):
        """
        :param pos: V*T-V*T
        :param neg: V*T-V*T
        :param dot: V*T-V*T
        :return:
        """
        # logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = dot  # - logits_max.detach()

        neg_logits = torch.exp(logits) * neg
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)
        # print('exp_logits ', has_inf_or_nan(exp_logits))
        log_prob = logits - torch.log(exp_logits + neg_logits)
        # print('log_prob ', has_inf_or_nan(log_prob))

        mean_log_prob_pos = (pos * log_prob).sum(1) / pos.sum(1)  # normalize by positives
        # print('\npositives: {} \nnegatives {}'.format(pos.sum(1), neg.sum(1)))
        # print('mean_log_prob_pos ', has_inf_or_nan(mean_log_prob_pos))
        loss = - mean_log_prob_pos
        # loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        loss = loss.mean()
        # print('loss.mean() ', has_inf_or_nan(loss))
        # print('loss {}'.format(loss))
        if has_inf_or_nan(loss)[0] or has_inf_or_nan(loss)[1]:
            printlog(f'\n rank {get_rank()} inf found in loss with positives {pos.sum(1)} and Negatives {neg.sum(1)}')
        return loss

    def get_dist_and_classes(self, label: torch.Tensor, scale: int) -> torch.Tensor:
        """Determines the distribution of the classes in each scale*scale patch of the ground truth label N-H-W,
        for given experiment, returning class_distribution as N-C-H//scale-W//scale tensor. Also determines dominant
        classes in each patch of the ground truth label N-H-W, based on the class distribution. Output is
        N-C-H//scale-W//scale where C might be 1 (just one dominant class) or more.

        If label_scaling_mode == 'nn' peforms nearest neighbour interpolation on the label without one_hot encoding and
        returns N-1-H//scale-W//scale
        """
        n, h, w = label.shape
        self._scale = scale
        if self.label_scaling_mode == 'nn':
            lbl_down = torch.nn.functional.interpolate(label.unsqueeze(1).float(), (h//scale, w//scale), mode='nearest')
            # non_ignored_anchors = non_ignored_anchors = (lbl_down != self.num_real_classes).view(n, h//scale * w//scale)
            return lbl_down.long(), lbl_down.long()

        elif self.label_scaling_mode == 'avg_pool':
            lbl_one_hot = one_hot(label.to(torch.int64), self.num_all_classes).permute(dims=[0, 3, 1, 2])
            class_distribution = torch.nn.AvgPool2d(kernel_size=scale)(lbl_one_hot.float())
            # class_distribution is:    N-C|all-H//scale-W//scale
            dominant_classes = self.get_dominant_classes(class_distribution)
            # dominant_classes is:      N-1-H//scale-W//scale
            non_ignored_anchors = (dominant_classes != self.num_real_classes).view(n, 1, h//scale, w//scale)
            # non_ignored_anchors is:   N-1-H//scale-W//scale
            # Note: if e.g. exp = 2, then we're looking for dom_class == num_all_classes - 1 (18 - 1 = 17), which is
            #   the same as num_real_classes (= 17), because num_real_classes is num_all_classes - 1
            #   (analogous when exp = 3)
            if self.experiment in [2, 3]:  # Need to cut and re-normalise the class_distribution
                class_distribution = class_distribution[:, :self.num_real_classes]
                # class_distribution is:    N-C|real-H//scale-W//scale
                norm_sum = torch.sum(class_distribution, dim=1, keepdim=True)
                norm_sum[norm_sum == 0] = 1
                class_distribution /= norm_sum
                class_distribution[~non_ignored_anchors.repeat(1, self.num_real_classes, 1, 1)] = \
                    1 / self.num_real_classes
                # NOTE: set class_distribution where ignored anchors are to default 1 / num_real_classes to avoid any
                #   issues with zeros during the loss calculation - eventually ignored anyway
            return class_distribution, dominant_classes, non_ignored_anchors.view(n, h//scale * w//scale)

    def get_dominant_classes(self, class_distribution: torch.Tensor, mode: str = None) -> torch.Tensor:
        """Determines dominant classes in each scale*scale patch of the ground truth label N-H-W, based on the N-C-H-W
        class distribution passed. Output is N-C-H-W where C might be 1 (just one dominant class) or more"""
        mode = self.dominant_mode if mode is None else mode
        # class_distribution is N-C-H*W
        if mode == 'all':
            dom_classes = torch.argmax(class_distribution, dim=1).unsqueeze(1)  # dom_classes is N-H-W
        elif mode in ['instruments', 'rare']:
            class_selection = DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][2][self.dominant_mode]
            dom_classes = torch.argmax(class_distribution[:, class_selection], dim=1).unsqueeze(1)
            cond_not_satisfied = torch.gather(class_distribution, dim=1, index=dom_classes) < self.dominant_thresh
            dom_classes[cond_not_satisfied] = self.get_dominant_classes(class_distribution, 'all')[cond_not_satisfied]
        # elif self.dominant_mode == 'multiple':
        #     class_selection = DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][2]['instruments']
        #     dom_classes = None
        else:
            raise ValueError("Mode '{}' not recognised".format(self.dominant_mode))
        return dom_classes