import gdal
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import base
import pandas as pd
import skimage
from shapely.wkt import dumps, loads
from shapely.geometry import shape, Polygon
import cv2
import skimage.segmentation
from skimage import measure, io
from skimage.morphology import square, erosion, dilation, remove_small_objects, remove_small_holes
import warnings
warnings.filterwarnings("ignore")
#from basicsr.archs.rrdbnet_arch import RRDBNet
#from realesrgan import RealESRGANer
#from realesrgan.archs.srvgg_arch import SRVGGNetCompact
imgs = []

def plot(imgs):
    fig = plt.figure(figsize=(9, 9))
    rows = 1
    columns = 1
    for i in range(len(imgs)):
        print("Plotting_"+str(i))
        #fig.add_subplot(rows, columns, i+1)
        plt.imshow(imgs[i])
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('./plots_of_predictions/pred_'+str(i)+'.jpg')
        plt.close()
        if i>10:
            break

#pred_files_sar = sorted([f for f in glob.glob(os.path.join('/home/wirin/Aniruddh/Spacenet_codes/wdata_overall_net_exp3/pred_fold_{0}_0/*.tif'))])
pred_files_sml_sar = sorted([f for f in glob.glob(os.path.join('/home/user/Perception/Buildingsegmentation/Sumanth/Spacenet-codes/wdata_pngsave/pred_fold_{0}_0/*.tif'))])
#pred_files_sml_sar = sorted([f for f in glob.glob(os.path.join('/home/wirin/Aniruddh/SN6_dataset/train_10percent/AOI_11_Rotterdam/Wo_Big_buildings_RGB/*.tif'))])
for i in range(len(pred_files_sml_sar)):

    #predictions_sar = gdal.Open(pred_files_sar[i])
    predictions_sml_sar = gdal.Open(pred_files_sml_sar[i])
    '''band1 = predictions_sar.GetRasterBand(1) 
    band2 = predictions_sar.GetRasterBand(2) 
    band3 = predictions_sar.GetRasterBand(3) 
    b1 = band1.ReadAsArray()
    b2 = band2.ReadAsArray()
    b3 = band3.ReadAsArray()
    img_pred_sar = np.dstack((b1, b2, b3))'''

    band1_sml = predictions_sml_sar.GetRasterBand(1) 
    band2_sml = predictions_sml_sar.GetRasterBand(2) 
    band3_sml = predictions_sml_sar.GetRasterBand(3) 
    b1_sml = band1_sml.ReadAsArray()
    b2_sml = band2_sml.ReadAsArray()
    b3_sml = band3_sml.ReadAsArray()
    img_pred_sml_sar = np.dstack((b1_sml, b2_sml, b3_sml))
    
    #imgs.append(img_pred_sar)
    imgs.append(img_pred_sml_sar)

plot(imgs)

'''tiled_data_sr_path = '/home/wirin/Aniruddh/SN6_dataset/train_10percent/AOI_11_Rotterdam/Tiled-SR-RGB'
tiled_data_bilin_path = '/home/wirin/Aniruddh/SN6_dataset/train_10percent/AOI_11_Rotterdam/Tiled-Bilin-RGB'
tiled_mask_path = '/home/wirin/Aniruddh/SN6_dataset/train_10percent/AOI_11_Rotterdam/Tiled-SR-masks'
sr_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
netscale = 4

upsampler = RealESRGANer(
                scale=netscale,
                model_path='experiments/pretrained_models/RealESRGAN_x4plus.pth',
                model=sr_model,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                #half=not args.fp32,
                gpu_id=3)'''

#rgb = sorted([f for f in glob.glob(os.path.join('/home/wirin/Aniruddh/SN6_dataset/train_10percent/AOI_11_Rotterdam/Tiled-SR-RGB/*.tif'))])
#masks = sorted([f for f in glob.glob(os.path.join('/home/wirin/Aniruddh/SN6_dataset/train_10percent/AOI_11_Rotterdam/Tiled-Bin-masks/*.tif'))])
#bilin = sorted([f for f in glob.glob(os.path.join('/home/wirin/Aniruddh/SN6_dataset/train_10percent/AOI_11_Rotterdam/Tiled-Bilin-RGB/*.tif'))])
#dest = '/home/wirin/Aniruddh/SN6_dataset/train_10percent/AOI_11_Rotterdam/Fine-Small-Coarse-Big-RGB'
#dest_msk = '/home/wirin/Aniruddh/SN6_dataset/train_10percent/AOI_11_Rotterdam/Tiled-Bin-masks'
#sup_res = sorted([f for f in glob.glob(os.path.join('/home/wirin/Aniruddh/SN6_dataset/train_10percent/AOI_11_Rotterdam/SAR-3ch/*.tif'))])

"""for i in range(1):

    '''imgs = gdal.Open(rgb[i])
    imgs = imgs.ReadAsArray()
    imgs = np.swapaxes(imgs,0,2)
    imgs = np.swapaxes(imgs,0,1)'''

    '''bilinear = gdal.Open(bilin[i])
    bilinear = bilinear.ReadAsArray()
    bilinear = np.swapaxes(bilinear,0,2)
    bilinear = np.swapaxes(bilinear,0,1)

    mask = gdal.Open(masks[i])
    mask = mask.ReadAsArray()
    mask = np.swapaxes(mask,0,2)
    mask = np.swapaxes(mask,0,1)
    mask = skimage.io.imread(masks[i])
    mask = np.dstack((mask,mask,mask))
    mask = mask/255.0'''

    #sup_res_img = skimage.io.imread(sup_res[i])
    '''sup_res_img = cv2.imread(sup_res[i])
    sup_res_img=cv2.cvtColor(sup_res_img, cv2.COLOR_BGR2GRAY)

    dft = cv2.dft(np.float32(sup_res_img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    rows, cols = sup_res_img.shape
    crow, ccol = int(rows / 2), int(cols / 2)

    mask = np.ones((rows, cols, 2), np.uint8)
    r = 80
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 0
    fshift_mask_mag = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift_mask_mag)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    tilename = '_'.join(os.path.splitext(os.path.basename(sup_res[i]))[0].split('_')[-12:])
    print(tilename)
    #fig = plt.figure(figsize=(9,9))
    #fig.add_subplot(1,1,1)
    #plt.imshow(sup_res_img.astype(np.uint8))
    fig = plt.figure(figsize=(9, 9))

    ax1 = fig.add_subplot(2,2,1)
    ax1.imshow(sup_res_img, cmap='gray')
    ax1.title.set_text('Input Image')
    ax2 = fig.add_subplot(2,2,2)
    ax2.imshow(magnitude_spectrum, cmap='gray')
    ax2.title.set_text('FFT of image')
    ax3 = fig.add_subplot(2,2,3)
    ax3.imshow(img_back, cmap='gray')
    ax3.title.set_text('After inverse FFT')

    plt.savefig('Fourier_try_SAR.jpg')
    plt.close()'''
    
    '''msk_tilename = '_'.join(os.path.splitext(os.path.basename(masks[i]))[0].split('_')[-10:])
    tile_h, tile_w = imgs.shape[0]//4, imgs.shape[1]//4
    tiled_data = [imgs[x:x+tile_h,y:y+tile_w,:] for x in range(0,imgs.shape[0],tile_h) for y in range(0,imgs.shape[1],tile_w)]
    tiled_mask_data = [mask[x:x+tile_h,y:y+tile_w,:] for x in range(0,imgs.shape[0],tile_h) for y in range(0,imgs.shape[1],tile_w)]
    for i in range(len(tiled_mask_data)):
        img_name = tilename+'_'+f'{i}.tif'
        mask_name = msk_tilename+'_'+f'{i}.tif'
        if(np.sum(tiled_mask_data[i])!=0):
            tiled_data_sr_file = os.path.join(tiled_data_sr_path,img_name)
            tiled_data_bilin_file = os.path.join(tiled_data_bilin_path,img_name)
            tiled_mask_data_file = os.path.join(tiled_mask_path,mask_name)
            a,_ = upsampler.enhance(tiled_data[i], outscale=2.4)
            msk,_ = upsampler.enhance(tiled_mask_data[i], outscale=2.4)
            bilin_upscale = cv2.resize(tiled_data[i], (540,540), interpolation=cv2.INTER_LINEAR)
            skimage.io.imsave(tiled_data_sr_file,arr=a)
            skimage.io.imsave(tiled_data_bilin_file,arr=bilin_upscale)
            skimage.io.imsave(tiled_mask_data_file,arr=msk)'''

    '''masks_0 = mask[:,:,0]
    #print(np.unique(masks_0))
    #print(hey)
    bin_msk = np.zeros(mask[:,:,0].shape)
    for i in range(bin_msk.shape[0]):
        for j in range(bin_msk.shape[1]):
            #print(masks_0[i][j])
            if(masks_0[i][j]>=128):
                #print("255s")
                bin_msk[i][j]+=255
            else:
                #print("zeros")
                bin_msk[i][j]+=0'''

    '''inv_sml_msk = 1.0-mask
    sr_small_img = imgs*mask
    
    bilin_inv_img = inv_sml_msk*bilinear
    overall_img = sr_small_img+bilin_inv_img
    
    save_path = os.path.join(dest,tilename)
    skimage.io.imsave(save_path, arr=overall_img)'''
    #print(hey)"""

print('done')


        