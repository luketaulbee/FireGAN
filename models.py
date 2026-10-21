import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19
import math
import torchvision.models as models
import torch.nn.functional as F
import cartopy.feature as cf


class PerceptualLoss(nn.Module):
    def __init__(self, feature_extractor):
        super(PerceptualLoss, self).__init__()
        self.feature_extractor = feature_extractor
        self.loss = nn.L1Loss()

    def forward(self, generated, target):
        gen_features = self.feature_extractor(generated)
        target_features = self.feature_extractor(target).detach()
        return self.loss(gen_features, target_features)



class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.vgg19_54 = nn.Sequential(*list(vgg19_model.features.children())[:35])
        # first_conv_layer = [nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
        # first_conv_layer.extend(list(vgg19_model.features.children())[:35])
        # self.feature_extractor = nn.Sequential(*first_conv_layer)

    def forward(self, img):
        return self.vgg19_54(img)
        # return self.feature_extractor(img)



class DenseResidualBlock(nn.Module):
    """
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, filters, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            layers = [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU()]
            return nn.Sequential(*layers)

        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=2 * filters)
        self.b3 = block(in_features=3 * filters)
        self.b4 = block(in_features=4 * filters)
        self.b5 = block(in_features=5 * filters, non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

    def forward(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        return out.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters), DenseResidualBlock(filters), DenseResidualBlock(filters)
        )

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x


class GeneratorRRDB(nn.Module):
    def __init__(self, channels=1, filters=64, num_res_blocks=16): #, num_upsample=2):
        super(GeneratorRRDB, self).__init__()

        # First layer
        self.conv1 = nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1)
        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])
        # Second conv layer post residual blocks
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        
        # Upsampling layers
            
        self.upsample_6x = nn.Sequential(
            nn.Conv2d(filters, filters * 36, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.PixelShuffle(upscale_factor=6),  
        )
        upsample_layers = [self.upsample_6x]
        
        
        self.upsampling = nn.Sequential(*upsample_layers)


        # Final output block
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        # print(f"Generator output shape: {out.shape}")
        return out


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)
        

def norm_viirs(viirs_data):
    # shape of viirs_data: (3, 120, 160)
    # min-max normalization for VIIRS data
    viirs_data[0, :, :] = (viirs_data[0, :, :] - 220.000) / (350.000 - 220.000) 
    viirs_data[1, :, :] = (viirs_data[1, :, :] - 220.000) / (320.000 - 220.000) 
    viirs_data[2, :, :] = (viirs_data[2, :, :] - (0.000)) / (8.000 - (0.000))
    return viirs_data

def norm_goes(goes_data):
    # min-max normalization for GOES data
    goes_data[0, :, :] = (goes_data[0, :, :] - 220.000) / (350.000 - 220.000)
    goes_data[1, :, :] = (goes_data[1, :, :] - 220.000) / (320.000 - 220.000)
    goes_data[2, :, :] = (goes_data[2, :, :] - (0.000)) / (8.000 - (0.000))
    return goes_data

def denorm_viirs(sr_viirs_data):
    # min-max de-normalization for SR-VIIRS data
    sr_viirs_data[:, 0] = sr_viirs_data[:, 0] * (350.000 - 220.000) + 220.000
    sr_viirs_data[:, 1] = sr_viirs_data[:, 1] * (320.000 - 220.000) + 220.000
    sr_viirs_data[:, 2] = sr_viirs_data[:, 2] * (8.000 - (0.000)) + (0.000)
    return sr_viirs_data

def denorm_goes(sr_goes_data):
    # min-max de-normalization for SR-GOES data
    sr_goes_data[:, 0] = sr_goes_data[:, 0] * (350.000 - 220.000) + 220.000
    sr_goes_data[:, 1] = sr_goes_data[:, 1] * (320.000 - 220.000) + 220.000
    sr_goes_data[:, 2] = sr_goes_data[:, 2] * (8.000 - (0.000)) + (0.000)
    return sr_goes_data

    
# def plot_example(viirs, goes, sr_viirs, interpolation, batches_done, channel_name, vmin=None, vmax=None):
    
    # # TODO: add an arg for savefigure path
    # fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # im1 = axes[0, 0].imshow(viirs, cmap='Spectral_r', vmin=vmin, vmax=vmax)
    # axes[0, 0].set_title(f'VIIRS {channel_name}')
    # plt.colorbar(im1, ax=axes[0, 0], fraction=0.04, pad=0.025)

    # im2 = axes[0, 1].imshow(goes, cmap='Spectral_r', vmin=vmin, vmax=vmax)
    # axes[0, 1].set_title(f'GOES {channel_name}')
    # plt.colorbar(im2, ax=axes[0, 1], fraction=0.04, pad=0.025)

    # im3 = axes[1, 0].imshow(sr_viirs, cmap='Spectral_r', vmin=vmin, vmax=vmax)
    # axes[1, 0].set_title(f'SR {channel_name}')
    # plt.colorbar(im3, ax=axes[1, 0], fraction=0.04, pad=0.025)

    # im4 = axes[1, 1].imshow(interpolation, cmap='Spectral_r', vmin=vmin, vmax=vmax)
    # axes[1, 1].set_title(f'Interpolation {channel_name}')
    # plt.colorbar(im4, ax=axes[1, 1], fraction=0.04, pad=0.025)

    # Save figure
    # plt.savefig(f"/content/drive/MyDrive/Projects/GOES_Super_Resolution_HW/Code/FireGAN_HW/Images/Train/{batches_done}_{channel_name}_{Date}.png", format='png', bbox_inches='tight', facecolor=(1, 1, 1, 0))

    plt.close()
    
# import cartopy.feature as cf    
# def plot_example_afi(viirs, goes, sr, interpolation, batches_done, channel_name, abox, vmin=None, vmax=None, facecolor=None, cf=cf):

#     # Select colormap based on channel_name
#     colormap = 'bwr' if channel_name == "AFI" else 'Spectral_r'

#     # Load the states and provinces features
#     states_provinces = cf.NaturalEarthFeature(
#         category='cultural',
#         name='admin_1_states_provinces_lines',
#         scale='50m',
#         facecolor='none'
#     )

#     fig, axs = plt.subplots(2, 2, figsize=[10, 8], subplot_kw={'projection': ccrs.PlateCarree()})

#     data_dict = {
#         "VIIRS": viirs,
#         "GOES": goes,
#         "SR-VIIRS": sr,
#         "Interpolation-VIIRS": interpolation
#     }

#     lon = np.linspace(start=abox[0], stop=abox[2], num=viirs.shape[1])
#     lat = np.linspace(start=abox[1], stop=abox[3], num=viirs.shape[0])
#     lon, lat = np.meshgrid(lon, lat)

#     for ax, (key, data) in zip(axs.flat, data_dict.items()):
#         ax.set_title(f"{key}-{channel_name}")
#         ax.add_feature(states_provinces, edgecolor='black', linewidth=2.5, zorder=3)
#         ax.add_feature(cf.LAND)
#         ax.add_feature(cf.OCEAN, zorder=3)
#         ax.add_feature(cf.COASTLINE, linewidth=2.5)
#         ax.add_feature(cf.BORDERS, linestyle=':')

#         cm = ax.pcolormesh(lon, lat, np.flipud(data), cmap=colormap, transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)

#         gls = ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), linestyle='--', zorder=4)
#         ax.coastlines()

#         x = np.round(np.linspace(lon.min(), lon.max(), 5), 2)
#         y = np.round(np.linspace(lat.min(), lat.max(), 4), 2)
#         gls.xlocator = mticker.FixedLocator(x)
#         gls.ylocator = mticker.FixedLocator(y)
#         ax.set_xticks(x)
#         ax.set_yticks(y)
#         ax.set_xlim(lon.min(), lon.max())
#         ax.set_ylim(lat.min(), lat.max())
#         ax.set_xlabel('Longitude')
#         ax.set_ylabel('Latitude')

#     # Add colorbar
#     cbaxes = fig.add_axes([0.92, 0.12, 0.01, 0.75])  # [left, bottom, width, height]
#     cb = fig.colorbar(cm, cax=cbaxes)
#     cb.set_label(channel_name, fontsize=14)
    
#     year = viirs_file.split('.')[0]
#     doy = viirs_file.split('.')[1]
    
#     # Save figure
#     plt.savefig(f"/content/drive/MyDrive/Projects/GOES_Super_Resolution_HW/Plots/ESRGAN_Output/{batches_done}.{channel_name}.{year}.{doy}.png", format='png', bbox_inches='tight', facecolor=(1, 1, 1, 0))
#     plt.close()
    
    
    import glob
import random
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from PIL import Image
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cf

import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.mpl.ticker as cticker



def plot_function(viirs, goes, sr, interpolation, ftype):

    if ftype == "AFI":
        colormap = 'bwr'
        vmin = -1
        vmax = 6
    else:
        colormap = 'Spectral_r'
        vmin = 270
        vmax = 350

    states_provinces = cf.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')


    afi_path = '/content/drive/MyDrive/Projects/GOES_Super_Resolution_HW/Data/Images/Train/2022.221.HW.h5'

    abox = [-156.75, 20.45, -155.95, 21.05]

    fig = plt.figure(figsize=[10, 8])
    ax1 = fig.add_subplot(2, 2, 1, projection=ccrs.PlateCarree())
    ax2 = fig.add_subplot(2, 2, 2, projection=ccrs.PlateCarree())
    ax3 = fig.add_subplot(2, 2, 3, projection=ccrs.PlateCarree())
    ax4 = fig.add_subplot(2, 2, 4, projection=ccrs.PlateCarree())

    # ---------
    #   VIIRS
    # ---------

    lon = np.linspace(start=abox[0], stop=abox[2], num=viirs.shape[1])
    lat = np.linspace(start=abox[1], stop=abox[3], num=viirs.shape[0])
    lon, lat = np.meshgrid(lon, lat)

    ax1.set_title("VIIRS-" + ftype)

    ax1.add_feature(states_provinces, edgecolor='black', linewidth=2.5, zorder=3)
    ax1.add_feature(cf.LAND)
    ax1.add_feature(cf.OCEAN, zorder=3)
    ax1.add_feature(cf.COASTLINE, linewidth=2.5)
    ax1.add_feature(cf.BORDERS, linestyle=':')

    ax1.pcolormesh(lon, lat, np.flipud(viirs), cmap=colormap, transform=ccrs.PlateCarree(), vmax=vmax, vmin=vmin)

    gls = ax1.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), linestyle='--', zorder=4)

    ax1.coastlines()

    x = np.round(np.linspace(lon.min(), lon.max(), 5), 2)
    y = np.round(np.linspace(lat.min(), lat.max(), 4), 2)

    gls.xlocator = mticker.FixedLocator(x)
    gls.ylocator = mticker.FixedLocator(y)

    ax1.set_xticks(x)
    ax1.set_yticks(y)
    ax1.set_xlim(lon.min(), lon.max())
    ax1.set_ylim(lat.min(), lat.max())
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')

    # ---------
    #  GOES
    # ---------

    lon = np.linspace(start=abox[0], stop=abox[2], num=goes.shape[1])
    lat = np.linspace(start=abox[1], stop=abox[3], num=goes.shape[0])
    lon, lat = np.meshgrid(lon, lat)

    ax2.set_title("GOES-" + ftype)

    ax2.add_feature(states_provinces, edgecolor='black', linewidth=2.5, zorder=3)
    ax2.add_feature(cf.LAND)
    ax2.add_feature(cf.OCEAN, zorder=3)
    ax2.add_feature(cf.COASTLINE, linewidth=2.5)
    ax2.add_feature(cf.BORDERS, linestyle=':')

    ax2.pcolormesh(lon, lat, np.flipud(goes), cmap=colormap, transform=ccrs.PlateCarree(), vmax=vmax, vmin=vmin)

    gls = ax2.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), linestyle='--', zorder=4)

    ax2.coastlines()

    x = np.round(np.linspace(lon.min(), lon.max(), 5), 2)
    y = np.round(np.linspace(lat.min(), lat.max(), 4), 2)

    gls.xlocator = mticker.FixedLocator(x)
    gls.ylocator = mticker.FixedLocator(y)

    ax2.set_xticks(x)
    ax2.set_yticks(y)
    ax2.set_xlim(lon.min(), lon.max())
    ax2.set_ylim(lat.min(), lat.max())
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')

    # ---------
    #    SR
    # ---------

    lon = np.linspace(start=abox[0], stop=abox[2], num=viirs.shape[1])
    lat = np.linspace(start=abox[1], stop=abox[3], num=viirs.shape[0])
    lon, lat = np.meshgrid(lon, lat)


    ax3.set_title("SR-GOES-" + ftype)

    ax3.add_feature(states_provinces, edgecolor='black', linewidth=2.5, zorder=3)
    ax3.add_feature(cf.LAND)
    ax3.add_feature(cf.OCEAN, zorder=3)
    ax3.add_feature(cf.COASTLINE, linewidth=2.5)
    ax3.add_feature(cf.BORDERS, linestyle=':')

    ax3.pcolormesh(lon, lat, np.flipud(sr), cmap=colormap, transform=ccrs.PlateCarree(), vmax=vmax, vmin=vmin)

    gls = ax3.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), linestyle='--', zorder=4)

    ax3.coastlines()

    x = np.round(np.linspace(lon.min(), lon.max(), 5), 2)
    y = np.round(np.linspace(lat.min(), lat.max(), 4), 2)

    gls.xlocator = mticker.FixedLocator(x)
    gls.ylocator = mticker.FixedLocator(y)

    ax3.set_xticks(x)
    ax3.set_yticks(y)
    ax3.set_xlim(lon.min(), lon.max())
    ax3.set_ylim(lat.min(), lat.max())
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')

    # ---------
    #  Interpo
    # ---------

    ax4.set_title("Interpolation-GOES-" + ftype)

    ax4.add_feature(states_provinces, edgecolor='black', linewidth=2.5, zorder=3)
    ax4.add_feature(cf.LAND)
    ax4.add_feature(cf.OCEAN, zorder=3)
    ax4.add_feature(cf.COASTLINE, linewidth=2.5)
    ax4.add_feature(cf.BORDERS, linestyle=':')

    cm = ax4.pcolormesh(lon, lat, np.flipud(interpolation), cmap=colormap, transform=ccrs.PlateCarree(), vmax=vmax, vmin=vmin)

    gls = ax4.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), linestyle='--', zorder=4)

    ax4.coastlines()

    x = np.round(np.linspace(lon.min(), lon.max(), 5), 2)
    y = np.round(np.linspace(lat.min(), lat.max(), 4), 2)

    gls.xlocator = mticker.FixedLocator(x)
    gls.ylocator = mticker.FixedLocator(y)

    ax4.set_xticks(x)
    ax4.set_yticks(y)
    ax4.set_xlim(lon.min(), lon.max())
    ax4.set_ylim(lat.min(), lat.max())
    ax4.set_xlabel('Longitude')
    ax4.set_ylabel('Latitude')

    cbaxes = fig.add_axes([0.92, 0.12, 0.01, 0.75])  # [left, bottom, width, height]
    cb = fig.colorbar(cm, cax=cbaxes)
    cb.set_label(ftype, fontsize=14)

    # save fig
    figpath = '/content/drive/MyDrive/Projects/GOES_Super_Resolution_HW/Data/FireGAN_HW/ESRGAN_Output'

    plt.savefig(figpath + 'NonFire_' + ftype + ".png", format='png', dpi=600,
            bbox_inches='tight', facecolor=(1, 1, 1, 0))
    plt.close();


import torch
from torch.autograd import Variable
import torch.nn as nn
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as mticker
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
    
def calculate_rmse(pred, gt):
    if pred.ndim == 2:
        pred = pred[np.newaxis, :, :]
    if gt.ndim == 2:
        gt = gt[np.newaxis, :, :]
    
    if pred.shape[0] == 1:
        pred = np.repeat(pred, 3, axis=0)
    if gt.shape[0] == 1:
        gt = np.repeat(gt, 3, axis=0)
    pred = norm_viirs(np.array(pred))
    gt = norm_viirs(np.array(gt))
    return np.sqrt(np.mean((pred - gt) ** 2))

def calculate_r2(pred, gt):
    if pred.ndim == 2:
        pred = pred[np.newaxis, :, :]
    if gt.ndim == 2:
        gt = gt[np.newaxis, :, :]
    
    if pred.shape[0] == 1:
        pred = np.repeat(pred, 3, axis=0)
    if gt.shape[0] == 1:
        gt = np.repeat(gt, 3, axis=0)
    
    pred = norm_viirs(pred)
    gt = norm_viirs(gt)
    pred = np.array(pred)
    gt = np.array(gt)
    ss_res = np.sum((gt - pred) ** 2)
    ss_tot = np.sum((gt - np.mean(gt)) ** 2)
    # print(f"ss_res: {ss_res}, ss_tot: {ss_tot}") 
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0



# def calculate_psnr(pred, gt):
#     # Ensure images are in the same shape
#     if pred.ndim == 2:
#         pred = pred[np.newaxis, :, :]
#     if gt.ndim == 2:
#         gt = gt[np.newaxis, :, :]

#     # Replicate grayscale to RGB if necessary
#     if pred.shape[0] == 1:
#         pred = np.repeat(pred, 3, axis=0)
#     if gt.shape[0] == 1:
#         gt = np.repeat(gt, 3, axis=0)

#     # Normalize images if necessary (assuming norm_viirs normalizes to range [0, 1])
#     pred = norm_viirs(np.array(pred))
#     gt = norm_viirs(np.array(gt))

#     # Calculate Mean Squared Error (MSE)
#     mse = np.mean((pred - gt) ** 2)

#     # Handle perfect match (MSE = 0)
#     if mse == 0:
#         return float('inf')

#     # Calculate Peak Signal-to-Noise Ratio (PSNR)
#     max_pixel = gt.max()  # Assuming pixel values are in range [0, 1]
#     psnr_value = 10 * np.log10((max_pixel ** 2) / mse)

#     return psnr_value

def calculate_psnr(pred, gt):

    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()
    max_val = np.max(gt.max())
  

    mse = np.mean((pred - gt) ** 2)
    
    if mse == 0:
        return float('inf')
    
    psnr = 10 * np.log10(max_val**2 / mse)
    
    return psnr


def calculate_ssim(pred, gt):
    if pred.ndim == 2:
        pred = pred[np.newaxis, :, :]
    if gt.ndim == 2:
        gt = gt[np.newaxis, :, :]
    
    if pred.shape[0] == 1:
        pred = np.repeat(pred, 3, axis=0)
    if gt.shape[0] == 1:
        gt = np.repeat(gt, 3, axis=0)
    
    pred = norm_viirs(np.array(pred))
    gt = norm_viirs(np.array(gt))

    win_size = min(pred.shape[1], pred.shape[2], 7)
    # data_range = gt.max() - gt.min()
    # print(data_range)

    c1 = (0.01 * 1) ** 2
    c2 = (0.03 * 1) ** 2
    return ssim(pred, gt, data_range=gt.max() - gt.min(), channel_axis=0, C1=c1, C2=c2)

    
    