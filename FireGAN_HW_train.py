

def plot_example(viirs, goes, sr_viirs, interpolation, batches_done, channel_name, vmin=None, vmax=None, cmap=None):

    # TODO: add an arg for savefigure path
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    im1 = axes[0, 0].imshow(viirs, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(f'VIIRS {channel_name}')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.04, pad=0.025)

    im2 = axes[0, 1].imshow(goes, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f'GOES {channel_name}')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.04, pad=0.025)

    im3 = axes[1, 0].imshow(sr_viirs, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1, 0].set_title(f'SR {channel_name}')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.04, pad=0.025)

    im4 = axes[1, 1].imshow(interpolation, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1, 1].set_title(f'Interpolation {channel_name}')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.04, pad=0.025)

    #Save figure
    plt.savefig(f"{workpath}training_experimental\\plot\\{batches_done}.{train_filelist[i]}.{channel_name}.png", format='png', bbox_inches='tight', facecolor=(1, 1, 1, 0))

    plt.close()

import sys
import datetime as dt
import os
from random import shuffle
import h5py
from torchvision.transforms import GaussianBlur
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch
import torch.nn as nn
import cartopy.feature as cf
# import cv2

workpath = 'C:\\Users\\lhtau\\OneDrive\\Desktop\\Research\\FireGAN_HW\\'

import sys
sys.path.append(workpath)
from models import *

# monitor start time

os.makedirs(workpath + "/Images", exist_ok=True)
os.makedirs(workpath + "/Saved_Models", exist_ok=True)

# ----------
#  Training
# ----------

x_mask = np.load(workpath + 'Masks/Mask.2250m.npy')
y_mask = np.load(workpath + 'Masks/Mask.375m.npy')


train_filelist = os.listdir(workpath + 'Train1/') 

train_filelist.sort()
len_train = len(train_filelist)

start_time = dt.datetime.now()
print('Start learning at {}'.format(str(start_time)))

os.makedirs(workpath + "/Images", exist_ok=True)
os.makedirs(workpath + "/Saved_Models", exist_ok=True)

# ------------------
#  Hyper-Parameters
# ------------------
# ------------------

class HpyerParameters:
    def __init__(self):
        self.load_model = 0             # load previously trained model
        self.n_epochs = 301            # number of epochs of training  
        self.lr = 0.00001          # adam: learning rate
        # self.weight_decay = 1e-5        # adam: weight decay (L2 regularization)
        # self.l1_lambda = 1e-5           # L1 regularization weight
        self.b1 = 0.9                   # adam: decay of first order momentum of gradient
        self.b2 = 0.999                 # adam: decay of second order momentum of gradient
        self.n_cpu = 0                  # number of cpu threads to use during batch generation
        self.hr_height = 210 #160            # high res. image height
        self.hr_width = 210 #160             # high res. image width
        self.channels = 3               # number of image channels
        self.sample_interval = 10        # interval between saving image samples
        self.checkpoint_interval = 10    # epoch interval between model checkpoints
        self.residual_blocks = 32        # number of residual blocks in the generator
        # self.lambda_adv = 5e-3             # adversarial loss weight
        # self.lambda_pixel = 1e-2          # pixel-wise loss weight

opt = HpyerParameters()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

hr_shape = (opt.hr_height, opt.hr_width)

# Initialize generator and discriminator
generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks).to(device)
discriminator = Discriminator(input_shape=(opt.channels, *hr_shape)).to(device)
feature_extractor = FeatureExtractor().to(device)

# Set feature extractor to inference mode
feature_extractor.eval()

# Losses
criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
criterion_content = torch.nn.MSELoss().to(device)
criterion_pixel = torch.nn.MSELoss().to(device)

if opt.load_model:
   generator.load_state_dict(torch.load(workpath + "training_experimental/Saved_Models/gen_1e3lr_newMAXmin_24000.pth"))  # gen_1e3lr_newMAXmin_18000
   discriminator.load_state_dict(torch.load(workpath + "training_experimental/Saved_Models/disc_1e3lr_newMAXmin_24000.pth"))  #disc_1e3lr_newMAXmin_18000


optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor


# ----------
#  Training
# ----------
train_filelist = os.listdir(workpath + 'Train1/') 

train_filelist.sort()
len_train = len(train_filelist)
for epoch in range(0, opt.n_epochs):

    shuffle(train_filelist)

    for i in range(len_train):
        batches_done = epoch * len_train + i

        # Configure model input
        file = h5py.File(workpath + 'Train1/' + train_filelist[i], 'r') 
        tmp_lr = np.array(file['goes'][:])
        tmp_hr = np.array(file['viirs'][:])
        file.close()

        tmp_lr = norm_goes(tmp_lr)
        tmp_hr = norm_viirs(tmp_hr)

        tmp_lr = Tensor(tmp_lr)
        tmp_hr = Tensor(tmp_hr)

        tmp_lr = torch.reshape(tmp_lr, (1, 3, 27, 35))
        tmp_hr = torch.reshape(tmp_hr, (1, 3, 162, 210))

        # covert to pytorch tensor
        imgs_lr = Variable(tmp_lr.type(Tensor))
        imgs_hr = Variable(tmp_hr.type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

        # ------------------
        #  Train Generator
        # ------------------

        optimizer_G.zero_grad()

        gen_hr = generator(imgs_lr)
        loss_pixel = criterion_pixel(gen_hr, imgs_hr)

        pred_real = discriminator(imgs_hr).detach()
        pred_fake = discriminator(gen_hr)

        valid = Variable(Tensor(np.ones(pred_real.shape)), requires_grad=False)
        fake = Variable(Tensor(np.zeros(pred_fake.shape)), requires_grad=False)

        loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)

        # Content loss
        gen_features = feature_extractor(gen_hr[:,:3])
        real_features = feature_extractor(imgs_hr[:,:3]).detach()
        loss_content = criterion_content(gen_features, real_features)

        # Total generator loss
        loss_G = loss_content + 0.3 * loss_GAN + 150 * loss_pixel 

        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        pred_real = discriminator(imgs_hr)
        pred_fake = discriminator(gen_hr.detach())

        loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
        loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

        loss_D = loss_real + loss_fake

        loss_D.backward()
        optimizer_D.step()

    # -----------------------
    #  Log Progress per Epoch
    # -----------------------

    print(
        "[Epoch %d/%d] [D loss: %f] [G loss: %f, content: %f, adv: %f, pixel: %f]"
        % (
            epoch,
            opt.n_epochs,
            loss_D.item(),
            loss_G.item(),
            loss_content.item(),
            loss_GAN.item(),
            loss_pixel.item()
        )
    )

    if epoch % opt.sample_interval == 0:
        intero = nn.functional.interpolate(imgs_lr, scale_factor=4)

        imgs_lr = denorm_goes(imgs_lr)
        intero = denorm_goes(intero)
        imgs_hr = denorm_viirs(imgs_hr)
        gen_hr = denorm_viirs(gen_hr)

        x_mask_tensor = Tensor(x_mask)
        y_mask_tensor = Tensor(y_mask)

        # Replace zeros with NaN in masks
        x_mask_tensor[x_mask == 0] = float('nan')
        y_mask_tensor[y_mask == 0] = float('nan')

        # plot_example((imgs_hr[0, 2].cpu()), (imgs_lr[0, 2].cpu()), (gen_hr[0, 2].cpu().detach().numpy()), intero[0, 2].cpu(), epoch, "AFI", cmap='bwr')
        # plot_example((imgs_hr[0, 1].cpu()), (imgs_lr[0, 1].cpu()), (gen_hr[0, 1].cpu().detach().numpy()), intero[0, 0].cpu(), epoch, "BT07", cmap='Spectral_r')
        # plot_example((imgs_hr[0, 0].cpu()), (imgs_lr[0, 0].cpu()), (gen_hr[0, 0].cpu().detach().numpy()), intero[0, 1].cpu(), epoch, "BT14", cmap='Spectral_r')

        if epoch % opt.checkpoint_interval == 0:
              torch.save(generator.state_dict(), workpath + "training_experimental/Saved_Models/riviera_g001.pth")
              torch.save(discriminator.state_dict(), workpath + "training_experimental/Saved_Models/riviera_d001.pth")


        print(f"[Epoch {epoch}/{opt.n_epochs}] [File {i}/{len_train}] [D loss: {loss_D.item()}] [G loss: {loss_G.item()}]")

# monitor end time
end_time = dt.datetime.now()
print('Stop learning {}'.format(str(end_time)))
elapsed_time = end_time - start_time
print('Elapsed learning {}'.format(str(elapsed_time)))