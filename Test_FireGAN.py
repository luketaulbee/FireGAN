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
import pywt

workpath = r''

import sys
sys.path.append(workpath)
from models import GeneratorRRDB
from models import *

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class HyperParameters:
    def __init__(self):
        self.checkpoint_model = os.path.join(workpath, 'Models', 'gen_1e3lr_newMAXmin_24000.pth')
        self.channels = 3
        self.residual_blocks = 32

opt = HyperParameters()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks).to(device)
generator.load_state_dict(torch.load(opt.checkpoint_model, weights_only=True))
generator.eval()

x_mask_path = os.path.join(workpath, 'Data', 'Masks', 'Mask.2250m.npy')
x_mask = np.load(x_mask_path)
x_mask = torch.tensor(x_mask, dtype=torch.float32, device=device)
x_mask = Variable(x_mask)

y_mask_path = os.path.join(workpath, 'Data', 'Masks', 'Mask.375m.npy')
y_mask = np.load(y_mask_path)
y_mask = torch.tensor(y_mask, dtype=torch.float32, device=device)
y_mask = Variable(y_mask)

output_dirs = [
    os.path.join(workpath, 'Test_Outputs', 'Test_Interpolation_HW'),
    os.path.join(workpath, 'Test_Outputs', 'Test_Outputs_HW'),
    os.path.join(workpath, 'Test_Outputs', 'Plots'),
]
for output_dir in output_dirs:
    os.makedirs(output_dir, exist_ok=True)

test_dir = os.path.join(workpath, 'Data', 'Test1')
filelist = os.listdir(test_dir)
filelist.sort()
len_file = len(filelist)



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


# Mean gradient magnitude (Grad-Mag) -- sharpness metric 
def MeanGradientMagnitude(img):
    # Compute gradients in x and y directions
    grad_x = np.gradient(img, axis=1)
    grad_y = np.gradient(img, axis=0)
    
    # Compute gradient magnitude
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Return mean gradient magnitude
    return np.mean(grad_mag)

# Gradient TV (Grad-TV) -- sharpness metric 
def GradientTV(img):
    # Compute gradients in x and y directions
    grad_x = np.gradient(img, axis=1)
    grad_y = np.gradient(img, axis=0)
    
    # Compute total variation
    tv = np.sum(np.sqrt(grad_x**2 + grad_y**2))
    
    return tv

# Fourier-RMSE [bivariate] -- Sharpness metrics in spectral space
def FourierRMSE(pred, gt):
    # Compute the 2D Fourier Transform of both images
    pred_fft = np.fft.fft2(pred)
    gt_fft = np.fft.fft2(gt)
    
    # Compute the magnitude of the Fourier coefficients
    pred_mag = np.abs(pred_fft)
    gt_mag = np.abs(gt_fft)
    
    # Compute RMSE in the spectral domain
    rmse = np.sqrt(np.mean((pred_mag - gt_mag) ** 2))
    
    return rmse

# Wavelet-TV [univariate] -- Sharpness metrics in spectral space
def WaveletTV(img):
    
    # Perform wavelet decomposition
    coeffs = pywt.wavedec2(img, 'db1', level=1)
    
    # Compute total variation of the wavelet coefficients
    tv = 0
    for coeff in coeffs[1:]:  # Skip the approximation coefficients
        for detail in coeff:
            grad_x = np.gradient(detail, axis=1)
            grad_y = np.gradient(detail, axis=0)
            tv += np.sum(np.sqrt(grad_x**2 + grad_y**2))
    
    return tv

fontsize = 17 
rmse_list = []
r2_list = []
psnr_list = []
ssim_list = []


def compute_metrics(pred_img, gt_img):
    """Return a dict of all metrics for one image pair."""
    return {
        'grad_mag':    MeanGradientMagnitude(pred_img),
        'grad_tv':     GradientTV(pred_img),
        'fourier_rmse': FourierRMSE(pred_img, gt_img),
        'wavelet_tv':  WaveletTV(pred_img),
    }

def print_metrics(label, metrics_dict, indent=0):
    pad = ' ' * indent
    print(f"{pad}{label}")
    for k, v in metrics_dict.items():
        if isinstance(v, str):
            print(f"{pad}  {k:<20s}: {v}")
        else:
            try:
                print(f"{pad}  {k:<20s}: {v:.4f}")
            except Exception:
                # fallback to generic str conversion if formatting fails
                print(f"{pad}  {k:<20s}: {v}")

all_metrics = []          # list of dicts, one per image

for i in range(len_file):
    filepath = os.path.join(test_dir, filelist[i])
    file = h5py.File(filepath, 'r')
    tmp_lr = norm_goes(np.array(file['goes'][:]))
    tmp_hr = norm_viirs(np.array(file['viirs'][:]))

    tmp_lr = torch.tensor(tmp_lr, dtype=torch.float32, device=device).reshape((1, 3, 27, 35))
    tmp_hr = torch.tensor(tmp_hr, dtype=torch.float32, device=device).reshape((1, 3, 162, 210))

    imgs_lr = Variable(tmp_lr.type(Tensor)).to(device)
    imgs_hr = Variable(tmp_hr.type(Tensor)).to(device)

    gen_hr  = generator(imgs_lr)
    intero  = nn.functional.interpolate(imgs_lr, scale_factor=4)

    gen_hr = denorm_viirs(gen_hr)
    imgs_lr = denorm_goes(imgs_lr)
    imgs_hr = denorm_viirs(imgs_hr)
    intero  = denorm_goes(intero)

    x_mask_tensor = x_mask.clone()
    y_mask_tensor = y_mask.clone()
    x_mask_tensor[x_mask == 0] = float('nan')
    y_mask_tensor[y_mask == 0] = float('nan')

    imgs_lr = imgs_lr[:, :, :27,  :]
    intero  = intero [:, :, :162, :]
    imgs_hr = imgs_hr[:, :, :162, :]
    gen_hr  = gen_hr [:, :, :162, :]

    pred_np = gen_hr[0].cpu().detach().numpy()   # (C, H, W)
    gt_np   = imgs_hr[0].cpu().detach().numpy()

    m = compute_metrics(pred_np, gt_np)
    m['filename'] = filelist[i]          # keep provenance
    all_metrics.append(m)

    print_metrics(f"[{i+1:03d}/{len_file}] {filelist[i]}", m, indent=2)
    plt.close()

metric_keys = ['grad_mag', 'grad_tv', 'fourier_rmse', 'wavelet_tv']

global_metrics = {
    k: {
        'mean':   np.mean([m[k] for m in all_metrics]),
        'std':    np.std ([m[k] for m in all_metrics]),
        'min':    np.min ([m[k] for m in all_metrics]),
        'max':    np.max ([m[k] for m in all_metrics]),
    }
    for k in metric_keys
}

print("\n" + "=" * 55)
print("GLOBAL METRICS")
print("=" * 55)
print(f"{'Metric':<20s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s}")
print("-" * 55)
for k, stats in global_metrics.items():
    print(f"{k:<20s} {stats['mean']:8.4f} {stats['std']:8.4f} "
          f"{stats['min']:8.4f} {stats['max']:8.4f}")
print("=" * 55)
