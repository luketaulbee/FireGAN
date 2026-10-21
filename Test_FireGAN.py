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

workpath = r'C:\Users\lhtau\OneDrive\Desktop\FireGAN_LukeTaulbee'

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

# for i in range(len_file):
#     file = h5py.File(workpath + 'Test1/' + filelist[i], 'r')



#     tmp_lr = np.array(file['goes'][:]) 
#     tmp_hr = np.array(file['viirs'][:])
#     # file.close()

#     tmp_lr = norm_goes(tmp_lr)  
#     tmp_hr = norm_viirs(tmp_hr)  

#     # pad_lr = np.zeros((3, 27, 35))
#     # pad_hr = np.zeros((3, 162, 210))

#     # pad_lr[:, :27, :35] = tmp_lr
#     # pad_hr[:, :162, :210] = tmp_hr

#     # tmp_lr = Tensor(pad_lr)
#     # tmp_hr = Tensor(pad_hr)

#     # tmp_lr = torch.reshape(tmp_lr, (1, 3, 27, 35))
#     # tmp_hr = torch.reshape(tmp_hr, (1, 3, 162, 210))

#     # imgs_lr = Variable(tmp_lr.type(Tensor)).to(device)
#     # imgs_hr = Variable(tmp_hr.type(Tensor)).to(device)

#     tmp_lr = Tensor(tmp_lr)
#     tmp_hr = Tensor(tmp_hr)
#     tmp_lr = torch.reshape(tmp_lr, (1, 3, 27, 35))
#     tmp_hr = torch.reshape(tmp_hr, (1, 3, 162, 210))
#     imgs_lr = Variable(tmp_lr.type(Tensor)).to(device)
#     imgs_hr = Variable(tmp_hr.type(Tensor)).to(device)

#     gen_hr = generator(imgs_lr)

#     intero = nn.functional.interpolate(imgs_lr, scale_factor=4)

#     gen_hr = denorm_viirs(gen_hr)
#     tmp_lr = denorm_goes(tmp_lr)
#     tmp_hr = denorm_viirs(tmp_hr)
#     intero = denorm_goes(intero)

#     np.save(workpath + 'Test_Outputs/Test_Interpolation_HW/' + filelist[i][:-3] + '.npy', intero[0].cpu().detach().numpy())
#     np.save(workpath + 'Test_Outputs/Test_Outputs_HW/' + filelist[i][:-3] + '.npy', gen_hr[0].cpu().detach().numpy())

#     # Apply masking if needed
#     x_mask_tensor = Tensor(x_mask)
#     y_mask_tensor = Tensor(y_mask)

#     x_mask_tensor[x_mask == 0] = float('nan')
#     y_mask_tensor[y_mask == 0] = float('nan')

#     imgs_lr = imgs_lr[:, :, :27, :]
#     intero = intero[:, :, :162, :]
#     imgs_hr = imgs_hr[:, :, :162, :]
#     gen_hr = gen_hr[:, :, :162, :]

#     tmp_lr = imgs_lr[:, :, :27, :]
#     tmp_hr = imgs_hr[:, :, :162, :]

#     fig, axs = plt.subplots(3, 3, figsize=(16, 14), subplot_kw={'projection': ccrs.PlateCarree()})

#     vmin_af, vmax_af = -1, 5
#     vmin_lwir, vmax_lwir = 270, 320
#     vmin_swir, vmax_swir = 270, 320

#     pad = 0.07
#     fraction = 0.035

#     bbox = [-156.75, 20.45, -155.95, 21.05]
#     extent = [bbox[0], bbox[2], bbox[1], bbox[3]]

#     images = [
#         (tmp_lr[0, 2].cpu().numpy(), 'bwr', vmin_af, vmax_af, "(a) GOES AFI", 0, 0),
#         (tmp_hr[0, 2].cpu().numpy(), 'bwr', vmin_af, vmax_af, "(d) VIIRS AFI", 1, 0),
#         (gen_hr[0, 2].cpu().detach().numpy(), 'bwr', vmin_af, vmax_af, "(g) Downscaled AFI", 2, 0),
#         (tmp_lr[0, 1].cpu().numpy(), 'Spectral_r', vmin_lwir, vmax_lwir, "(b) GOES C07 BT", 0, 1),
#         (tmp_hr[0, 1].cpu().numpy(), 'Spectral_r', vmin_lwir, vmax_lwir, "(e) VIIRS I04 BT", 1, 1),
#         (gen_hr[0, 1].cpu().detach().numpy(), 'Spectral_r', vmin_lwir, vmax_lwir, "(h) Downscaled C07 BT", 2, 1),
#         (tmp_lr[0, 0].cpu().numpy(), 'Spectral_r', vmin_swir, vmax_swir, "(c) GOES C14 BT", 0, 2),
#         (tmp_hr[0, 0].cpu().numpy(), 'Spectral_r', vmin_swir, vmax_swir, "(f) VIIRS I05 BT", 1, 2),
#         (gen_hr[0, 0].cpu().detach().numpy(), 'Spectral_r', vmin_swir, vmax_swir, "(i) Downscaled C14 BT", 2, 2)
#     ]


#     for data, cmap, vmin, vmax, title, row, col in images:
#         ax = axs[row, col]
#         ax.set_extent(extent, crs=ccrs.PlateCarree())
#         ax.add_feature(cfeature.OCEAN, zorder=2)
#         ax.add_feature(cfeature.COASTLINE, linewidth=2.5)
#         ax.add_feature(cfeature.BORDERS, linestyle=':')

#         gl = ax.gridlines(
#             crs=ccrs.PlateCarree(),
#             draw_labels=False,
#             linewidth=1,
#             color='black',
#             linestyle='--'
#         )
        
#         x_ticks = np.linspace(bbox[0], bbox[2], 5)  
#         y_ticks = np.linspace(bbox[1], bbox[3], 4) 
        
#         gl.xlocator = mticker.FixedLocator(x_ticks)
#         gl.ylocator = mticker.FixedLocator(y_ticks)

#         ax.set_title(title, fontsize=fontsize)

#         # gl.top_labels = False  
#         # gl.right_labels = False  
#         # gl.bottom_labels = True  
#         # gl.left_labels = True   
        

#         ax.set_xticks(x_ticks)  
#         ax.set_yticks(y_ticks)  
#         ax.set_xlim(bbox[0], bbox[2])  
#         ax.set_ylim(bbox[1], bbox[3])  

#         im = ax.imshow(
#             data, 
#             cmap=cmap, 
#             vmin=vmin, 
#             vmax=vmax, 
#             extent=extent, 
#             transform=ccrs.PlateCarree(),
#             rasterized=True 
#         )
#         ax.set_title(title)
#         # fig.colorbar(im, ax=ax, orientation='vertical', fraction=fraction, pad=pad)

#         # Define the colorbar label based on the colormap used
#         if cmap == 'bwr':
#             cbar_label = "AFI"
#         else:
#             cbar_label = "Brightness Temperature (°K)"

#         cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=fraction, pad=pad)
#         cbar.set_label(cbar_label, fontsize=12)


#         # # Metrics 
#         # if row == 2:  
#         #     # Use gen_hr for comparison with tmp_hr
#         #     rmse = calculate_rmse(gen_hr[0, col].cpu().detach().numpy(), tmp_hr[0, col].cpu().numpy())
#         #     r2 = calculate_r2(gen_hr[0, col].cpu().detach().numpy(), tmp_hr[0, col].cpu().numpy())

#         #     psnr_value = calculate_psnr(gen_hr[0, col].cpu().detach().numpy(), tmp_hr[0, col].cpu().numpy())

#         #     ssim_value = calculate_ssim(gen_hr[0, col].cpu().detach().numpy(), tmp_hr[0, col].cpu().numpy())
#         #     # ax.text(0.5, -0.3, f'RMSE: {rmse:.4f}\nR²: {r2:.4f}\nPSNR: {psnr_value:.4f}\nSSIM: {ssim_value:.4f}', 
#         #     #         transform=ax.transAxes, ha='center', va='top', fontsize=12)

#         #     # Append to lists
#         #     rmse_list.append(rmse)
#         #     r2_list.append(r2)
#         #     psnr_list.append(psnr_value)
#         #     ssim_list.append(ssim_value)
#         #     print(f"For file {filelist[i]}, RMSE: {rmse:.4f}, R²: {r2:.4f}, PSNR: {psnr_value:.4f}, SSIM: {ssim_value:.4f}")
#         # Show metrics on plot
#         # ax.text(0.5, -0.3, f'RMSE: {rmse:.4f}\nR²: {r2:.4f}\nPSNR: {psnr_value:.4f}\nSSIM: {ssim_value:.4f}', 
#         #         transform=ax.transAxes, ha='center', va='top', fontsize=12)  

         
#         # plt.subplots_adjust(
#         #     wspace=0.38, 
#         #     hspace=-0.4 
#         #     # bottom=0.2, 
#         #     # left=0.2     
#         # )
#         # print(f"Saving {filelist[i][:-3]}.png")
#         # print('-------------------------------------------------------')
#         # plt.savefig(
#         #     workpath + 'Test_Outputs/Plots/' + filelist[i][:-3] + '.png', 
#         #     bbox_inches='tight', 
#         #     facecolor=(1, 1, 1, 0)
#         # )
#         # print(f"Saving {filelist[i][:-3]}.pdf")
#         # plt.savefig(
#         #     workpath + 'Test_Outputs/Plots/' + filelist[i][:-3] + '.pdf', 
#         #     bbox_inches='tight', 
#         #     facecolor=(1, 1, 1, 0)
#         # )
#         plt.close()



# average_rmse = np.mean(rmse_list)
# average_r2 = np.mean(r2_list)
# average_psnr = np.mean(psnr_list)
# average_ssim = np.mean(ssim_list)

# Print or log the results
# print(f'Average RMSE: {average_rmse:.4f}')
# print(f'Average R²: {average_r2:.4f}')
# print(f'Average PSNR: {average_psnr:.4f}')
# print(f'Average SSIM: {average_ssim:.4f}')


# Standard deviations
# std_rmse = np.std(rmse_list)
# std_r2 = np.std(r2_list)
# std_psnr = np.std(psnr_list)
# std_ssim = np.std(ssim_list)

# # Print with std
# print(f'Average RMSE: {average_rmse:.4f} ± {std_rmse:.4f}')
# print(f'Average R²: {average_r2:.4f} ± {std_r2:.4f}')
# print(f'Average PSNR: {average_psnr:.4f} ± {std_psnr:.4f}')
# print(f'Average SSIM: {average_ssim:.4f} ± {std_ssim:.4f}')




# ── Metric helpers ────────────────────────────────────────────────────────────
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
        # some metrics may be strings (e.g. filename); format accordingly
        if isinstance(v, str):
            print(f"{pad}  {k:<20s}: {v}")
        else:
            try:
                print(f"{pad}  {k:<20s}: {v:.4f}")
            except Exception:
                # fallback to generic str conversion if formatting fails
                print(f"{pad}  {k:<20s}: {v}")

# ── Per-image inference + metric collection ───────────────────────────────────
all_metrics = []          # list of dicts, one per image

for i in range(len_file):
    # ── Load & preprocess ────────────────────────────────────────────────────
    filepath = os.path.join(test_dir, filelist[i])
    file = h5py.File(filepath, 'r')
    tmp_lr = norm_goes(np.array(file['goes'][:]))
    tmp_hr = norm_viirs(np.array(file['viirs'][:]))

    # create tensors directly with appropriate dtype/device to avoid deprecated constructors
    tmp_lr = torch.tensor(tmp_lr, dtype=torch.float32, device=device).reshape((1, 3, 27, 35))
    tmp_hr = torch.tensor(tmp_hr, dtype=torch.float32, device=device).reshape((1, 3, 162, 210))

    imgs_lr = Variable(tmp_lr.type(Tensor)).to(device)
    imgs_hr = Variable(tmp_hr.type(Tensor)).to(device)

    # ── Inference ────────────────────────────────────────────────────────────
    gen_hr  = generator(imgs_lr)
    intero  = nn.functional.interpolate(imgs_lr, scale_factor=4)

    # ── Denormalise ──────────────────────────────────────────────────────────
    gen_hr = denorm_viirs(gen_hr)
    imgs_lr = denorm_goes(imgs_lr)
    imgs_hr = denorm_viirs(imgs_hr)
    intero  = denorm_goes(intero)

    # ── Mask tensors ─────────────────────────────────────────────────────────
    # avoid deprecated constructors by creating from existing tensors
    x_mask_tensor = x_mask.clone()
    y_mask_tensor = y_mask.clone()
    x_mask_tensor[x_mask == 0] = float('nan')
    y_mask_tensor[y_mask == 0] = float('nan')

    # ── Spatial crop ─────────────────────────────────────────────────────────
    imgs_lr = imgs_lr[:, :, :27,  :]
    intero  = intero [:, :, :162, :]
    imgs_hr = imgs_hr[:, :, :162, :]
    gen_hr  = gen_hr [:, :, :162, :]

    # ── Convert to numpy for metric functions ────────────────────────────────
    pred_np = gen_hr[0].cpu().detach().numpy()   # (C, H, W)
    gt_np   = imgs_hr[0].cpu().detach().numpy()

    # ── Per-image metrics ────────────────────────────────────────────────────
    m = compute_metrics(pred_np, gt_np)
    m['filename'] = filelist[i]          # keep provenance
    all_metrics.append(m)

    print_metrics(f"[{i+1:03d}/{len_file}] {filelist[i]}", m, indent=2)
    plt.close()

# ── Global (aggregate) metrics ────────────────────────────────────────────────
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