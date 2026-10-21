import torch
import torch.nn as nn
import os
import numpy as np
import h5py

try:
    import cv2
except ImportError:
    cv2 = None
    from scipy.signal import convolve2d
    print("[warning] cv2 not found; using scipy convolution fallback.")


workpath = r''

import sys
sys.path.append(workpath)

from models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HyperParameters:
    def __init__(self):
        self.checkpoint_model = os.path.join(
            workpath, 'Models', 'gen_1e3lr_newMAXmin_24000.pth'
        )
        self.channels = 3
        self.residual_blocks = 32

opt = HyperParameters()

generator = GeneratorRRDB(
    opt.channels,
    filters=64,
    num_res_blocks=opt.residual_blocks
).to(device)

generator.load_state_dict(
    torch.load(opt.checkpoint_model, map_location=device, weights_only=True)
)

generator.eval()

def _validate_inputs(pred, gt):
    if pred.shape != gt.shape:
        raise ValueError("pred and gt must have same shape")
    if pred.ndim != 3:
        raise ValueError("Expected shape (C,H,W)")
    return pred.astype(np.float32), gt.astype(np.float32)


#  Gradient RMSE (vector form) 
def GradientRMSE(pred, gt):
    pred, gt = _validate_inputs(pred, gt)
    diff = pred - gt

    grad_x = np.gradient(diff, axis=2)
    grad_y = np.gradient(diff, axis=1)

    return np.sqrt(np.mean(grad_x**2 + grad_y**2))


#  Laplace RMSE 
def LaplaceRMSE(pred, gt):
    pred, gt = _validate_inputs(pred, gt)

    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]], dtype=np.float32)

    pred_lap = np.zeros_like(pred)
    gt_lap = np.zeros_like(gt)

    if cv2 is not None:
        for c in range(pred.shape[0]):
            pred_lap[c] = cv2.filter2D(
                pred[c], -1, kernel,
                borderType=cv2.BORDER_REFLECT
            )
            gt_lap[c] = cv2.filter2D(
                gt[c], -1, kernel,
                borderType=cv2.BORDER_REFLECT
            )
    else:
        for c in range(pred.shape[0]):
            pred_lap[c] = convolve2d(
                pred[c], kernel, mode='same', boundary='symm'
            )
            gt_lap[c] = convolve2d(
                gt[c], kernel, mode='same', boundary='symm'
            )

    return np.sqrt(np.mean((pred_lap - gt_lap) ** 2))


#  Fourier RMSE 
def FourierRMSE(pred, gt, normalize=True):
    pred, gt = _validate_inputs(pred, gt)

    pred_fft = np.fft.fft2(pred, axes=(-2, -1))
    gt_fft = np.fft.fft2(gt, axes=(-2, -1))

    if normalize:
        scale = pred.shape[-2] * pred.shape[-1]
        pred_fft /= scale
        gt_fft /= scale

    pred_mag = np.abs(pred_fft)
    gt_mag = np.abs(gt_fft)

    return np.sqrt(np.mean((pred_mag - gt_mag) ** 2))



def compute_metrics(pred_img, gt_img):

    metrics = {}

    for c in range(pred_img.shape[0]):
        pred_c = pred_img[c:c+1]  # keep (1,H,W)
        gt_c   = gt_img[c:c+1]

        metrics[f'grad_rmse_c{c}'] = GradientRMSE(pred_c, gt_c)
        metrics[f'laplace_rmse_c{c}'] = LaplaceRMSE(pred_c, gt_c)
        metrics[f'fourier_rmse_c{c}'] = FourierRMSE(pred_c, gt_c)

    return metrics



test_dir = os.path.join(workpath, 'Data', 'Test1')
filelist = sorted(os.listdir(test_dir))
len_file = len(filelist)

all_metrics = []

for i in range(len_file):

    filepath = os.path.join(test_dir, filelist[i])
    file = h5py.File(filepath, 'r')

    tmp_lr = norm_goes(np.array(file['goes'][:]))
    tmp_hr = norm_viirs(np.array(file['viirs'][:]))

    tmp_lr = torch.tensor(tmp_lr, dtype=torch.float32, device=device).reshape((1,3,27,35))
    tmp_hr = torch.tensor(tmp_hr, dtype=torch.float32, device=device).reshape((1,3,162,210))

    with torch.no_grad():
        gen_hr = generator(tmp_lr)

    gen_hr = denorm_viirs(gen_hr)
    tmp_hr = denorm_viirs(tmp_hr)

    gen_hr = gen_hr[:, :, :162, :]
    tmp_hr = tmp_hr[:, :, :162, :]

    pred_np = gen_hr[0].cpu().numpy()
    gt_np   = tmp_hr[0].cpu().numpy()

    m = compute_metrics(pred_np, gt_np)
    m["filename"] = filelist[i]

    all_metrics.append(m)

    print(f"[{i+1:03d}/{len_file}] {filelist[i]}")
    for k, v in m.items():
        if k != "filename":
            print(f"  {k:<20s}: {v:.6f}")


metric_keys = [k for k in all_metrics[0].keys() if k != "filename"]

global_metrics = {
    k: {
        "mean": np.mean([m[k] for m in all_metrics]),
        "std":  np.std([m[k] for m in all_metrics]),
        "min":  np.min([m[k] for m in all_metrics]),
        "max":  np.max([m[k] for m in all_metrics]),
    }
    for k in metric_keys
}

print("\n" + "="*65)
print("GLOBAL METRICS (PER CHANNEL)")
print("="*65)
print(f"{'Metric':<25s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}")
print("-"*65)

for k, stats in global_metrics.items():
    print(f"{k:<25s} "
          f"{stats['mean']:10.6f} "
          f"{stats['std']:10.6f} "
          f"{stats['min']:10.6f} "
          f"{stats['max']:10.6f}")

print("="*65)
