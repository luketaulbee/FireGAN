# # Full patched training + testing pipeline for SR 6x on GOES -> VIIRS-style data.
# # Fixes in v2:
# # - Corrected interpolation baseline: bicubic performed in GOES physical units, then compared to VIIRS GT.
# # - Kept prior fixes (deterministic seeds, safer DataLoader, PSNR/SSIM fixes, Kaiming init, best checkpoints).
# # - Small diagnostic prints added.
# #
# # Save this file and run as before.

# import os
# import sys
# import math
# import random
# import datetime as dt
# from glob import glob
# from typing import Tuple, List, Dict

# import h5py
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader, random_split

# from skimage.metrics import structural_similarity as ssim

# mode = 'linear'  # interpolation mode for baseline

# # ----------------------------
# # Configuration / Hyperparams
# # ----------------------------
# class Config:
#     workpath = r'C:\Users\lhtau\OneDrive\Desktop\FireGAN_LukeTaulbee\\'
#     train_dir = os.path.join(workpath, 'Data/Train1/')
#     val_split = 0.1
#     test_dir = os.path.join(workpath, 'Data/Test1/')
#     savedir = os.path.join(workpath, 'other_models', 'Saved_Models6')
#     image_out_dir = os.path.join(workpath, 'other_models', 'Images')
#     os.makedirs(savedir, exist_ok=True)
#     os.makedirs(image_out_dir, exist_ok=True)

#     seed = 42
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     channels = 3
#     batch_size = 4            # change to 1 only if necessary
#     n_epochs = 10000
#     lr = 1e-4
#     betas = (0.9, 0.999)
#     checkpoint_interval = 500
#     sample_interval = 200
#     num_workers = 0           # safer default when using h5py in __getitem__

# cfg = Config()

# # reproducibility
# def set_reproducible(seed: int):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# set_reproducible(cfg.seed)

# # ----------------------------
# # Utilities: compute ranges
# # ----------------------------
# def compute_channel_ranges(filelist: List[str]) -> Tuple[np.ndarray, np.ndarray]:
#     mins = []
#     maxs = []
#     for f in filelist:
#         try:
#             with h5py.File(f, 'r') as fh:
#                 goes = np.array(fh['goes'][:], dtype=np.float32)
#                 viirs = np.array(fh['viirs'][:], dtype=np.float32)
#         except Exception:
#             continue
#         mins.append((goes.min(axis=(1,2)), viirs.min(axis=(1,2))))
#         maxs.append((goes.max(axis=(1,2)), viirs.max(axis=(1,2))))
#     if len(mins) == 0:
#         raise RuntimeError("No readable files found to compute ranges.")
#     goes_mins = np.min(np.stack([m[0] for m in mins], axis=0), axis=0)
#     viirs_mins = np.min(np.stack([m[1] for m in mins], axis=0), axis=0)
#     goes_maxs = np.max(np.stack([M[0] for M in maxs], axis=0), axis=0)
#     viirs_maxs = np.max(np.stack([M[1] for M in maxs], axis=0), axis=0)
#     return (goes_mins, goes_maxs), (viirs_mins, viirs_maxs)

# # ----------------------------
# # Dataset
# # ----------------------------
# class GOESVIIRSDataset(Dataset):
#     def __init__(self, filelist: List[str],
#                  goes_range: Tuple[np.ndarray, np.ndarray],
#                  viirs_range: Tuple[np.ndarray, np.ndarray],
#                  lr_shape=(3,27,35), hr_shape=(3,162,210)):
#         self.filelist = filelist
#         self.go_min, self.go_max = goes_range
#         self.vi_min, self.vi_max = viirs_range
#         self.lr_shape = lr_shape
#         self.hr_shape = hr_shape

#     def __len__(self):
#         return len(self.filelist)

#     def norm(self, arr: np.ndarray, ch_min: np.ndarray, ch_max: np.ndarray) -> np.ndarray:
#         out = np.empty_like(arr, dtype=np.float32)
#         for c in range(arr.shape[0]):
#             denom = float(ch_max[c] - ch_min[c])
#             if denom == 0:
#                 out[c] = 0.0
#             else:
#                 out[c] = (arr[c] - ch_min[c]) / denom
#         return out.clip(0., 1.)

#     def __getitem__(self, idx):
#         fpath = self.filelist[idx]
#         with h5py.File(fpath, 'r') as fh:
#             goes = np.array(fh['goes'][:], dtype=np.float32)   # (C,H,W)
#             viirs = np.array(fh['viirs'][:], dtype=np.float32) # (C,H,W)

#         goes_n = self.norm(goes, self.go_min, self.go_max)
#         viirs_n = self.norm(viirs, self.vi_min, self.vi_max)

#         lr = torch.from_numpy(goes_n).float()
#         hr = torch.from_numpy(viirs_n).float()

#         assert lr.shape == torch.Size(self.lr_shape), f"LR shape mismatch {lr.shape} expected {self.lr_shape} for {fpath}"
#         assert hr.shape == torch.Size(self.hr_shape), f"HR shape mismatch {hr.shape} expected {self.hr_shape} for {fpath}"

#         return lr, hr, os.path.basename(fpath)

# # ----------------------------
# # Models
# # ----------------------------
# class ResidualBlockIN(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(channels, channels, 3, 1, 1),
#             nn.InstanceNorm2d(channels, affine=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels, channels, 3, 1, 1),
#             nn.InstanceNorm2d(channels, affine=True),
#         )
#     def forward(self, x):
#         return x + self.block(x)

# class ResNetSR6x(nn.Module):
#     def __init__(self, channels=3, num_res_blocks=16, base_filters=64):
#         super().__init__()
#         self.conv1 = nn.Conv2d(channels, base_filters, kernel_size=3, stride=1, padding=1)
#         self.relu = nn.ReLU(inplace=True)
#         self.res_blocks = nn.Sequential(*[ResidualBlockIN(base_filters) for _ in range(num_res_blocks)])
#         self.conv_mid = nn.Conv2d(base_filters, base_filters, kernel_size=3, stride=1, padding=1)
#         self.inst_mid = nn.InstanceNorm2d(base_filters, affine=True)
#         self.up2 = nn.Sequential(
#             nn.Conv2d(base_filters, base_filters * 4, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.PixelShuffle(2)
#         )
#         self.up3 = nn.Sequential(
#             nn.Conv2d(base_filters, base_filters * 9, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.PixelShuffle(3)
#         )
#         self.conv_out = nn.Conv2d(base_filters, channels, kernel_size=3, stride=1, padding=1)

#     def forward(self, x):
#         x1 = self.relu(self.conv1(x))
#         r = self.res_blocks(x1)
#         r = self.inst_mid(self.conv_mid(r))
#         r = x1 + r
#         u = self.up2(r)
#         u = self.up3(u)
#         out = self.conv_out(u)
#         return out

# class ConvBlockIN(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
#             nn.InstanceNorm2d(out_channels, affine=True),
#             nn.ReLU(inplace=True),
#         )
#     def forward(self, x):
#         return self.block(x)

# class AutoencoderSR6x(nn.Module):
#     def __init__(self, channels=3, base_filters=64, num_bottleneck_blocks=6):
#         super().__init__()
#         self.enc1 = nn.Sequential(ConvBlockIN(channels, base_filters), ConvBlockIN(base_filters, base_filters))
#         self.enc2 = nn.Sequential(ConvBlockIN(base_filters, base_filters*2), ConvBlockIN(base_filters*2, base_filters*2))
#         self.enc3 = nn.Sequential(ConvBlockIN(base_filters*2, base_filters*4), ConvBlockIN(base_filters*4, base_filters*4))
#         bott = []
#         for _ in range(num_bottleneck_blocks):
#             bott.append(ConvBlockIN(base_filters*4, base_filters*4))
#         self.bottleneck = nn.Sequential(*bott)
#         self.dec3 = nn.Sequential(ConvBlockIN(base_filters*4, base_filters*2), ConvBlockIN(base_filters*2, base_filters*2))
#         self.dec2 = nn.Sequential(ConvBlockIN(base_filters*2, base_filters), ConvBlockIN(base_filters, base_filters))
#         self.up2 = nn.Sequential(
#             nn.Conv2d(base_filters, base_filters * 4, kernel_size=3, padding=1),
#             nn.LeakyReLU(inplace=True),
#             nn.PixelShuffle(2)
#         )
#         self.up3 = nn.Sequential(
#             nn.Conv2d(base_filters, base_filters * 9, kernel_size=3, padding=1),
#             nn.LeakyReLU(inplace=True),
#             nn.PixelShuffle(3)
#         )
#         self.out_conv = nn.Conv2d(base_filters, channels, kernel_size=3, padding=1)

#     def forward(self, x):
#         e1 = self.enc1(x)
#         e2 = self.enc2(e1)
#         e3 = self.enc3(e2)
#         b = self.bottleneck(e3)
#         d3 = self.dec3(b) + e2
#         d2 = self.dec2(d3) + e1
#         u = self.up2(d2)
#         u = self.up3(u)
#         out = self.out_conv(u)
#         return out

# class InterpolationSR6x(nn.Module):
#     def __init__(self, mode=mode, align_corners=False):
#         super().__init__()
#         self.mode = mode
#         self.align_corners = align_corners
#         self.scale_factor = 6
#     def forward(self, x):
#         if self.mode in ('linear', 'bilinear', 'bicubic', 'trilinear'):
#             return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
#         else:
#             return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)

# # ----------------------------
# # Losses & Metrics
# # ----------------------------
# l1_loss = nn.L1Loss()
# mse_loss = nn.MSELoss()

# def hybrid_loss(pred, gt, use_perceptual=False, perceptual_fn=None,
#                 weight_l1=1.0, weight_content=0.0):
#     loss_pixel = weight_l1 * l1_loss(pred, gt)
#     loss_content = 0.0
#     if use_perceptual and (perceptual_fn is not None):
#         with torch.no_grad():
#             feat_pred = perceptual_fn(pred)
#             feat_gt = perceptual_fn(gt)
#         loss_content = weight_content * mse_loss(feat_pred, feat_gt)
#     return loss_pixel + loss_content

# def calculate_rmse(pred: np.ndarray, gt: np.ndarray) -> float:
#     return float(np.sqrt(np.mean((pred - gt) ** 2)))

# def calculate_r2(pred: np.ndarray, gt: np.ndarray) -> float:
#     pred = np.ravel(pred)
#     gt = np.ravel(gt)
#     ss_res = np.sum((gt - pred) ** 2)
#     ss_tot = np.sum((gt - np.mean(gt)) ** 2)
#     return float(1 - ss_res/ss_tot) if ss_tot != 0 else 0.0

# def calculate_psnr(pred: np.ndarray, gt: np.ndarray) -> float:
#     mse = np.mean((pred - gt) ** 2)
#     if mse == 0:
#         return float('inf')
#     data_range = float(np.max(gt) - np.min(gt))
#     if data_range == 0:
#         data_range = 1.0
#     return 10.0 * math.log10((data_range ** 2) / mse)

# def calculate_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
#     if gt.max() == gt.min():
#         return 1.0 if np.allclose(pred, gt, atol=1e-6) else 0.0
#     h, w = pred.shape
#     win = min(7, h, w)
#     if win % 2 == 0:
#         win -= 1
#     win = max(3, win)
#     try:
#         return float(ssim(pred, gt, data_range=float(gt.max()-gt.min()), win_size=win))
#     except Exception:
#         return 0.0

# # ----------------------------
# # Utilities: initialization & checkpoint helpers
# # ----------------------------
# def init_weights(m):
#     if isinstance(m, nn.Conv2d):
#         nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
#         if m.bias is not None:
#             nn.init.zeros_(m.bias)
#     elif isinstance(m, nn.Linear):
#         nn.init.xavier_normal_(m.weight)
#         if m.bias is not None:
#             nn.init.zeros_(m.bias)

# def save_checkpoint_if_best(model, optimizer, epoch, val_loss, best_loss, savedir, name_prefix):
#     if val_loss < best_loss:
#         os.makedirs(savedir, exist_ok=True)
#         ckpt_path = os.path.join(savedir, f"{name_prefix}_best.pth")
#         torch.save({
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'val_loss': val_loss,
#         }, ckpt_path)
#         print(f"Saved new best {name_prefix} checkpoint with val_loss={val_loss:.6f} -> {ckpt_path}")
#         return val_loss
#     return best_loss

# # ----------------------------
# # Training / Validation / Testing
# # ----------------------------
# def train_and_evaluate():
#     # gather file lists
#     train_files = sorted([os.path.join(cfg.train_dir, f) for f in os.listdir(cfg.train_dir) if f.endswith('.h5')])
#     if len(train_files) == 0:
#         raise RuntimeError("No training files found in: " + cfg.train_dir)

#     # compute ranges from training files (goes and viirs)
#     (go_min, go_max), (vi_min, vi_max) = compute_channel_ranges(train_files)
#     print("Computed ranges:")
#     print("GOES min:", go_min, "GOES max:", go_max)
#     print("VIIRS min:", vi_min, "VIIRS max:", vi_max)

#     # split train / val
#     n_val = max(1, int(len(train_files) * cfg.val_split))
#     n_train = len(train_files) - n_val
#     random.shuffle(train_files)
#     train_list = train_files[:n_train]
#     val_list = train_files[n_train:]

#     # create datasets
#     ds_train = GOESVIIRSDataset(train_list, (go_min, go_max), (vi_min, vi_max))
#     ds_val = GOESVIIRSDataset(val_list, (go_min, go_max), (vi_min, vi_max))

#     loader_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True,
#                               num_workers=cfg.num_workers, pin_memory=True)
#     loader_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False,
#                             num_workers=cfg.num_workers, pin_memory=True)

#     # instantiate models
#     autoencoder = AutoencoderSR6x(channels=cfg.channels).to(cfg.device)
#     resnet = ResNetSR6x(channels=cfg.channels).to(cfg.device)
#     interp = InterpolationSR6x(mode=mode).to(cfg.device)

#     # initialize weights
#     autoencoder.apply(init_weights)
#     resnet.apply(init_weights)

#     opt_ae = torch.optim.Adam(autoencoder.parameters(), lr=cfg.lr, betas=cfg.betas)
#     opt_res = torch.optim.Adam(resnet.parameters(), lr=cfg.lr, betas=cfg.betas)

#     scheduler_ae = torch.optim.lr_scheduler.StepLR(opt_ae, step_size=500, gamma=0.5)
#     scheduler_res = torch.optim.lr_scheduler.StepLR(opt_res, step_size=500, gamma=0.5)

#     history = {'ae_train':[], 'res_train':[], 'ae_val':[], 'res_val':[]}

#     start = dt.datetime.now()
#     print("Start training:", start)

#     best_val_ae = float('inf')
#     best_val_res = float('inf')

#     # prepare GOES range tensors for on-device conversion (used for interpolation baseline)
#     go_min_t = torch.tensor(go_min, dtype=torch.float32).view(1, cfg.channels, 1, 1).to(cfg.device)
#     go_max_t = torch.tensor(go_max, dtype=torch.float32).view(1, cfg.channels, 1, 1).to(cfg.device)

#     for epoch in range(1, cfg.n_epochs + 1):
#         autoencoder.train()
#         resnet.train()
#         train_losses_ae = []
#         train_losses_res = []

#         for lr_batch, hr_batch, _ in loader_train:
#             lr_batch = lr_batch.to(cfg.device)      # normalized using GOES ranges
#             hr_batch = hr_batch.to(cfg.device)      # normalized using VIIRS ranges

#             # Autoencoder step
#             opt_ae.zero_grad()
#             pred_ae = autoencoder(lr_batch)
#             loss_ae = hybrid_loss(pred_ae, hr_batch, use_perceptual=False, weight_l1=1.0, weight_content=0.0)
#             loss_ae.backward()
#             opt_ae.step()
#             train_losses_ae.append(loss_ae.item())

#             # ResNet step
#             opt_res.zero_grad()
#             pred_res = resnet(lr_batch)
#             loss_res = hybrid_loss(pred_res, hr_batch, use_perceptual=False, weight_l1=1.0, weight_content=0.0)
#             loss_res.backward()
#             opt_res.step()
#             train_losses_res.append(loss_res.item())

#         scheduler_ae.step()
#         scheduler_res.step()

#         mean_ae = float(np.mean(train_losses_ae))
#         mean_res = float(np.mean(train_losses_res))
#         history['ae_train'].append(mean_ae)
#         history['res_train'].append(mean_res)

#         # Validation
#         autoencoder.eval()
#         resnet.eval()
#         val_losses_ae = []
#         val_losses_res = []
#         metrics_acc = []

#         with torch.no_grad():
#             for lr_batch, hr_batch, fnames in loader_val:
#                 lr_batch = lr_batch.to(cfg.device)      # GOES-normalized
#                 hr_batch = hr_batch.to(cfg.device)      # VIIRS-normalized

#                 pred_ae = autoencoder(lr_batch)
#                 pred_res = resnet(lr_batch)

#                 # INTERPOLATION BASELINE (CORRECTED):
#                 # 1) convert lr_batch (GOES-normalized) -> GOES physical units
#                 goes_phys = lr_batch * (go_max_t - go_min_t) + go_min_t
#                 # 2) upsample in physical units (bicubic)
#                 pred_interp_phys = F.interpolate(goes_phys, scale_factor=6, mode=mode, align_corners=False)

#                 # losses for models (they output VIIRS-normalized)
#                 loss_a = hybrid_loss(pred_ae, hr_batch, use_perceptual=False, weight_l1=1.0, weight_content=0.0)
#                 loss_r = hybrid_loss(pred_res, hr_batch, use_perceptual=False, weight_l1=1.0, weight_content=0.0)
#                 val_losses_ae.append(loss_a.item())
#                 val_losses_res.append(loss_r.item())

#                 # numpy arrays for denorm and metrics
#                 pred_ae_np = pred_ae.cpu().numpy()
#                 pred_res_np = pred_res.cpu().numpy()
#                 # pred_interp_phys is in physical units already
#                 pred_interp_phys_np = pred_interp_phys.cpu().numpy()
#                 hr_np = hr_batch.cpu().numpy()

#                 # denormalize AE/Res (they are VIIRS-normalized)
#                 for b in range(hr_np.shape[0]):
#                     den_ae = np.empty_like(pred_ae_np[b])
#                     den_res = np.empty_like(pred_res_np[b])
#                     den_interp = np.empty_like(pred_interp_phys_np[b])
#                     den_gt = np.empty_like(hr_np[b])
#                     for ch in range(cfg.channels):
#                         den_ae[ch] = pred_ae_np[b,ch] * (vi_max[ch]-vi_min[ch]) + vi_min[ch]
#                         den_res[ch] = pred_res_np[b,ch] * (vi_max[ch]-vi_min[ch]) + vi_min[ch]
#                         # pred_interp_phys is already in GOES physical units; convert to numpy and compare directly
#                         den_interp[ch] = pred_interp_phys_np[b,ch]
#                         den_gt[ch] = hr_np[b,ch] * (vi_max[ch]-vi_min[ch]) + vi_min[ch]

#                     sample_metrics = {}
#                     for ch in range(cfg.channels):
#                         p_ae = den_ae[ch]
#                         p_res = den_res[ch]
#                         p_int = den_interp[ch]   # physical units
#                         g = den_gt[ch]

#                         sample_metrics.update({
#                             f"ch{ch+1}_ae_rmse": calculate_rmse(p_ae, g),
#                             f"ch{ch+1}_res_rmse": calculate_rmse(p_res, g),
#                             f"ch{ch+1}_int_rmse": calculate_rmse(p_int, g),
#                         })
#                     metrics_acc.append(sample_metrics)

#         mean_val_ae = float(np.mean(val_losses_ae)) if len(val_losses_ae)>0 else float('inf')
#         mean_val_res = float(np.mean(val_losses_res)) if len(val_losses_res)>0 else float('inf')
#         history['ae_val'].append(mean_val_ae if mean_val_ae != float('inf') else 0.0)
#         history['res_val'].append(mean_val_res if mean_val_res != float('inf') else 0.0)

#         # save best checkpoints
#         best_val_ae = save_checkpoint_if_best(autoencoder, opt_ae, epoch, mean_val_ae, best_val_ae, cfg.savedir, 'autoencoder')
#         best_val_res = save_checkpoint_if_best(resnet, opt_res, epoch, mean_val_res, best_val_res, cfg.savedir, 'resnet')

#         if epoch % 1 == 0:
#             print(f"[Epoch {epoch}/{cfg.n_epochs}] Train AE {mean_ae:.6f} Res {mean_res:.6f} | Val AE {mean_val_ae:.6f} Res {mean_val_res:.6f}")

#         if epoch % cfg.checkpoint_interval == 0 or epoch == cfg.n_epochs:
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': autoencoder.state_dict(),
#                 'optimizer_state_dict': opt_ae.state_dict(),
#             }, os.path.join(cfg.savedir, f'autoencoder_epoch{epoch}.pth'))
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': resnet.state_dict(),
#                 'optimizer_state_dict': opt_res.state_dict(),
#             }, os.path.join(cfg.savedir, f'resnet_epoch{epoch}.pth'))
#             print(f"Saved periodic checkpoints at epoch {epoch}")

#     torch.save(autoencoder.state_dict(), os.path.join(cfg.savedir, 'autoencoder_final.pth'))
#     torch.save(resnet.state_dict(), os.path.join(cfg.savedir, 'resnet_final.pth'))
#     print("Training complete. Models saved.")

#     # plot training curves
#     plt.figure(figsize=(8,5))
#     plt.plot(history['ae_train'], label='AE train')
#     plt.plot(history['res_train'], label='ResNet train')
#     plt.plot(history['ae_val'], label='AE val')
#     plt.plot(history['res_val'], label='ResNet val')
#     plt.xlabel('epoch')
#     plt.ylabel('loss')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(os.path.join(cfg.image_out_dir, 'training_loss_curve.png'), dpi=200)
#     plt.show()

# # ----------------------------
# # Testing / Evaluation (separate)
# # ----------------------------
# def evaluate_test_set(model_paths: Dict[str,str]=None):
#     test_files = sorted([os.path.join(cfg.test_dir, f) for f in os.listdir(cfg.test_dir) if f.endswith('.h5')])
#     if len(test_files) == 0:
#         raise RuntimeError("No test files found in: " + cfg.test_dir)

#     train_files = sorted([os.path.join(cfg.train_dir,f) for f in os.listdir(cfg.train_dir) if f.endswith('.h5')])
#     (go_min, go_max), (vi_min, vi_max) = compute_channel_ranges(train_files)

#     ds_test = GOESVIIRSDataset(test_files, (go_min, go_max), (vi_min, vi_max))
#     loader_test = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=0)

#     ae = AutoencoderSR6x(channels=cfg.channels).to(cfg.device)
#     rn = ResNetSR6x(channels=cfg.channels).to(cfg.device)
#     interp = InterpolationSR6x(mode=mode).to(cfg.device)

#     if model_paths is None:
#         model_paths = {
#             'AE': os.path.join(cfg.savedir, 'autoencoder_final.pth'),
#             'ResNet': os.path.join(cfg.savedir, 'resnet_final.pth')
#         }

#     if os.path.exists(model_paths['AE']):
#         ae.load_state_dict(torch.load(model_paths['AE'], map_location=cfg.device))
#     else:
#         print("AE weights not found:", model_paths['AE'])
#     if os.path.exists(model_paths['ResNet']):
#         rn.load_state_dict(torch.load(model_paths['ResNet'], map_location=cfg.device))
#     else:
#         print("ResNet weights not found:", model_paths['ResNet'])

#     ae.eval(); rn.eval(); interp.eval()

#     # create GOES min/max tensors for denorm to physical for interpolation baseline
#     go_min_t = torch.tensor(go_min, dtype=torch.float32).view(1, cfg.channels, 1, 1).to(cfg.device)
#     go_max_t = torch.tensor(go_max, dtype=torch.float32).view(1, cfg.channels, 1, 1).to(cfg.device)

#     metrics_rows = []
#     with torch.no_grad():
#         for lr, hr, fname in loader_test:
#             lr = lr.to(cfg.device)   # GOES-normalized
#             hr = hr.to(cfg.device)   # VIIRS-normalized

#             ae_out = ae(lr)          # VIIRS-normalized
#             res_out = rn(lr)         # VIIRS-normalized

#             # Interp baseline: convert LR -> GOES physical, then upsample in physical units
#             goes_phys = lr * (go_max_t - go_min_t) + go_min_t
#             int_phys = F.interpolate(goes_phys, scale_factor=6, mode=mode, align_corners=False)

#             ae_np = ae_out.squeeze(0).cpu().numpy()
#             res_np = res_out.squeeze(0).cpu().numpy()
#             int_np = int_phys.squeeze(0).cpu().numpy()
#             gt_np = hr.squeeze(0).cpu().numpy()

#             den_ae = np.empty_like(ae_np)
#             den_res = np.empty_like(res_np)
#             den_int = np.empty_like(int_np)
#             den_gt = np.empty_like(gt_np)
#             for ch in range(cfg.channels):
#                 den_ae[ch] = ae_np[ch] * (vi_max[ch]-vi_min[ch]) + vi_min[ch]
#                 den_res[ch] = res_np[ch] * (vi_max[ch]-vi_min[ch]) + vi_min[ch]
#                 den_int[ch] = int_np[ch]   # already physical (GOES units)
#                 den_gt[ch] = gt_np[ch] * (vi_max[ch]-vi_min[ch]) + vi_min[ch]

#             row = {'filename': fname[0]}
#             for ch in range(cfg.channels):
#                 p_ae = den_ae[ch]; p_res = den_res[ch]; p_int = den_int[ch]; g = den_gt[ch]
#                 row.update({
#                     f"AE_CH{ch+1}_RMSE": calculate_rmse(p_ae, g),
#                     f"AE_CH{ch+1}_R2":   calculate_r2(p_ae, g),
#                     f"AE_CH{ch+1}_PSNR": calculate_psnr(p_ae, g),
#                     f"AE_CH{ch+1}_SSIM": calculate_ssim(p_ae, g),

#                     f"Res_CH{ch+1}_RMSE": calculate_rmse(p_res, g),
#                     f"Res_CH{ch+1}_R2":   calculate_r2(p_res, g),
#                     f"Res_CH{ch+1}_PSNR": calculate_psnr(p_res, g),
#                     f"Res_CH{ch+1}_SSIM": calculate_ssim(p_res, g),

#                     f"Int_CH{ch+1}_RMSE": calculate_rmse(p_int, g),
#                     f"Int_CH{ch+1}_R2":   calculate_r2(p_int, g),
#                     f"Int_CH{ch+1}_PSNR": calculate_psnr(p_int, g),
#                     f"Int_CH{ch+1}_SSIM": calculate_ssim(p_int, g),
#                 })
#             metrics_rows.append(row)

#     df = pd.DataFrame(metrics_rows)
#     out_csv = os.path.join(cfg.savedir, 'test_metrics_fixed_v2.csv')
#     df.to_csv(out_csv, index=False)
#     print("Saved test metrics to:", out_csv)

#     summary = {}
#     for model_tag in ['AE','Res','Int']:
#         for metric in ['RMSE','R2','PSNR','SSIM']:
#             cols = [f"{model_tag}_CH{i}_{metric}" for i in range(1, cfg.channels+1)]
#             if all(c in df.columns for c in cols):
#                 summary[f"{model_tag}_{metric}_mean"] = df[cols].mean().mean()
#                 summary[f"{model_tag}_{metric}_std"] = df[cols].mean().std()
#     print("Overall summary (mean across channels):")
#     for k,v in summary.items():
#         print(k, ":", v)

#     return df, summary

# # ----------------------------
# # Main
# # ----------------------------
# if __name__ == '__main__':
#     train_and_evaluate()
#     evaluate_test_set()


# train_resnet6x_upgraded_patched.py
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# ----------------------------
# Configuration
# ----------------------------
class CFG:
    # raw base path (keep single slashes/backslashes consistent)
    raw_workpath = r'C:\Users\lhtau\OneDrive\Desktop\FireGAN_LukeTaulbee'
    workpath = os.path.normpath(raw_workpath)

    train_dir = os.path.normpath(os.path.join(workpath, 'Data', 'Train1'))
    test_dir = os.path.normpath(os.path.join(workpath, 'Data', 'Test1'))
    savedir = os.path.normpath(os.path.join(workpath, 'other_models', 'Saved_Models5'))
    image_out_dir = os.path.normpath(os.path.join(workpath, 'other_models', 'Images'))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    channels = 3  # BT04, BT05, AFI
    n_epochs = 3000
    batch_size = 4
    lr = 2e-4
    min_lr = 1e-5
    checkpoint_interval = 50
    use_perceptual = True  # optional perceptual loss

    # saving robustness
    save_retry_count = 3
    save_retry_delay = 1.0  # seconds

cfg = CFG()

# ensure dirs exist
os.makedirs(cfg.train_dir, exist_ok=True)
os.makedirs(cfg.test_dir, exist_ok=True)
os.makedirs(cfg.savedir, exist_ok=True)
os.makedirs(cfg.image_out_dir, exist_ok=True)

# ----------------------------
# Utilities
# ----------------------------
def calculate_rmse(pred, gt):
    return float(np.sqrt(np.mean((pred - gt)**2)))

def calculate_r2(pred, gt):
    ss_res = np.sum((gt - pred)**2)
    ss_tot = np.sum((gt - np.mean(gt))**2)
    return float(1 - ss_res / (ss_tot + 1e-8))

def calculate_psnr(pred, gt):
    mse = np.mean((pred - gt)**2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = np.max(gt) - np.min(gt)
    if PIXEL_MAX == 0:
        PIXEL_MAX = 1.0
    return float(20 * np.log10(PIXEL_MAX / np.sqrt(mse)))

def calculate_ssim(pred, gt):
    # naive SSIM approximation (keeps previous placeholder behavior)
    if np.max(gt) == np.min(gt):
        return 1.0 if np.allclose(pred, gt, atol=1e-6) else 0.0
    C1 = (0.01*np.max(gt))**2
    C2 = (0.03*np.max(gt))**2
    mu_x = np.mean(pred)
    mu_y = np.mean(gt)
    sigma_x = np.var(pred)
    sigma_y = np.var(gt)
    sigma_xy = np.mean((pred - mu_x)*(gt - mu_y))
    denom = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    if denom == 0:
        return 0.0
    ssim = (2*mu_x*mu_y + C1)*(2*sigma_xy + C2) / denom
    return float(ssim)

def safe_save(obj, path, retry=3, delay=1.0):
    """
    Save atomically by writing to a temporary path then os.replace.
    Retry a few times in case of transient file-locks (e.g., OneDrive).
    """
    tmp = path + '.tmp'
    for attempt in range(1, retry+1):
        try:
            torch.save(obj, tmp)
            os.replace(tmp, path)
            return
        except Exception as e:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass
            if attempt == retry:
                raise
            time.sleep(delay)

# ----------------------------
# Dataset
# ----------------------------
class GOESVIIRSDataset(torch.utils.data.Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        with h5py.File(f, 'r') as h5f:
            lr = h5f['goes'][...]  # low-res: shape (C,h,w)
            hr = h5f['viirs'][...]  # high-res: shape (C,H,W)
        # cast to float32
        lr = lr.astype(np.float32)
        hr = hr.astype(np.float32)

        # normalize 0-1 per-file (keeps same behavior as before)
        lr_min, lr_max = lr.min(), lr.max()
        hr_min, hr_max = hr.min(), hr.max()
        if lr_max - lr_min > 0:
            lr = (lr - lr_min) / (lr_max - lr_min)
        else:
            lr = np.zeros_like(lr, dtype=np.float32)

        if hr_max - hr_min > 0:
            hr = (hr - hr_min) / (hr_max - hr_min)
        else:
            hr = np.zeros_like(hr, dtype=np.float32)

        lr = torch.from_numpy(lr).float()
        hr = torch.from_numpy(hr).float()
        return lr, hr, os.path.basename(f)

# ----------------------------
# Model: RRDB block
# ----------------------------
class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf+gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf+2*gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf+3*gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf+4*gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x,x1],1)))
        x3 = self.lrelu(self.conv3(torch.cat([x,x1,x2],1)))
        x4 = self.lrelu(self.conv4(torch.cat([x,x1,x2,x3],1)))
        x5 = self.conv5(torch.cat([x,x1,x2,x3,x4],1))
        return x + 0.2*x5

class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf,gc)
        self.rdb2 = ResidualDenseBlock(nf,gc)
        self.rdb3 = ResidualDenseBlock(nf,gc)
    def forward(self,x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + 0.2*out

# ----------------------------
# ResNetSR6x
# ----------------------------
class ResNetSR6x(nn.Module):
    def __init__(self, channels=3, nf=128, n_blocks=32):  # increased nf and n_blocks
        super().__init__()
        self.conv_first = nn.Conv2d(channels, nf, 3, 1, 1)
        self.blocks = nn.Sequential(*[RRDB(nf=nf) for _ in range(n_blocks)])
        self.conv_trunk = nn.Conv2d(nf, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.upsample = nn.Sequential(
            nn.Conv2d(nf, nf*4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf*9, 3, 1, 1),
            nn.PixelShuffle(3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, channels, 3, 1, 1)
        )

    def forward(self, x):
        fea = self.lrelu(self.conv_first(x))
        trunk = self.conv_trunk(self.blocks(fea))
        fea = fea + trunk
        out = self.upsample(fea)
        return out

    
# ----------------------------
# Autoencoder SR 6x
# ----------------------------
class AutoencoderSR6x(nn.Module):
    def __init__(self, channels=3, nf=128):
        super().__init__()
        # encoder without downsampling
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # decoder with PixelShuffle 2 + 3 → 6×
        self.decoder = nn.Sequential(
            nn.Conv2d(nf, nf*4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf*9, 3, 1, 1),
            nn.PixelShuffle(3),
            nn.Conv2d(nf, channels, 3, 1, 1)
        )

    def forward(self, x):
        fea = self.encoder(x)
        out = self.decoder(fea)
        return out




# ----------------------------
# Loss functions
# ----------------------------
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            vgg = torch.hub.load('pytorch/vision:v0.14.0','vgg16', pretrained=True)
            self.feature_extractor = nn.Sequential(*list(vgg.features)[:16]).eval()
            for p in self.feature_extractor.parameters(): p.requires_grad=False
        except Exception as e:
            # if hub fails (offline), gracefully disable perceptual
            print("Warning: VGG16 feature extractor failed to load:", e)
            self.feature_extractor = None
        self.l1 = nn.L1Loss()

    def forward(self, pred, gt):
        if self.feature_extractor is None:
            # fallback to L1 if vgg not available
            return self.l1(pred, gt)
        pred_feat = self.feature_extractor(pred)
        gt_feat = self.feature_extractor(gt)
        return self.l1(pred_feat, gt_feat)

# ----------------------------
# Training loop
# ----------------------------
def train():
    train_files = sorted([os.path.join(cfg.train_dir,f) for f in os.listdir(cfg.train_dir) if f.endswith('.h5')])
    if len(train_files) == 0:
        raise RuntimeError(f"No training files found in {cfg.train_dir} -- aborting.")
    ds_train = GOESVIIRSDataset(train_files)
    loader_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=0,
                              pin_memory=True if cfg.device.startswith('cuda') else False)

    # Instantiate both models
    resnet = ResNetSR6x(channels=cfg.channels, nf=128, n_blocks=32).to(cfg.device)
    ae = AutoencoderSR6x(channels=cfg.channels, nf=128).to(cfg.device)

    # Optimizers
    opt_resnet = torch.optim.Adam(resnet.parameters(), lr=cfg.lr)
    opt_ae = torch.optim.Adam(ae.parameters(), lr=cfg.lr)

    # LR schedulers
    sched_resnet = CosineAnnealingLR(opt_resnet, T_max=cfg.n_epochs, eta_min=cfg.min_lr)
    sched_ae = CosineAnnealingLR(opt_ae, T_max=cfg.n_epochs, eta_min=cfg.min_lr)

    # Loss
    criterion_l1 = nn.L1Loss()
    criterion_perc = PerceptualLoss().to(cfg.device) if cfg.use_perceptual else None

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_resnet_loss = float('inf')
    best_ae_loss = float('inf')

    history = {'resnet': [], 'ae': []}

    for epoch in range(1, cfg.n_epochs+1):
        resnet.train()
        ae.train()
        batch_loss_resnet = []
        batch_loss_ae = []

        pbar = tqdm(loader_train, desc=f'Epoch {epoch}')
        for lr_batch, hr_batch, _ in pbar:
            lr_batch = lr_batch.to(cfg.device)
            hr_batch = hr_batch.to(cfg.device)

            # --------------------
            # Train ResNet
            # --------------------
            opt_resnet.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                sr_out = resnet(lr_batch)
                loss_r = criterion_l1(sr_out, hr_batch)
                if cfg.use_perceptual and criterion_perc is not None:
                    loss_r += 0.2 * criterion_perc(sr_out, hr_batch)
            scaler.scale(loss_r).backward()
            scaler.step(opt_resnet)
            batch_loss_resnet.append(loss_r.item())

            # --------------------
            # Train AE
            # --------------------
            opt_ae.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                ae_out = ae(lr_batch)
                loss_ae = criterion_l1(ae_out, hr_batch)
                if cfg.use_perceptual and criterion_perc is not None:
                    loss_ae += 0.2 * criterion_perc(ae_out, hr_batch)
            scaler.scale(loss_ae).backward()
            scaler.step(opt_ae)
            scaler.update()
            batch_loss_ae.append(loss_ae.item())

            pbar.set_postfix({'ResNet_loss': loss_r.item(), 'AE_loss': loss_ae.item()})

        # Epoch metrics
        epoch_loss_r = float(np.mean(batch_loss_resnet))
        epoch_loss_ae = float(np.mean(batch_loss_ae))
        history['resnet'].append(epoch_loss_r)
        history['ae'].append(epoch_loss_ae)

        sched_resnet.step()
        sched_ae.step()

        # Save best ResNet
        if epoch_loss_r < best_resnet_loss:
            best_resnet_loss = epoch_loss_r
            safe_save(resnet.state_dict(), os.path.join(cfg.savedir, 'resnet_best.pth'),
                      retry=cfg.save_retry_count, delay=cfg.save_retry_delay)

        # Save best AE
        if epoch_loss_ae < best_ae_loss:
            best_ae_loss = epoch_loss_ae
            safe_save(ae.state_dict(), os.path.join(cfg.savedir, 'autoencoder_best.pth'),
                      retry=cfg.save_retry_count, delay=cfg.save_retry_delay)

        print(f"Epoch {epoch} complete. ResNet loss: {epoch_loss_r:.6f}, AE loss: {epoch_loss_ae:.6f}")

    # Save final weights
    safe_save(resnet.state_dict(), os.path.join(cfg.savedir,'resnet_final.pth'),
              retry=cfg.save_retry_count, delay=cfg.save_retry_delay)
    safe_save(ae.state_dict(), os.path.join(cfg.savedir,'autoencoder_final.pth'),
              retry=cfg.save_retry_count, delay=cfg.save_retry_delay)

    # Plot training curves
    plt.figure(figsize=(8,5))
    plt.plot(history['resnet'], label='ResNet train')
    plt.plot(history['ae'], label='AE train')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.image_out_dir,'training_loss_curve.png'), dpi=200)
    plt.show()



# ----------------------------
# Evaluation (complete with CSV + summary)
# ----------------------------
def safe_interp_upscale(tensor_lr, scale=6, mode='bicubic'):
    # expects tensor_lr shape (B,C,h,w) on CPU or device
    return F.interpolate(tensor_lr, scale_factor=scale, mode=mode, align_corners=False if mode in ('linear','bilinear','bicubic','trilinear') else None)

def evaluate_test_set(model_paths: dict = None):
    """
    Evaluate AE (optional), ResNet (required), and bicubic interpolation on the test set.
    Produces:
      - metrics_csv (per-file rows)
      - test_metrics_summary.csv (per-channel summary + overall)
    """
    test_files = sorted([os.path.join(cfg.test_dir, f) for f in os.listdir(cfg.test_dir) if f.endswith('.h5')])
    if len(test_files) == 0:
        print("No test files found in:", cfg.test_dir)
        return None

    # model path defaults
    if model_paths is None:
        model_paths = {
            'AE': os.path.join(cfg.savedir, 'autoencoder_final.pth'),
            'ResNet': os.path.join(cfg.savedir, 'resnet_final.pth')
        }

    # instantiate networks (architectures must match saved checkpoints)
    # AE is optional (skip if weights not present)
    ae = None
    if os.path.exists(model_paths.get('AE', '')):
        try:
            # lightweight AE architecture: try to import if exists in your other file
            from __main__ import AutoencoderSR6x  # if present in same module; otherwise user must add AE class
            ae = AutoencoderSR6x(channels=cfg.channels).to(cfg.device)
            ae.load_state_dict(torch.load(model_paths['AE'], map_location=cfg.device))
            ae.eval()
            print("Loaded AE from", model_paths['AE'])
        except Exception:
            print("AE not loaded (class not found or load failed). AE will be skipped.")
            ae = None

    # ResNet (should exist)
    rn = ResNetSR6x(channels=cfg.channels).to(cfg.device)
    if os.path.exists(model_paths['ResNet']):
        rn.load_state_dict(torch.load(model_paths['ResNet'], map_location=cfg.device))
        rn.eval()
    else:
        print("ResNet weights not found:", model_paths['ResNet'])
        return None

    interp_mode = 'bicubic'
    models_to_eval = []
    if ae is not None:
        models_to_eval.append(('AE', ae))
    models_to_eval.append(('ResNet', rn))
    # Interp handled separately

    metrics_list = []
    metrics_csv = os.path.join(cfg.savedir, 'test_metrics_per_file.csv')

    loader_test = DataLoader(GOESVIIRSDataset(test_files), batch_size=1, shuffle=False, num_workers=0)

    with torch.no_grad():
        for lr_batch, hr_batch, fname in loader_test:
            # shapes: lr (1,C,h,w), hr (1,C,H,W)
            lr_batch = lr_batch.to(cfg.device)
            hr_batch = hr_batch.to(cfg.device)

            # forward AE (if available)
            ae_np = None
            if ae is not None:
                try:
                    ae_out = ae(lr_batch)
                    ae_np = ae_out.squeeze(0).cpu().numpy()
                except Exception as e:
                    print("AE forward failed for", fname, ":", e)
                    ae_np = None

            # forward ResNet
            res_out = rn(lr_batch)
            res_np = res_out.squeeze(0).cpu().numpy()

            # interpolation baseline
            int_out = safe_interp_upscale(lr_batch, scale=6, mode=interp_mode)
            int_np = int_out.squeeze(0).cpu().numpy()

            gt_np = hr_batch.squeeze(0).cpu().numpy()

            # helper to compute per-channel metrics and return dict with keys "CH1_RMSE"... etc
            def calculate_metrics_per_channel(pred_np, gt_np):
                # pred_np, gt_np shape (C,H,W)
                d = {}
                for ch in range(cfg.channels):
                    p = pred_np[ch]
                    g = gt_np[ch]
                    d[f"CH{ch+1}_RMSE"] = calculate_rmse(p, g)
                    d[f"CH{ch+1}_R2"] = calculate_r2(p, g)
                    d[f"CH{ch+1}_PSNR"] = calculate_psnr(p, g)
                    d[f"CH{ch+1}_SSIM"] = calculate_ssim(p, g)
                return d

            # build metrics dict per file (use filename only)
            metrics_dict = {"filename": fname[0]}
            # order: AE, ResNet, Interp (AE may be None)
            for model_name, pred_np in zip(["AE", "ResNet", "Interp"], [ae_np, res_np, int_np]):
                if pred_np is None:
                    # skip AE if not available; still put NaNs for keys for compatibility
                    if model_name == "AE":
                        for ch in range(1, cfg.channels+1):
                            for m in ["RMSE","R2","PSNR","SSIM"]:
                                metrics_dict[f"{model_name}_CH{ch}_{m}"] = np.nan
                        continue
                ch_metrics = calculate_metrics_per_channel(pred_np, gt_np)
                # prefix model name: keys become AE_CH1_RMSE etc.
                metrics_dict.update({f"{model_name}_{k}": v for k, v in ch_metrics.items()})
            metrics_list.append(metrics_dict)

    # ==============================
    # Save & Display Metrics
    # ==============================
    df = pd.DataFrame(metrics_list)
    df.to_csv(metrics_csv, index=False)
    print(f"\n Metrics saved to {metrics_csv}\n")

    # ==============================
    # Summary (Per-channel & Overall Mean ± Std)
    # ==============================
    summary_rows = []

    models = ["AE", "ResNet", "Interp"]
    channels = ["CH1", "CH2", "CH3"]
    metrics = ["RMSE", "R2", "PSNR", "SSIM"]
    print("\n================ METRIC SUMMARY ================\n")

    for model in models:
        print(f"===== {model} =====")
        for ch in channels:
            # Build column names for this model/channel
            ch_cols = [f"{model}_{ch}_{m}" for m in metrics]

            # Compute mean & std
            means = df[ch_cols].mean()
            stds = df[ch_cols].std()

            summary_rows.append({
                "Model": model,
                "Channel": ch,
                **{f"{m}_mean": means[f"{model}_{ch}_{m}"] for m in metrics},
                **{f"{m}_std": stds[f"{model}_{ch}_{m}"] for m in metrics},
            })

            print(
                f"{ch}  "
                f"RMSE: {means[f'{model}_{ch}_RMSE']:.4f} ± {stds[f'{model}_{ch}_RMSE']:.4f}, "
                f"R²: {means[f'{model}_{ch}_R2']:.4f} ± {stds[f'{model}_{ch}_R2']:.4f}, "
                f"PSNR: {means[f'{model}_{ch}_PSNR']:.4f} ± {stds[f'{model}_{ch}_PSNR']:.4f}, "
                f"SSIM: {means[f'{model}_{ch}_SSIM']:.4f} ± {stds[f'{model}_{ch}_SSIM']:.4f}"
            )
        print()


    # Convert detailed per-channel summary to DataFrame
    summary_df = pd.DataFrame(summary_rows)

    # ==============================
    # Overall mean ± std across all channels per model
    # ==============================

    overall_summary = summary_df.groupby("Model").agg(
        RMSE_mean=("RMSE_mean", "mean"),
        RMSE_std=("RMSE_mean", "std"),
        R2_mean=("R2_mean", "mean"),
        R2_std=("R2_mean", "std"),
        PSNR_mean=("PSNR_mean", "mean"),
        PSNR_std=("PSNR_mean", "std"),
        SSIM_mean=("SSIM_mean", "mean"),
        SSIM_std=("SSIM_mean", "std")
    )

    print("\n================ OVERALL SUMMARY ================\n")
    for model, row in overall_summary.iterrows():
        print(
            f"{model:<8}  "
            f"RMSE: {row['RMSE_mean']:.4f} ± {row['RMSE_std']:.4f}, "
            f"R²: {row['R2_mean']:.4f} ± {row['R2_std']:.4f}, "
            f"PSNR: {row['PSNR_mean']:.4f} ± {row['PSNR_std']:.4f}, "
            f"SSIM: {row['SSIM_mean']:.4f} ± {row['SSIM_std']:.4f}"
        )

    # Save overall summary
    overall_summary_path = os.path.join(cfg.workpath, "other_models", "test_metrics_summary.csv")
    overall_summary.to_csv(overall_summary_path, index=True)
    print(f"\n Overall summary saved to {overall_summary_path}")

    return df, overall_summary

# ----------------------------
# Main
# ----------------------------
if __name__ == '__main__':
    # train()
    evaluate_test_set()


