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
    raw_workpath = r''
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



def safe_interp_upscale(tensor_lr, scale=6, mode='bicubic'):

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


    ae = None
    if os.path.exists(model_paths.get('AE', '')):
        try:
            from __main__ import AutoencoderSR6x  
            ae = AutoencoderSR6x(channels=cfg.channels).to(cfg.device)
            ae.load_state_dict(torch.load(model_paths['AE'], map_location=cfg.device))
            ae.eval()
            print("Loaded AE from", model_paths['AE'])
        except Exception:
            print("AE not loaded (class not found or load failed). AE will be skipped.")
            ae = None

    # ResNet 
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

    # Save and Display Metrics
    
    df = pd.DataFrame(metrics_list)
    df.to_csv(metrics_csv, index=False)
    print(f"\n Metrics saved to {metrics_csv}\n")

    # Summary (Per-channel and Overall Mean ± Std)
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


if __name__ == '__main__':
    # train()
    evaluate_test_set()


