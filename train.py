import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from einops import rearrange

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class HaarDWT(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x1, x2, x3, x4 = x[:,:,0::2,0::2], x[:,:,1::2,0::2], x[:,:,0::2,1::2], x[:,:,1::2,1::2]
        return (x1+x2+x3+x4)/2, (x1-x2+x3-x4)/2, (x1+x2-x3-x4)/2, (x1-x2-x3+x4)/2

class HaarIDWT(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, ll, lh, hl, hh):
        B, C, H, W = ll.shape
        out = torch.zeros(B, C, H*2, W*2, device=ll.device)
        out[:,:,0::2,0::2], out[:,:,1::2,0::2] = (ll+lh+hl+hh)/2, (ll-lh+hl-hh)/2
        out[:,:,0::2,1::2], out[:,:,1::2,1::2] = (ll+lh-hl-hh)/2, (ll-lh-hl+hh)/2
        return out

class WaveletSpectralTokenizer(nn.Module):
    def __init__(self, in_c, lat_d):
        super().__init__()
        self.dwt = HaarDWT()
        self.comp = nn.Conv2d(in_c, lat_d, 1)
    def forward(self, x):
        ll, lh, hl, hh = self.dwt(x)
        return self.comp(ll), {'lh': lh, 'hl': hl, 'hh': hh}

class DDMUNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(dim, 64, 3, 1, 1), nn.ReLU(), nn.Conv2d(64, dim, 3, 1, 1))
    def forward(self, x, cond):
        return self.net(x + cond)

class WFMDiff(nn.Module):
    def __init__(self, channels=128, latent=32):
        super().__init__()
        self.tokenizer = WaveletSpectralTokenizer(channels, latent)
        self.unet = DDMUNet(latent)
        self.expander = nn.Conv2d(latent, channels, 1)
        self.idwt = HaarIDWT()
    def forward(self, lr, pan, t, noise_scheduler):
        z_lr, z_res = self.tokenizer(lr)
        hsi_up = F.interpolate(z_lr, size=pan.shape[-2:], mode='bicubic', align_corners=False)
        noise = torch.randn_like(hsi_up)
        z_t = noise_scheduler.add_noise(hsi_up, noise, t)
        pred_noise = self.unet(z_t, hsi_up)
        z_0_hat = z_t - pred_noise
        sr = self.idwt(self.expander(z_0_hat), z_res['lh'], z_res['hl'], z_res['hh'])
        return sr, pred_noise, noise

class DiffusionScheduler:
    def __init__(self, steps=1000):
        self.steps = steps
        self.betas = torch.linspace(1e-4, 0.02, steps)
        self.alpha_hat = torch.cumprod(1. - self.betas, dim=0)
    def add_noise(self, x0, noise, t):
        sqrt_ah = torch.sqrt(self.alpha_hat[t])[:, None, None, None].to(x0.device)
        sqrt_om = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None].to(x0.device)
        return sqrt_ah * x0 + sqrt_om * noise

class CombinedLoss(nn.Module):
    def __init__(self, l_diff=1.0, l_pan=0.1, l_spec=0.01):
        super().__init__()
        self.l_diff, self.l_pan, self.l_spec = l_diff, l_pan, l_spec
    def forward(self, sr, gt, pred_n, true_n, pan):
        loss_diff = F.mse_loss(pred_n, true_n)
        loss_pan = F.l1_loss(sr.mean(1, keepdim=True), pan)
        loss_spec = F.mse_loss(sr, gt)
        return self.l_diff * loss_diff + self.l_pan * loss_pan + self.l_spec * loss_spec

class HSIDataset(Dataset):
    def __init__(self, num_samples=100, channels=128, patch_size=256):
        self.num_samples = num_samples
        self.c, self.ps = channels, patch_size
    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        gt = torch.randn(self.c, self.ps, self.ps)
        if random.random() > 0.5: gt = torch.flip(gt, dims=[1])
        if random.random() > 0.5: gt = torch.flip(gt, dims=[2])
        k = random.choice([0, 1, 2, 3])
        gt = torch.rot90(gt, k, dims=[1, 2])
        lr = F.interpolate(gt.unsqueeze(0), scale_factor=0.5, mode='bicubic', align_corners=False).squeeze(0)
        pan = gt.mean(0, keepdim=True)
        return lr, pan, gt

def train():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = WFMDiff().to(device)
    scheduler_diff = DiffusionScheduler(steps=1000)
    criterion = CombinedLoss(l_diff=1.0, l_pan=0.1, l_spec=0.01).to(device)
    
    optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler_lr = CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)
    scaler = GradScaler()
    
    dataset = HSIDataset(num_samples=800, patch_size=256)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
    
    for epoch in range(200):
        model.train()
        epoch_loss = 0
        for lr_img, pan_img, gt_img in dataloader:
            lr_img, pan_img, gt_img = lr_img.to(device), pan_img.to(device), gt_img.to(device)
            t = torch.randint(0, 1000, (lr_img.shape[0],), device=device)
            
            optimizer.zero_grad()
            with autocast():
                sr, pred_n, true_n = model(lr_img, pan_img, t, scheduler_diff)
                loss = criterion(sr, gt_img, pred_n, true_n, pan_img)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
        scheduler_lr.step()
        print(f"Epoch [{epoch+1}/200] - Loss: {epoch_loss/len(dataloader):.6f} - LR: {optimizer.param_groups[0]['lr']:.8f}")
        
        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), f"wfmdiff_best.pth")

if __name__ == "__main__":
    train()