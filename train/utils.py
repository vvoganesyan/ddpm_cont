from torch import nn
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from models import ddpm
from models import ema
from tqdm.auto import tqdm, trange
import torch
import numpy as np
import wandb

# Model
class Unet(nn.Module):
    def __init__(self, config):
        super(Unet, self).__init__()
        
        self.unet = ddpm.DDPM(config)
        
    def forward(self, t, x):
        bs = x.shape[0]
        
        t = t.reshape(-1)
        t = t.expand(bs)
        
        return self.unet(x, t)

# DDPM

# def alpha(t):
#     return torch.exp(-5*t**2)
    
# def sigma_2(t):
#     return -torch.expm1(-10*t**2)

def snr(t):
    return 1 / torch.expm1(1e-4 + 10 * t ** 2)

def alpha(t):
    return (snr(t) / (1 + snr(t))) ** 0.5
    
def sigma_2(t):
    return 1 / (1 + snr(t))


def s(eps_th, t, x):
    return - eps_th(t, x) / sigma_2(t) ** 0.5

# Training

## DDPM

def downsample(x):
    m = nn.AvgPool2d(2)
    return m(x)

def log_losses(title, loss, step):
    title = f'{title}_losses'
    
    if loss is not None:
        wandb.log({f'{title}/loss_sm': loss}, step=step)

def calc_losses(eps_th, x, downsample_step=0, T_ds=0.15):
    device = x.device
    bs = x.shape[0]
    data_shape = x.shape[1:]
    
    xd = x
    
    
    if downsample_step == 0:
        t = T_ds*torch.rand(bs).to(device)
        image_size = 32
        s_2 = sigma_2(t)
    else:
        t = (1-T_ds)*torch.rand(bs).to(device) + T_ds
        xd = downsample(xd)
        image_size = 16
        s_2 = sigma_2(t)/4
    eps = torch.randn_like(xd).to(device)
    a = alpha(t)
    z = a[:, None, None, None] * xd + s_2[:, None, None, None] ** 0.5 * eps
    
    loss_sm = (t/s_2) * ((eps - eps_th(t, z)) ** 2).sum(dim=(1, 2, 3))    
    loss_sm = loss_sm.mean()

    return loss_sm

def train(eps_th, train_loader, optim, ema, epochs, downsample_step, T_ds, expname, device):    
    step = 0
    grad_clip = 1.0
    warmup = 5000
    lr = 2e-4
    for epoch in trange(epochs):
        eps_th.train()
        for x, _ in train_loader:
            x = x.to(device)
            loss_total = calc_losses(eps_th, x, downsample_step=downsample_step, T_ds=T_ds)
            optim.zero_grad()
            loss_total.backward()

            if warmup > 0:
                for g in optim.param_groups:
                    g['lr'] = lr * np.minimum(step / warmup, 1.0)
            if grad_clip >= 0:
                torch.nn.utils.clip_grad_norm_(eps_th.parameters(), max_norm=grad_clip)
            optim.step()
            ema.update(eps_th.parameters())
            log_losses('Train', loss_total, step)
            step += 1
        torch.save(eps_th.state_dict(), expname+'_model')
        torch.save(ema.state_dict(), expname+'_ema')
        torch.save(optim.state_dict(), expname+'_optim')

        wandb.log({'epoch': epoch}, step=step)
    
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
def get_datasets(BATCH_SIZE = 128):
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_data = CIFAR10(root='./data/', train=True, download=True, transform=transform)
    val_data = CIFAR10(root='./data/', train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        drop_last=True
    )
    return train_loader, val_loader