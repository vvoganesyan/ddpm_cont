from torch import nn
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data import DataLoader
from models import ddpm
from models import ema
from tqdm.auto import tqdm, trange
import torch
import numpy as np
import wandb
import os
import shutil
from PIL import Image


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

def SDE_noise(x, t, device):
    a = alpha(t)
    s_2 = sigma_2(t)
    eps = torch.randn_like(x).to(device)
    z = a[:, None, None, None] * x + s_2[:, None, None, None] ** 0.5 * eps
    return z
        
        
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
    
    loss_sm = ((eps - eps_th(t, z)) ** 2).sum(dim=(1, 2, 3))    
    loss_sm = loss_sm.mean()

    return loss_sm

def train(eps_th, train_loader, optim, ema, epochs, downsample_step, T_ds, device, config):
    step = 0
    for epoch in trange(epochs):
        eps_th.train()
        for x, _ in train_loader:
            x = x.to(device)
            loss_total = calc_losses(eps_th, x, downsample_step=downsample_step, T_ds=T_ds)
            optim.zero_grad()
            loss_total.backward()

            if config.train.warmup > 0:
                for g in optim.param_groups:
                    g['lr'] = config.train.lr * np.minimum(step / config.train.warmup, 1.0)
            if config.train.grad_clip >= 0:
                torch.nn.utils.clip_grad_norm_(eps_th.parameters(), max_norm=config.train.grad_clip)
            optim.step()
            ema.update(eps_th.parameters())
            log_losses('Train', loss_total, step)
            step += 1
        torch.save({'model': eps_th.state_dict(), 'ema': ema.state_dict(), 'optim': optim.state_dict()}, config.model.savepath)

        wandb.log({'epoch': epoch}, step=step)
    
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
def get_dataset_CIFAR10(config):

    BATCH_SIZE = config.data.batch_size
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(config.data.norm_mean, config.data.norm_std)
    ])

    train_data = CIFAR10(root='../data/', train=True, download=True, transform=transform)
    val_data = CIFAR10(root='../data/', train=False, download=True, transform=transform)

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

def get_dataset_MNIST(config):

    BATCH_SIZE = config.data.batch_size
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize(config.data.norm_mean, config.data.norm_std)
    ])

    train_data = MNIST(root='../data/', train=True, download=True, transform=transform)
    val_data = MNIST(root='../data/', train=False, download=True, transform=transform)

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


@torch.no_grad()
def gen_discrete(eps_th, n=1000, bs=128):
    sp = nn.Softplus()
    sm = nn.Sigmoid()
    
    z = torch.randn(bs, 3, 32, 32, device=device)
    
    dt = 1 / n
    for t in np.linspace(1, 0, n + 1)[:-1]:
        t_t = torch.ones(1, device=device)[0] * t
        t_s = torch.ones(1)[0] * (t - dt)
        
        g_t = gamma(t_t)
        g_s = gamma(t_s)
        
        a_s_a_t_2 = torch.exp(g_t - sp(g_s)) + sm(-g_s)
        a_s_a_t = a_s_a_t_2 ** 0.5
        
        k_eps = torch.expm1(g_s - g_t) * sigma_2(t_t) ** 0.5
        
        mu_q = a_s_a_t * (z + k_eps * eps_th(t_t, z))
        
        sigma_q_2 = - sigma_2(t_s) * torch.expm1(g_s - g_t)
        sigma_q = sigma_q_2 ** 0.5
        
        eps = torch.randn_like(z, device=device)
        z = mu_q + eps * sigma_q
        
    return z

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        
def rmdir(path):
    shutil.rmtree(path)

def save_img(p, path, num, scale):
    mkdir(path)
    m = torch.tensor(scale[0])
    sc = torch.tensor(scale[1])
    for i in range(len(sc)):
        p[i,:,:] = p[i,:,:]*sc[i] + m[i]
    p = p * 255
    p = p.clamp(0, 255)
    p = p.detach().cpu().numpy()
    p = p.astype(np.uint8)
    p = p.transpose((1,2,0))
    if p.shape[-1] == 3:
        p = Image.fromarray(p, mode='RGB')
    elif p.shape[-1] == 1:
        p = p.squeeze(2)
        p = Image.fromarray(p, mode='L')
    p.save(f"{path}/{num}.png", format="png")
    
def save_batch(x, path, num, scale):
    for p in x:
        save_img(p, path, num, scale)
        num += 1
    return num

def save_dataloader(loader, path, scale, n=2048):
    m = 0
    for x, _ in loader:
        m = save_batch(x, path, m, scale)
        if m >= n:
            break
            
def save_callable(foo, path, scale, n=2048):
    m = 0
    while m < n:
        m = save_batch(foo(), path, m, scale)
        
@torch.no_grad()
def calc_fid(foo):
    path_1 = "data_1"
    path_2 = "data_2"
    
    save_dataloader(train_loader, path_1, 16*1024)
    save_callable(foo, path_2, 16*1024)
    
    res = fid_score.calculate_fid_given_paths(
        paths=[path_1, path_2],
        batch_size=128,
        device=device,
        dims=2048
    )
    
#     rmdir(path_1)
#     rmdir(path_2)
    
    return res

@torch.no_grad()
def solve_sde(device, x, f, g, ts=0, tf=1, dt=1e-3):
    for t in np.arange(ts, tf, dt):
        tt = torch.FloatTensor([t])[0].to(device)
        z = torch.randn_like(x).to(device)
        x = x + f(tt, x) * dt + g(tt, x) * z * abs(dt) ** 0.5    
    return x
def sample_sde(device, eps_th, image_size, num_channels, batch_size_sample):
    x = solve_sde(device,
        torch.randn(batch_size_sample, num_channels, image_size, image_size).to(device),
        f=lambda t, x: -10*t*x - 20*t*s(eps_th, t, x),
        g=lambda t, x: (20*t) ** 0.5,
        ts=1, tf=0.0, dt=-1e-3
    )
    return x
def sample_sde_mid(device, eps_th, x_mid, ts, tf):
    x = solve_sde(device,
        x_mid,
        f=lambda t, x: -10*t*x - 20*t*s(eps_th, t, x),
        g=lambda t, x: (20*t) ** 0.5,
        ts=ts, tf=tf, dt=-1e-3
    )
    return x

def upsample(x_dash, initial_cov, alpha_t, sigma_t_2):
    """
    :param x_dash: a tensor of pixels that are going to be conditioned by
    :param initial_cov: covariance matrix of the pixels before downsampling or any other dynamics applied.
    Best be estimated from the dataset
    :param alpha_t: contraction magnitude of the forward dynamics
    :param sigma_t_2: variance of the noise added by the forward dynamics
    """

    device = x_dash.device
    ones_vector = torch.ones((4, 1)).float().to(device)

    shape = x_dash.shape
    x_dash = x_dash.reshape(-1)

    initial_cov = torch.tensor(initial_cov, device=device, dtype=torch.float)

    cov_x_tilde = initial_cov * alpha_t**2 + torch.eye(4).to(device) * sigma_t_2
    cov_ones = cov_x_tilde @ ones_vector
    normer = cov_ones.sum()

    full_conditional_mean = 4 * x_dash[:, None] * (cov_ones / normer)[None, :, 0]
    full_conditional_covariance = cov_x_tilde - cov_x_tilde @ (ones_vector @ ones_vector.T) @ cov_x_tilde / normer

    distr = D.MultivariateNormal(full_conditional_mean[:, :3], full_conditional_covariance[None, :3, :3])

    sampled = distr.sample()

    x_4 = (4 * x_dash - sampled.sum(dim=1))[:, None]

    one_dim_sampled = torch.cat([sampled, x_4], dim=1)

    return one_dim_sampled.reshape(*shape, 4)


def upsample_image(inp, cov_matrix=None, alpha_t=1.0, sigma_t_2=0.0):
    device = inp.device
    prev_state = upsample(inp, cov_matrix.to(device), alpha_t, sigma_t_2)
    shp = inp.shape
    return prev_state.reshape(*shp, 2, 2).permute(0, 1, 2, 4, 3, 5).reshape(shp[0], shp[1], shp[2] * 2, shp[3] * 2)

def noising_PET_image(image, k, alpha):
    poisson_mean = k*(1-alpha)*image
    perturbed_image = torch.poisson(poisson_mean)
    perturbed_image = perturbed_image / k
    return alpha*image + perturbed_image

def restore_image(x_noised, t0, device, eps_th):
    t0_array = t0*torch.ones(len(x_noised),).to(device)
    z = SDE_noise(x_noised, t0_array, device)
    z_restored = sample_sde_mid(device, eps_th, z, t0, 0.0)
    return z_restored, z