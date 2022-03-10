from utils import *
import os
import shutil
from PIL import Image
from pytorch_fid import fid_score
import copy
import torch.distributions as D
import torch.optim.lr_scheduler as lrsc
import wandb
import matplotlib.pyplot as plt
from tqdm.auto import tqdm, trange

from config import *
config = get_configs()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
wandb.login()
train_loader, val_loader = get_datasets()

eps_th = Unet(config)
eps_th.to(device)
ema_ = ema.ExponentialMovingAverage(eps_th.parameters(), decay=0.9999)
state = torch.load(config.model.savepath)
eps_th.load_state_dict(state['model'], strict=False)
ema_.load_state_dict(state['ema'])
ema_.copy_to(eps_th.parameters())
eps_th.eval()

val_dataset_iter = iter(val_loader)
x, y = val_dataset_iter.next()
x = x.to(device)
x_noised = noising_PET_image( x, 60.0/90, 2.0/3)

t0 = 0.22
t0_array = t0*torch.ones(128,).to(device)
z = SDE_noise(x_noised, t0_array)

z_restored = sample_sde_mid(device, eps_th, z, t0, 0.0)
err = torch.abs(x - z_restored)-1
n = 16 #number of images from batch to save
big_img = np.zeros((n*config.data.image_size,5*config.data.image_size),dtype=np.uint8)
for i, batch in enumerate([x, x_noised, z, z_restored, err]):
    for j, p in enumerate(batch):
        if j<n:
            p = p/20.0
            p = p * 255
            p = p.clamp(0, 255)
            p = p.detach().cpu().numpy()
            p = p.astype(np.uint8)
            p = p.transpose((1,2,0))
            big_img[j*config.data.image_size:(j+1)*config.data.image_size, i*config.data.image_size:(i+1)*config.data.image_size] = p[:,:,0]
big_img = Image.fromarray(big_img)
big_img.save(f"restoring_image.png", format="png")