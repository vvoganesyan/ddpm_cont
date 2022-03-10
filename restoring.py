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
train_loader, val_loader = get_datasets(config)

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

z_restored, z = restore_image(x_noised, 0.22, device, eps_th)
