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

with torch.no_grad():
    save_callable(lambda: sample_sde(device, eps_th, config.data.image_size, config.data.num_channels, 128), "sampled_images", 128)
