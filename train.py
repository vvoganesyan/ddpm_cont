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

model_dict = dict()
model_dict['nf'] = 128
model_dict['ch_mult'] = (1, 2, 2)
model_dict['num_res_blocks'] = 2
model_dict['attn_resolutions'] = (16,)
model_dict['dropout'] = 0.1
model_dict['resamp_with_conv'] = True
model_dict['conditional'] = True
model_dict['nonlinearity'] = 'swish'
model_dict['sigma_max'] = 50
model_dict['sigma_min'] = 0.01
model_dict['num_scales'] = 1000
model_dict['savepath'] = 'ddpm'
data_dict = dict()
data_dict['image_size'] = 32
data_dict['num_channels'] = 3
data_dict['centered'] = True
train_dict = dict()
train_dict['grad_clip'] = 1.0
train_dict['warmup'] = 5000
train_dict['lr'] = 2e-4
config_dict = dict()
config_dict['model'] = dotdict(model_dict)
config_dict['data'] = dotdict(data_dict)
config_dict['train'] = dotdict(train_dict)
config = dotdict(config_dict)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
wandb.login()
train_loader, val_loader = get_datasets()

# eps_th = nn.DataParallel(Unet(config))
eps_th = Unet(config)
eps_th.to(device)

optim = torch.optim.Adam(eps_th.parameters(), lr=2e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
ema_ = ema.ExponentialMovingAverage(eps_th.parameters(), decay=0.9999)
sched = lrsc.StepLR(optim, step_size=30, gamma=0.1)

wandb.init(project='sota_cifar')
train(eps_th, train_loader, optim, ema_, 1000, 0, 1.0, device, config)