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

# eps_th = nn.DataParallel(Unet(config))
eps_th = Unet(config)
eps_th.to(device)

optim = torch.optim.Adam(eps_th.parameters(), lr=2e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
ema_ = ema.ExponentialMovingAverage(eps_th.parameters(), decay=0.9999)
sched = lrsc.StepLR(optim, step_size=30, gamma=0.1)

wandb.init(project='sota_cifar')
train(eps_th, train_loader, optim, ema_, 1000, 0, 1.0, device, config)