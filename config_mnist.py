from utils import dotdict

def get_configs():
    model_dict = dotdict()
    model_dict.nf = 32
    model_dict.ch_mult = (1, 2, 2)
    model_dict.num_res_blocks = 2
    model_dict.attn_resolutions = (16,)
    model_dict.dropout = 0.1
    model_dict.resamp_with_conv = True
    model_dict.conditional = True
    model_dict.nonlinearity = 'swish'
    model_dict.sigma_max = 50
    model_dict.sigma_min = 0.01
    model_dict.num_scales = 1000
    model_dict.savepath = 'ddpm_mnist'
    data_dict = dotdict()
    data_dict.image_size = 32
    data_dict.num_channels = 1
    data_dict.centered = True
    data_dict.batch_size = 128
    data_dict.norm_mean = (0.5)
    data_dict.norm_std = (0.5)
    train_dict = dotdict()
    train_dict.grad_clip = 1.0
    train_dict.warmup = 5000
    train_dict.lr = 2e-4
    config_dict = dotdict()
    config_dict.model = model_dict
    config_dict.data = data_dict
    config_dict.train = train_dict
    return config_dict