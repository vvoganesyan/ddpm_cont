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

def save_img(p, path, num):
    mkdir(path)
    sc = torch.tensor([0.4914, 0.4822, 0.4465])
    m = torch.tensor([0.2023, 0.1994, 0.2010])
    for i in range(3):
        p[i,:,:] = p[i,:,:]*m[i] + sc[i]
    p = p * 255
    p = p.clamp(0, 255)
    p = p.detach().cpu().numpy()
    p = p.astype(np.uint8)
    p = p.transpose((1,2,0))
    p = Image.fromarray(p, mode='RGB')
    p.save(f"{path}/{num}.png", format="png")
    
def save_batch(x, path, num):
    for p in x:
        save_img(p, path, num)
        num += 1
    return num

def save_dataloader(loader, path, n=2048):
    m = 0
    for x, _ in loader:
        m = save_batch(x, path, m)
        if m >= n:
            break
            
def save_callable(foo, path, n=2048):
    m = 0
    while m < n:
        m = save_batch(foo(), path, m)
        
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
def solve_sde(x, f, g, ts=0, tf=1, dt=1e-3):
    for t in np.arange(ts, tf, dt):
        tt = torch.FloatTensor([t])[0].to(device)
        z = torch.randn_like(x).to(device)
        x = x + f(tt, x) * dt + g(tt, x) * z * abs(dt) ** 0.5    
    return x
def sample_sde():
    x = solve_sde(
        0.5*torch.randn(128, 3, 16, 16).to(device),
        f=lambda t, x: f(t) * x - g_2(t) * s(eps_th, t, x)/2,
        g=lambda t, x: g_2(t) ** 0.5/2,
        ts=1, tf=0.2, dt=-1e-3
    )
    return x
def sample_sde_mid(x_mid, ts, tf):
    x = solve_sde(
        x_mid,
        f=lambda t, x: f(t) * x - g_2(t) * s(eps_th_big, t, x),
        g=lambda t, x: g_2(t) ** 0.5,
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

