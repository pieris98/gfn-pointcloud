import torch
import random
import numpy as np
import math
import PIL

from gflownet_losses import *
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import Compose, SamplePoints, NormalizeScale
from torch.utils.data import DataLoader
def load_modelnet(num_points=2048, batch_size=32, num_classes=10):
    """
    Load the ModelNet dataset using PyTorch Geometric.
    
    Args:
        num_points (int): Number of points to sample from each mesh
        batch_size (int): Batch size for the data loader
        num_classes (int): Number of classes (10 or 40)
    
    Returns:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
    """
    # Define the transformations
    transforms = Compose([
        SamplePoints(num_points, include_normals=True),
        NormalizeScale()
    ])

    # Load training dataset
    train_dataset = ModelNet(
        root='data/ModelNet{}'.format(num_classes),
        name=str(num_classes),
        train=True,
        transform=transforms
    )

    # Load test dataset
    test_dataset = ModelNet(
        root='data/ModelNet{}'.format(num_classes),
        name=str(num_classes),
        train=False,
        transform=transforms
    )

    class PointCloudDataset(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            data = self.dataset[idx]
            # Extract xyz coordinates and reshape to (3, num_points)
            points = data.pos.T  # Transpose from (num_points, 3) to (3, num_points)
            return points, data.y

    # Wrap the datasets with our custom dataset class
    train_dataset = PointCloudDataset(train_dataset)
    test_dataset = PointCloudDataset(test_dataset)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, test_loader


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def cal_subtb_coef_matrix(lamda, N):
    """
    diff_matrix: (N+1, N+1)
    0, 1, 2, ...
    -1, 0, 1, ...
    -2, -1, 0, ...

    self.coef[i, j] = lamda^(j-i) / total_lambda  if i < j else 0.
    """
    range_vals = torch.arange(N + 1)
    diff_matrix = range_vals - range_vals.view(-1, 1)
    B = np.log(lamda) * diff_matrix
    B[diff_matrix <= 0] = -np.inf
    log_total_lambda = torch.logsumexp(B.view(-1), dim=0)
    coef = torch.exp(B - log_total_lambda)
    return coef


def logmeanexp(x, dim=0):
    return x.logsumexp(dim) - math.log(x.shape[dim])


def dcp(tensor):
    return tensor.detach().cpu()


def gaussian_params(tensor):
    mean, logvar = torch.chunk(tensor, 2, dim=-1)
    return mean, logvar


def fig_to_image(fig):
    fig.canvas.draw()

    return PIL.Image.frombytes(
        "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
    )


def get_gfn_optimizer(gfn_model, lr_policy, lr_flow, lr_back, back_model=False, conditional_flow_model=False, use_weight_decay=False, weight_decay=1e-7, energy=None):
    param_groups = [ {'params': gfn_model.t_model.parameters()},
                     {'params': gfn_model.s_model.parameters()},
                     {'params': gfn_model.joint_model.parameters()},
                     {'params': gfn_model.langevin_scaling_model.parameters()} ]
    if conditional_flow_model or energy=='vae' or energy=='vae_pointcloud':
        param_groups += [ {'params': gfn_model.flow_model.parameters(), 'lr': lr_flow} ]
    else:
        param_groups += [ {'params': [gfn_model.flow_model], 'lr': lr_flow} ]

    if back_model:
        param_groups += [ {'params': gfn_model.back_model.parameters(), 'lr': lr_back} ]

    if use_weight_decay:
        gfn_optimizer = torch.optim.Adam(param_groups, lr_policy, weight_decay=weight_decay)
    else:
        gfn_optimizer = torch.optim.Adam(param_groups, lr_policy)
    return gfn_optimizer


def get_gfn_forward_loss(mode, init_state, gfn_model, log_reward, coeff_matrix, exploration_std=None, return_exp=False, condition=None, repeats=10):
    if mode == 'tb':
        loss = fwd_tb(init_state, gfn_model, log_reward, exploration_std, return_exp=return_exp, condition=condition)
    elif mode == 'tb-avg':
        loss = fwd_tb_avg(init_state, gfn_model, log_reward, exploration_std, return_exp=return_exp, condition=condition)
    elif mode == 'cond-tb-avg':
        loss = fwd_tb_avg_cond(init_state, gfn_model, log_reward, exploration_std, return_exp=return_exp, condition=condition, repeats=repeats)
    elif mode == 'db':
        loss = db(init_state, gfn_model, log_reward, exploration_std, condition=condition)
    elif mode == 'subtb':
        loss = subtb(init_state, gfn_model, log_reward, coeff_matrix, exploration_std, condition=condition)
    return loss


def get_gfn_backward_loss(mode, samples, gfn_model, log_reward, exploration_std=None, condition=None, repeats=10):
    if mode == 'tb':
        loss = bwd_tb(samples, gfn_model, log_reward, exploration_std, condition=condition)
    elif mode == 'tb-avg':
        loss = bwd_tb_avg(samples, gfn_model, log_reward, exploration_std, condition=condition)
    elif mode == 'cond-tb-avg':
        loss = bwd_tb_avg_cond(samples, gfn_model, log_reward, exploration_std, condition=condition, repeats=repeats)
    elif mode == 'mle':
        loss = bwd_mle(samples, gfn_model, log_reward, exploration_std, condition=condition)
    return loss


def get_exploration_std(iter, exploratory, epochs, exploration_factor=0.1, exploration_wd=False):
    if exploratory is False:
        return None
    if exploration_wd:
        exploration_std = exploration_factor * max(0, 1. - (iter / (epochs/2)))
    else:
        exploration_std = exploration_factor
    expl = lambda x: exploration_std
    return expl


def get_name(args):
    name = ''
    if args.langevin:
        name = f'langevin_'
        if args.langevin_scaling_per_dimension:
            name = f'langevin_scaling_per_dimension_'
    if args.exploratory and (args.exploration_factor is not None):
        if args.exploration_wd:
            name = f'exploration_wd_{args.exploration_factor}_{name}_'
        else:
            name = f'exploration_{args.exploration_factor}_{name}_'

    if args.learn_pb:
        name = f'{name}learn_pb_scale_range_{args.pb_scale_range}_'

    if args.clipping:
        name = f'{name}clipping_lgv_{args.lgv_clip}_gfn_{args.gfn_clip}_'

    if args.mode_fwd == 'subtb':
        mode_fwd = f'subtb_subtb_lambda_{args.subtb_lambda}'
        if args.partial_energy:
            mode_fwd = f'{mode_fwd}_{args.partial_energy}'
    else:
        mode_fwd = args.mode_fwd

    if args.both_ways:
        ways = f'fwd_bwd/fwd_{mode_fwd}_bwd_{args.mode_bwd}'
    elif args.bwd:
        ways = f'bwd/bwd_{args.mode_bwd}'
    else:
        ways = f'fwd/fwd_{mode_fwd}'

    if args.local_search:
        local_search = f'local_search_iter_{args.max_iter_ls}_burn_{args.burn_in}_cycle_{args.ls_cycle}_step_{args.ld_step}_beta_{args.beta}_rankw_{args.rank_weight}_prioritized_{args.prioritized}'
        ways = f'{ways}/{local_search}'

    if args.pis_architectures:
        results = 'results_pis_architectures'
    else:
        results = 'results'

    name = f'{results}/{args.energy}/{name}gfn/{ways}/T_{args.T}/tscale_{args.t_scale}/lvr_{args.log_var_range}/'

    name = f'{name}/seed_{args.seed}/'

    return name
