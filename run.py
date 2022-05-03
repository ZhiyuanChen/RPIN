import os
import random
import shutil

import danling as dl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils import data

from rpin import datasets
from rpin.models import *
from rpin.runner import Runner
from rpin.utils.config import _C as cfg
from rpin.utils.logger import git_diff_config, setup_logger

plt.switch_backend('agg')


def arg_parse():
    parser = dl.ArgumentParser(description='RPIN Parameters')
    parser.add_argument('--cfg', required=True, help='path to config file', type=str)
    parser.add_argument('--checkpoint', type=str, help='(optionally) path to pretrained model', default='')
    parser.add_argument('--output', type=str, help='output name')
    parser.add_argument('--seed', type=int, help='set random seed use this command', default=0)
    parser.add_argument('--train', action='store_true', help='train model', default=True)
    parser.add_argument('--val', action='store_true', help='train model', default=True)
    parser.add_argument('--num_gpus', type=int, help='output name', default=4)
    parser.add_argument('-g', '--gradient_clip', type=float, default=1.0)
    return parser.parse_all_args()


def main():
    # this wrapper file contains the following procedure:
    # 1. setup training environment
    # 2. setup config
    # 3. setup logger
    # 4. setup model
    # 5. setup optimizer
    # 6. setup dataset

    # ---- setup training environment
    args = arg_parse()

    # ---- setup config files
    cfg.merge_from_file(args.cfg)
    cfg.SOLVER.BASE_LR *= args.num_gpus
    cfg.SOLVER.MAX_ITERS //= args.num_gpus
    cfg.SOLVER.VAL_INTERVAL //= args.num_gpus
    cfg.SOLVER.WARMUP_ITERS //= args.num_gpus
    cfg.freeze()
    os.makedirs(args.experiment_dir, exist_ok=True)
    shutil.copy(args.cfg, os.path.join(args.experiment_dir, 'config.yaml'))
    shutil.copy(os.path.join('rpin/models/', cfg.RPIN.ARCH + '.py'), os.path.join(args.experiment_dir, 'arch.py'))

    # ---- setup model
    model = eval(cfg.RPIN.ARCH + '.Net')()

    # ---- setup optimizer
    vae_params = [p for p_name, p in model.named_parameters() if 'vae_lstm' in p_name]
    other_params = [p for p_name, p in model.named_parameters() if 'vae_lstm' not in p_name]
    Optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER)
    if cfg.SOLVER.OPTIMIZER in ('SGD', 'RMSprop'):
        optimizer = Optimizer(
            [{'params': vae_params, 'weight_decay': 0.0}, {'params': other_params}],
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            momentum=cfg.SOLVER.MOMEMTUM
        )
    else:
        optimizer = Optimizer(
            [{'params': vae_params, 'weight_decay': 0.0}, {'params': other_params}],
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )

    # ---- setup dataset in the last, and avoid non-deterministic in data shuffling order
    train_set = getattr(datasets, cfg.DATASET_ABS)(data_root=cfg.DATA_ROOT, split='train', image_ext=cfg.RPIN.IMAGE_EXT)
    val_set = getattr(datasets, cfg.DATASET_ABS)(data_root=cfg.DATA_ROOT, split='test', image_ext=cfg.RPIN.IMAGE_EXT)
    kwargs = {'pin_memory': True, 'num_workers': 4}
    train_loader = data.DataLoader(
        train_set, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True, **kwargs,
    )
    val_loader = data.DataLoader(
        val_set, batch_size=1 if cfg.RPIN.VAE else cfg.SOLVER.BATCH_SIZE, shuffle=False, **kwargs,
    )
    print(f'size: train {len(train_loader)} / test {len(val_loader)}')

    # ---- setup trainer
    kwargs = {'config': args,
              'model': model,
              'optimizer': optimizer,
              'train_loader': train_loader,
              'val_loader': val_loader,
              'max_iters': cfg.SOLVER.MAX_ITERS}
    runner = Runner(**kwargs)
    runner.train()


if __name__ == '__main__':
    main()
