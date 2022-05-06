import os
import shutil

import chanfig
import matplotlib.pyplot as plt

from rpin.runner import Runner
from rpin.utils.config import _C as cfg

plt.switch_backend('agg')


if __name__ == '__main__':
    parser = chanfig.ConfigParser(description='RPIN Parameters')
    parser.add_argument('--config', required=True, help='path to config file', type=str)
    parser.add_argument('--checkpoint', type=str, help='(optionally) path to pretrained model', default='')
    parser.add_argument('--output', type=str, help='output name')
    parser.add_argument('--seed', type=int, help='set random seed use this command', default=0)
    parser.add_argument('--train', action='store_true', help='train model', default=True)
    parser.add_argument('--val', action='store_true', help='train model', default=True)
    parser.add_argument('-g', '--gradient_clip', type=float, default=1.0)
    config = parser.parse_config(config=cfg)
    runner = Runner(config)
    runner.train()
