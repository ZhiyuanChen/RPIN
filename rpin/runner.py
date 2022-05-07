import math
import os
import shutil
from timeit import default_timer as timer

import danling as dl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import gather, gather_object
from torch.utils import data
from tqdm import tqdm

from rpin import datasets
from rpin.models import *
from rpin.utils.bbox import xyxy_to_posf, xyxy_to_rois
from rpin.utils.config import _C as C


class Runner(dl.runner.BaseRunner):
    """
    runner
    """
    def __init__(self, config):
        self.batch_size = config.SOLVER.BATCH_SIZE
        self.input_size = config.RPIN.INPUT_SIZE
        self.ptrain_size, self.ptest_size = config.RPIN.PRED_SIZE_TRAIN, config.RPIN.PRED_SIZE_TEST
        self.input_height, self.input_width = config.RPIN.INPUT_HEIGHT, config.RPIN.INPUT_WIDTH
        # train loop settings
        self.iter_end = config.SOLVER.MAX_ITERS
        self.val_interval = config.SOLVER.VAL_INTERVAL
        self.gratient_clip = config.SOLVER.GRADIENT_CLIP
        super().__init__(config)

        if self.is_main_process:
            shutil.copy(os.path.join('rpin/models/', config.RPIN.ARCH + '.py'), os.path.join(self.dir, 'arch.py'))

        # ---- setup model
        model = eval(self.config.RPIN.ARCH + '.Net')()

        # ---- setup optimizer
        vae_params = [p for n, p in model.named_parameters() if 'vae_lstm' in n]
        other_params = [p for n, p in model.named_parameters() if 'vae_lstm' not in n]
        Optimizer = getattr(torch.optim, self.config.SOLVER.OPTIMIZER)
        if self.config.SOLVER.OPTIMIZER in ('SGD', 'RMSprop'):
            optimizer = Optimizer(
                [{'params': vae_params, 'weight_decay': 0.0}, {'params': other_params}],
                lr=self.config.SOLVER.BASE_LR,
                weight_decay=self.config.SOLVER.WEIGHT_DECAY,
                momentum=self.config.SOLVER.MOMEMTUM
            )
        else:
            optimizer = Optimizer(
                [{'params': vae_params, 'weight_decay': 0.0}, {'params': other_params}],
                lr=self.config.SOLVER.BASE_LR,
                weight_decay=self.config.SOLVER.WEIGHT_DECAY,
            )
        if checkpoint := getattr(config, 'checkpoint', None):
            self.load(checkpoint)
        self.model, self.optimizer = self.prepare(model, optimizer)
        self.scheduler = dl.Scheduler(self.optimizer, self.config.SOLVER.MAX_ITERS // (self.batch_size_actual), policy=self.config.SOLVER.SCHEDULER)

        # ---- setup dataset in the last, and avoid non-deterministic in data shuffling order
        Dataset = getattr(datasets, self.config.DATASET_ABS)
        train_set = Dataset(data_root=self.config.DATA_ROOT, split='train', image_ext=self.config.RPIN.IMAGE_EXT)
        val_set = Dataset(data_root=self.config.DATA_ROOT, split='test', image_ext=self.config.RPIN.IMAGE_EXT)
        kwargs = {'pin_memory': True, 'num_workers': 16}
        train_loader = data.DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True, **kwargs,)
        val_loader = data.DataLoader(
            val_set, batch_size=1 if self.config.RPIN.VAE else self.batch_size, shuffle=False, **kwargs,)
        print(f'size: train {len(train_loader)} / test {len(val_loader)}')
        self.dataloaders['train'], self.dataloaders['val'] = self.prepare(train_loader, val_loader)

        # loss settings
        self.train_loss = Loss(self)
        self.eval_loss = Loss(self)
        self.vae_loss = Loss(self) if self.config.RPIN.VAE else None
        # timer setting
        self.score_best = 1e6
        self.save_freq = 1
        self.time = timer()
        self.timer = dl.metrics.AverageMeter()

    def train(self):
        """
        train
        """
        print_msg = "| ".join(["progress  | mean "] + list(map("{:6}".format, self.train_loss.ams.keys())))
        self.model.train()
        print('\r', end='')
        print(print_msg)
        while self.iters < self.iter_end:
            self.train_epoch()
            self.epochs += 1

    def train_epoch(self):
        """
        train_epoch
        """
        for batch_idx, (data, data_t, rois, gt_boxes, gt_masks, valid, g_idx, seq_l) in enumerate(self.dataloaders['train']):
            rois = xyxy_to_rois(rois, batch=data.shape[0], time_step=data.shape[1], num_devices=self.num_processes)
            outputs = self.model(data, rois, num_rollouts=self.ptrain_size, g_idx=g_idx, x_t=data_t, phase='train')
            labels = {'boxes': gt_boxes, 'masks': gt_masks, 'valid': valid, 'seq_l': seq_l.squeeze(),}
            loss = self.train_loss(outputs, labels, 'train')
            self.backward(loss)
            self.step()
            # this is an approximation for printing; the dataset size may not divide the batch size

            mean_loss = self.train_loss.score()
            speed = self.iters / (timer() - self.time)
            eta = (self.iter_end - self.iters) / speed / 3600

            print_msg = f"{self.epochs:02}/{self.iters // 1000:04}k | {mean_loss:.3f} | " 
            print_msg += f" | ".join(["{:.3f}".format(v) for k, v in self.train_loss.val()])
            print_msg += f" | speed: {speed:.1f} | eta: {eta:.2f} h"
            print(print_msg)

            if self.iters % self.val_interval == 0:
                self.train_loss.reset()
                self.evaluate()
                self.model.train()

            if self.iters >= self.iter_end:
                break

    @torch.no_grad()
    def evaluate(self):
        """
        evaluate
        """
        self.model.eval()
        self.eval_loss.reset()

        if self.config.RPIN.VAE:
            self.vae_loss.reset()

        inf, gt = [], []
        for batch_idx, (data, _, rois, gt_boxes, gt_masks, valid, g_idx, seq_l) in enumerate(tqdm(self.dataloaders['val'])):

            rois = xyxy_to_rois(rois, batch=data.shape[0], time_step=data.shape[1], num_devices=self.num_processes)
            labels = {'boxes': gt_boxes, 'masks': gt_masks, 'valid': valid, 'seq_l': seq_l.squeeze(),}

            outputs = self.model(data, rois, num_rollouts=self.ptest_size, g_idx=g_idx, phase='test')
            gt.extend(labels['seq_l'].tolist())
            inf.extend(outputs['score'].tolist())
            self.eval_loss(outputs, labels, 'test')
            # VAE multiple runs
            if self.config.RPIN.VAE:
                vae_best_mean = self.eval_loss.score()
                for i in range(9):
                    outputs = self.model(data, rois, num_rollouts=self.ptest_size, g_idx=g_idx, phase='test')
                    self.vae_loss(outputs, labels, 'test')
                    mean_loss = self.vae_loss.score()
                    if mean_loss < vae_best_mean:
                    #     losses_t = self.losses.copy()
                    #     box_p_step_losses_t = self.box_p_step_losses.copy()
                    #     masks_step_losses_t = self.masks_step_losses.copy()
                        vae_best_mean = mean_loss

        # if self.config.RPIN.VAE:
        #     self.losses = losses.copy()
        #     self.box_p_step_losses = box_p_step_losses.copy()

        self.score_last = self.eval_loss.score()
        self.result_last = self.eval_loss.avg()
        if self.is_best:
            self.result_best = self.score_last
        self.save()
        print_msg = f"{self.epochs:03}/{self.iters // 1000:04}k | {self.score_last:.3f} | "
        print_msg += f" | ".join(["{:.3f}".format(v) for k, v in self.eval_loss.val()])
        if self.config.RPIN.SEQ_CLS_LOSS_WEIGHT:
            inf, gt = torch.tensor(gather_object(inf)), torch.tensor(gather_object(gt))
            acc = (inf >= 0.5).eq(gt)
            fg_acc = acc[gt == 1].sum().item() / ((gt == 1).sum().item() + 1e-7)
            bg_acc = acc[gt == 0].sum().item() / ((gt == 0).sum().item() + 1e-7)
            print_msg += f" | {fg_acc:.3f} | {bg_acc:.3f}"
        print(print_msg)


class Loss(nn.Module):
    """
    loss
    """

    def __init__(self, runner):
        self.runner = runner
        self.position_weight = runner.config.RPIN.POSITION_LOSS_WEIGHT
        self.mask_weight = runner.config.RPIN.MASK_LOSS_WEIGHT
        self.kl = runner.config.RPIN.VAE
        self.kl_weight = runner.config.RPIN.VAE_KL_LOSS_WEIGHT
        self.seq_weight = runner.config.RPIN.SEQ_CLS_LOSS_WEIGHT
        self.seq_ratio = runner.config.RPIN.SEQ_CLS_LOSS_RATIO
        self.discount_tau = runner.config.RPIN.DISCOUNT_TAU
        self.ptrain_size, self.ptest_size = runner.config.RPIN.PRED_SIZE_TRAIN, runner.config.RPIN.PRED_SIZE_TEST
        self.input_height, self.input_width = runner.config.RPIN.INPUT_HEIGHT, runner.config.RPIN.INPUT_WIDTH
        self.loss_name = ['p_1', 'p_2', 's_1', 's_2']
        if self.mask_weight:
            self.loss_name += ['m_1', 'm_2']
        if self.kl:
            self.loss_name += ['kl']
        if self.seq_weight:
            self.loss_name += ['seq', 'fg_acc', 'bg_acc']
        self.ams = {name: dl.metrics.AverageMeter(runner.batch_size_actual) for name in self.loss_name}
        super().__init__()

    def forward(self, outputs, labels, phase):
        """
        calculate loss
        """
        position_loss = self.position_loss(outputs['boxes'], labels['boxes'], labels['valid'], phase)
        mask_loss = self.mask_loss(outputs['masks'], labels['masks'], labels['valid'], phase) if self.mask_weight > 0 else 0
        seq_loss = self.seq_loss(outputs['score'], labels['seq_l']) if self.seq_weight > 0 else 0
        kl_loss = self.kl_loss(outputs['kl'], phase) if self.kl and phase == 'train' else 0
        return position_loss + mask_loss + kl_loss + seq_loss

    def position_loss(self, outputs, targets, valid, phase) -> float:
        """
        position_loss
        """
        # calculate bbox loss
        # of shape (batch, time, #obj, 4)
        loss = (outputs - targets) ** 2
        # take weighted sum over axis 2 (objs dim) since some index are not valid
        valid = valid[:, None, :, None]
        loss = loss * valid * self.position_weight
        loss = loss.sum(2) / valid.sum(2)
        box_p_step_losses = loss[:, :, :2].sum(dim=(0, 2))
        box_s_step_losses = loss[:, :, 2:].sum(dim=(0, 2))
        self.ams['p_1'].update(box_p_step_losses[:self.ptrain_size].mean())
        self.ams['p_2'].update(box_p_step_losses[self.ptrain_size:].mean() if self.ptrain_size < self.ptest_size else 0)
        self.ams['s_1'].update(box_s_step_losses[:self.ptrain_size].mean())
        self.ams['s_2'].update(box_s_step_losses[self.ptrain_size:].mean() if self.ptrain_size < self.ptest_size else 0)
        loss = self.tau(loss, phase)
        return loss

    def mask_loss(self, outputs, targets, valid, phase) -> float:
        """
        mask_loss
        """
        # of shape (batch, time, #obj, m_sz, m_sz)
        loss = F.binary_cross_entropy(outputs, targets, reduction='none')
        loss = loss.mean((3, 4))
        valid = valid[:, None, :]
        loss = loss * valid * self.mask_weight
        loss = loss.sum(2) / valid.sum(2)
        masks_step_losses = loss.sum(0)
        m1_loss = masks_step_losses[:self.ptrain_size]
        m2_loss = masks_step_losses[self.ptrain_size:]
        self.ams['m_1'].update(m1_loss.mean())
        self.ams['m_2'].update(m2_loss.mean() if self.ptrain_size < self.ptest_size else 0)
        loss = self.tau(loss, phase)
        return loss

    def seq_loss(self, outputs, targets) -> float:
        """
        seq_loss
        """
        weight = torch.ones_like(targets)
        weight[targets == 1] *= 1 / self.seq_ratio
        loss = F.binary_cross_entropy(outputs, targets, weight)
        loss *= self.seq_weight
        self.ams['seq'].update(loss.mean())
        # calculate accuracy
        inf, gt = gather(outputs), gather(targets)
        acc = (inf >= 0.5).eq(gt)
        fg_acc = acc[gt == 1].sum().item() / ((gt == 1).sum().item() + 1e-7)
        bg_acc = acc[gt == 0].sum().item() / ((gt == 0).sum().item() + 1e-7)
        self.ams['fg_acc'].update(fg_acc)
        self.ams['bg_acc'].update(bg_acc)
        return loss

    def kl_loss(self, kl_loss) -> float:
        """
        kl_loss
        """
        kl_loss = kl_loss.sum() * self.kl_weight
        self.ams['kl'].update(kl_loss)
        return kl_loss

    def tau(self, loss, phase):
        pred_size = eval(f'self.p{phase}_size')
        loss = loss.mean(0)
        init_tau = C.RPIN.DISCOUNT_TAU ** (1 / self.ptrain_size)
        tau = init_tau + (self.runner.progress) * (1 - init_tau)
        tau = torch.pow(tau, torch.arange(pred_size, out=torch.FloatTensor()))[:, None].to(self.runner.device)
        return ((loss * tau) / tau.sum(axis=0, keepdims=True)).sum()

    def val(self):
        return {k: v.val for k, v in self.ams.items()}.items()

    def avg(self):
        return {k: v.val for k, v in self.ams.items()}.items()

    def reset(self):
        for v in self.ams.values():
            v.reset()

    def score(self):
        return self.ams['p_1'].val
