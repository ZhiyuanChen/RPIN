import os
import math
from timeit import default_timer as timer

import danling as dl
import numpy as np
import torch
import torch.nn.functional as F

from rpin.utils.bbox import xyxy_to_posf, xyxy_to_rois
from rpin.utils.config import _C as C
from rpin.utils.misc import tprint


class Runner(dl.runner.BaseRunner):
    def __init__(self, config, train_loader, val_loader, model, optimizer, max_iters):
        super().__init__(config)
        if checkpoint := getattr(config, 'checkpoint', None):
            self.load(checkpoint)
        # data loading
        self.dataloaders['train'], self.dataloaders['val'] = self.prepare(train_loader, val_loader)
        # nn optimization
        self.model, self.optimizer = self.prepare(model, optimizer)
        self.scheduler = LRScheduler(
            self.optimizer, steps=max_iters, lr_final=C.SOLVER.MIN_LR, min_lr=C.SOLVER.MIN_LR,
            lr_policy=C.SOLVER.SCHEDULER, warmup_steps=C.SOLVER.WARMUP_ITERS)
        self.input_size = C.RPIN.INPUT_SIZE
        self.ptrain_size, self.ptest_size = C.RPIN.PRED_SIZE_TRAIN, C.RPIN.PRED_SIZE_TEST
        self.input_height, self.input_width = C.RPIN.INPUT_HEIGHT, C.RPIN.INPUT_WIDTH
        self.batch_size = C.SOLVER.BATCH_SIZE
        self.gradient_clip = C.SOLVER.GRADIENT_CLIP
        # train loop settings
        self.iterations = 0
        self.epochs = 0
        self.max_iters = max_iters
        self.val_interval = C.SOLVER.VAL_INTERVAL
        self.fg_correct, self.bg_correct, self.fg_num, self.bg_num = 0, 0, 0, 0
        # loss settings
        self._setup_loss()
        # timer setting
        self.result_best = 1e6

    def train(self):
        print_msg = "| ".join(["progress  | mean "] + list(map("{:6}".format, self.loss_name)))
        self.model.train()
        print('\r', end='')
        print(print_msg)
        while self.iterations < self.max_iters:
            self.train_epoch()
            self.epochs += 1

    def train_epoch(self):
        for batch_idx, (data, data_t, rois, gt_boxes, gt_masks, valid, g_idx, seq_l) in enumerate(self.dataloaders['train']):
            rois = xyxy_to_rois(rois, batch=data.shape[0], time_step=data.shape[1], num_devices=self.num_gpus)
            self.optimizer.zero_grad()

            outputs = self.model(data, rois, num_rollouts=self.ptrain_size, g_idx=g_idx, x_t=data_t, phase='train')
            labels = {
                'boxes': gt_boxes,
                'masks': gt_masks,
                'valid': valid,
                'seq_l': seq_l.squeeze(),
            }
            loss = self.loss(outputs, labels, 'train')
            loss.backward()
            if self.gradient_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()
            self.scheduler.step()
            # this is an approximation for printing; the dataset size may not divide the batch size
            self.iterations += self.batch_size

            print_msg = ""
            print_msg += f"{self.epochs:03}/{self.iterations // 1000:04}k"
            print_msg += f" | "
            mean_loss = np.mean(np.array(self.box_p_step_losses[:self.ptrain_size]) / self.loss_cnt) * 1e3
            print_msg += f"{mean_loss:.3f} | "
            print_msg += f" | ".join(
                ["{:.3f}".format(self.losses[name] * 1e3 / self.loss_cnt) for name in self.loss_name])
            if C.RPIN.SEQ_CLS_LOSS_WEIGHT:
                print_msg += f" | {self.fg_correct / (self.fg_num + 1e-9):.3f} | {self.bg_correct / (self.bg_num + 1e-9):.3f}"
            speed = self.loss_cnt / (timer() - self.time)
            eta = (self.max_iters - self.iterations) / speed / 3600
            print_msg += f" | speed: {speed:.1f} | eta: {eta:.2f} h"
            try:
                print_msg += (" " * (os.get_terminal_size().columns - len(print_msg) - 10))
            except OSError:
                print_msg += (" " * (100 - len(print_msg) - 10))
            print(print_msg)

            if self.iterations % self.val_interval == 0:
                self.evaluate()
                self._init_loss()
                self.model.train()

            if self.iterations >= self.max_iters:
                print('\r', end='')
                print(f'{self.result_best:.3f}')
                break

    def evaluate(self):
        self.model.eval()
        self._init_loss()

        if C.RPIN.VAE:
            losses = dict.fromkeys(self.loss_name, 0.0)
            box_p_step_losses = [0.0 for _ in range(self.ptest_size)]
            masks_step_losses = [0.0 for _ in range(self.ptest_size)]

        for batch_idx, (data, _, rois, gt_boxes, gt_masks, valid, g_idx, seq_l) in enumerate(self.dataloaders['val']):
            print(f"eval: {batch_idx}/{len(self.dataloaders['val'])}")
            with torch.no_grad():

                rois = xyxy_to_rois(rois, batch=data.shape[0], time_step=data.shape[1], num_devices=self.num_gpus)
                labels = {
                    'boxes': gt_boxes,
                    'masks': gt_masks,
                    'valid': valid,
                    'seq_l': seq_l.squeeze(),
                }

                outputs = self.model(data, rois, num_rollouts=self.ptest_size, g_idx=g_idx, phase='test')
                self.loss(outputs, labels, 'test')
                # VAE multiple runs
                if C.RPIN.VAE:
                    vae_best_mean = np.mean(np.array(self.box_p_step_losses[:self.ptest_size]) / self.loss_cnt) * 1e3
                    losses_t = self.losses.copy()
                    box_p_step_losses_t = self.box_p_step_losses.copy()
                    masks_step_losses_t = self.masks_step_losses.copy()
                    for i in range(9):
                        outputs = self.model(data, rois, num_rollouts=self.ptest_size, g_idx=g_idx, phase='test')
                        self.loss(outputs, labels, 'test')
                        mean_loss = np.mean(np.array(self.box_p_step_losses[:self.ptest_size]) / self.loss_cnt) * 1e3
                        if mean_loss < vae_best_mean:
                            losses_t = self.losses.copy()
                            box_p_step_losses_t = self.box_p_step_losses.copy()
                            masks_step_losses_t = self.masks_step_losses.copy()
                            vae_best_mean = mean_loss
                        self._init_loss()

                    for k, v in losses.items():
                        losses[k] += losses_t[k]
                    for i in range(len(box_p_step_losses)):
                        box_p_step_losses[i] += box_p_step_losses_t[i]
                        masks_step_losses[i] += masks_step_losses_t[i]

        if C.RPIN.VAE:
            self.losses = losses.copy()
            self.box_p_step_losses = box_p_step_losses.copy()
            self.loss_cnt = len(self.dataloaders['val'])

        print('\r', end='')
        print_msg = ""
        print_msg += f"{self.epochs:03}/{self.iterations // 1000:04}k"
        print_msg += f" | "
        mean_loss = np.mean(np.array(self.box_p_step_losses[:self.ptest_size]) / self.loss_cnt) * 1e3
        print_msg += f"{mean_loss:.3f} | "

        self.epoch_is_best = mean_loss < self.result_best
        self.result_last = mean_loss
        self.result_best = min(self.result_last, self.result_best)
        self.save()

        print_msg += f" | ".join(["{:.3f}".format(self.losses[name] * 1e3 / self.loss_cnt) for name in self.loss_name])
        if C.RPIN.SEQ_CLS_LOSS_WEIGHT:
            print_msg += f" | {self.fg_correct / (self.fg_num + 1e-9):.3f} | {self.bg_correct / (self.bg_num + 1e-9):.3f}"
        try:
            print_msg += (" " * (os.get_terminal_size().columns - len(print_msg) - 10))
        except OSError:
            print_msg += (" " * (100 - len(print_msg) - 10))
        print(print_msg)

    def loss(self, outputs, labels, phase):
        self.loss_cnt += labels['boxes'].shape[0]
        pred_size = eval(f'self.p{phase}_size')
        # calculate bbox loss
        # of shape (batch, time, #obj, 4)
        loss = (outputs['boxes'] - labels['boxes']) ** 2
        # take weighted sum over axis 2 (objs dim) since some index are not valid
        valid = labels['valid'][:, None, :, None]
        loss = loss * valid
        loss = loss.sum(2) / valid.sum(2)
        loss *= self.position_loss_weight

        for i in range(pred_size):
            self.box_p_step_losses[i] += loss[:, i, :2].sum().item()
            self.box_s_step_losses[i] += loss[:, i, 2:].sum().item()

        self.losses['p_1'] = float(np.mean(self.box_p_step_losses[:self.ptrain_size]))
        self.losses['p_2'] = float(np.mean(self.box_p_step_losses[self.ptrain_size:])) \
            if self.ptrain_size < self.ptest_size else 0
        self.losses['s_1'] = float(np.mean(self.box_s_step_losses[:self.ptrain_size]))
        self.losses['s_2'] = float(np.mean(self.box_s_step_losses[self.ptrain_size:])) \
            if self.ptrain_size < self.ptest_size else 0

        mask_loss = 0
        if C.RPIN.MASK_LOSS_WEIGHT > 0:
            # of shape (batch, time, #obj, m_sz, m_sz)
            mask_loss_ = F.binary_cross_entropy(outputs['masks'], labels['masks'], reduction='none')
            mask_loss = mask_loss_.mean((3, 4))
            valid = labels['valid'][:, None, :]
            mask_loss = mask_loss * valid
            mask_loss = mask_loss.sum(2) / valid.sum(2)

            for i in range(pred_size):
                self.masks_step_losses[i] += mask_loss[:, i].sum().item()

            m1_loss = self.masks_step_losses[:self.ptrain_size]
            m2_loss = self.masks_step_losses[self.ptrain_size:]
            self.losses['m_1'] = np.mean(m1_loss)
            self.losses['m_2'] = np.mean(m2_loss) if self.ptrain_size < self.ptest_size else 0

            mask_loss = mask_loss.mean(0)
            init_tau = C.RPIN.DISCOUNT_TAU ** (1 / self.ptrain_size)
            tau = init_tau + (self.iterations / self.max_iters) * (1 - init_tau)
            tau = torch.pow(tau, torch.arange(pred_size, out=torch.FloatTensor()))[:, None].to('cuda')
            mask_loss = ((mask_loss * tau) / tau.sum(axis=0, keepdims=True)).sum()
            mask_loss = mask_loss * C.RPIN.MASK_LOSS_WEIGHT

        seq_loss = 0
        if C.RPIN.SEQ_CLS_LOSS_WEIGHT > 0:
            seq_loss = F.binary_cross_entropy(outputs['score'], labels['seq_l'], reduction='none')
            self.losses['seq'] += seq_loss.sum().item()
            seq_loss = seq_loss.mean() * C.RPIN.SEQ_CLS_LOSS_WEIGHT
            # calculate accuracy
            s = (outputs['score'] >= 0.5).eq(labels['seq_l'])
            fg_correct = s[labels['seq_l'] == 1].sum().item()
            bg_correct = s[labels['seq_l'] == 0].sum().item()
            fg_num = (labels['seq_l'] == 1).sum().item()
            bg_num = (labels['seq_l'] == 0).sum().item()
            self.fg_correct += fg_correct
            self.bg_correct += bg_correct
            self.fg_num += fg_num
            self.bg_num += bg_num

        kl_loss = 0
        if C.RPIN.VAE and phase == 'train':
            kl_loss = outputs['kl']
            self.losses['kl'] += kl_loss.sum().item()
            kl_loss = C.RPIN.VAE_KL_LOSS_WEIGHT * kl_loss.sum()

        # no need to do precise batch statistics, just do mean for backward gradient
        loss = loss.mean(0)
        init_tau = C.RPIN.DISCOUNT_TAU ** (1 / self.ptrain_size)
        tau = init_tau + (self.iterations / self.max_iters) * (1 - init_tau)
        tau = torch.pow(tau, torch.arange(pred_size, out=torch.FloatTensor()))[:, None].to('cuda')
        loss = ((loss * tau) / tau.sum(axis=0, keepdims=True)).sum()
        loss = loss + mask_loss + kl_loss + seq_loss

        return loss

    def _setup_loss(self):
        self.loss_name = []
        self.position_loss_weight = C.RPIN.POSITION_LOSS_WEIGHT
        self.loss_name += ['p_1', 'p_2', 's_1', 's_2']
        if C.RPIN.MASK_LOSS_WEIGHT:
            self.loss_name += ['m_1', 'm_2']
        if C.RPIN.VAE:
            self.loss_name += ['kl']
        if C.RPIN.SEQ_CLS_LOSS_WEIGHT:
            self.loss_name += ['seq']
        self._init_loss()

    def _init_loss(self):
        self.losses = dict.fromkeys(self.loss_name, 0.0)
        self.box_p_step_losses = [0.0 for _ in range(self.ptest_size)]
        self.box_s_step_losses = [0.0 for _ in range(self.ptest_size)]
        self.masks_step_losses = [0.0 for _ in range(self.ptest_size)]
        # an statistics of each validation
        self.fg_correct, self.bg_correct, self.fg_num, self.bg_num = 0, 0, 0, 0
        self.loss_cnt = 0
        self.time = timer()


class LRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Schedule the learning rate according to the learning rate schedule policy
    Support `cosine` and `linear` only
    Note that the scheduler schedule learning rate by steps
    """
    def __init__(
        self,
        optimizer,
        steps,
        lr_final,
        min_lr=1e-6,
        lr_policy='cosine',
        warmup_steps=10_000,
        accum_steps=1,
        last_epoch=-1,
    ):
        if lr_policy not in ('cosine', 'linear'):
            raise ValueError(
                f'Only "cosine" or "linear" schedule policy are supported, but got {lr_policy}')
        self.steps = math.ceil(steps / accum_steps)
        self.lr_final = lr_final
        self.min_lr = min_lr
        self.lr_policy = lr_policy
        self.warmup_steps = math.ceil(warmup_steps / accum_steps)
        super(LRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        progress = (self._step_count - self.warmup_steps) / float(self.steps - self.warmup_steps)
        progress = np.clip(progress, 0.0, 1.0)
        ratio = getattr(self, self.lr_policy)(progress)
        if self.warmup_steps:
            ratio = ratio * np.minimum(1., self._step_count / self.warmup_steps)
        return [max(self.min_lr, lr * ratio) for lr in self.base_lrs]

    def linear(self, progress):
        return self.lr_final + (1 - self.lr_final) * (1.0 - progress)

    def cosine(self, progress):
        return 0.5 * (1. + np.cos(np.pi * progress))

    def __repr__(self):
        return f'{self.lr_policy.capitalize()}LRScheduler'