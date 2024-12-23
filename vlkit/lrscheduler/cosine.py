import numpy as np
import math
from .base import BaseScheduler


class CosineScheduler(BaseScheduler):
    def __init__(self,
            max_iters,
            warmup_iters: int=0,
            warmup_init_lr: float=0,
            max_lr: float=0.1,
            min_lr: float=0,
            restarts: int=0,
            decay_factor: float=0.1,
            noice_std: float=0,
            last_iter: int=-1, **kwargs):
        warmup_init_lr = warmup_init_lr if warmup_init_lr > 0 else min_lr
        super().__init__(
                warmup_iters,
                warmup_init_lr,
                noice_std,
                last_iter)

        assert restarts >= 0

        self.max_iters = max_iters
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.restarts = restarts
        self.decay_factor = decay_factor
        self.cycle = math.ceil((self.max_iters - self.warmup_iters) / (self.restarts + 1))

    def get_lr(self, iter):
        if iter < self.warmup_iters:
            lr = self.warmup_init_lr + (iter / self.warmup_iters) * (self.max_lr - self.warmup_init_lr)
        else:
            round = (iter - self.warmup_iters) // self.cycle
            base_lr = self.max_lr * (self.decay_factor ** round)
            assert base_lr > self.min_lr, f"{self.max_lr} | {(self.decay_factor ** round)} . {base_lr} v.s. {self.min_lr}"
            step = (iter - self.warmup_iters) % self.cycle
            lr = (base_lr - self.min_lr) * (1 + math.cos((step / self.cycle) * math.pi)) / 2 + self.min_lr

        return lr
