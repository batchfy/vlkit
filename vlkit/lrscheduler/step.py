import numpy as np
import math
from typing import Union, List
from .base import BaseScheduler



class MultiStepScheduler(BaseScheduler):
    def __init__(self,
            milestones: List[int],
            gammas,
            base_lr: float=0.1,
            warmup_iters: int=0,
            warmup_init_lr: float=0.0,
            noice_std: float=0.0,
            last_iter: int=-1,
            **kwargs):
        

        super().__init__(
                warmup_iters,
                warmup_init_lr,
                noice_std,
                last_iter)

        if isinstance(gammas, float):
            gammas = [gammas,] * len(milestones)

        self.base_lr = base_lr
        self.milestones = milestones
        self.gammas = gammas

        self.milestone_counter = 0

    def get_lr(self, iter):
        if self.warmup_iters > 0 and iter <= self.warmup_iters:
            lr = self.warmup_init_lr + (iter / self.warmup_iters) * (self.base_lr - self.warmup_init_lr)
        else:
            stage = np.digitize(iter, self.milestones)
            if stage == 0:
                lr = self.base_lr
            else:
                lr = self.base_lr * np.prod(self.gammas[:stage])
        return lr
