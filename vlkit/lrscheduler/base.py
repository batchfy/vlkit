import numpy as np


class BaseScheduler(object):
    def __init__(self,
            warmup_iters: int=0,
            warmup_init_lr: float=0,
            noice_std: float=.0,
            last_iter: int=-1):

        self.warmup_iters = warmup_iters
        self.warmup_init_lr = warmup_init_lr
        self.noice_std = noice_std
        self.last_iter = last_iter
        assert self.last_iter >= -1

        self.iter = last_iter + 1
        assert self.iter >= 0

    def step(self, n=1):
        lr = self.get_lr(self.iter)
        if self.noice_std > 0:
            lr = max(lr + np.random.normal(scale=self.noice_std * lr), 0)
        self.iter += n
        return lr

    def get_lr(self, iter):
        raise NotImplementedError
