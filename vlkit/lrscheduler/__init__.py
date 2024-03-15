"""Learning rate schedulers

Example:
    ::

        # multistep lr scheduler with warmup
        lr_scheduler = MultiStepScheduler(epoch_size=1000, epochs=20, milestones=[4, 8], base_lr=0.1,
            gamma=[0.1, 0.1], warmup_epochs=warmup_epochs, warmup_init_lr=0.05)

        # cosine scheduler with warmup restarts and noice
        lr_scheduler = CosineScheduler(epoch_size=1000, epochs=20, restarts=2,
                           restart_decay=0.8, max_lr=0.1, min_lr=0.01,
                           warmup_epochs=warmup_epochs, warmup_init_lr=0.05, noice_std=0.02)

.. image:: _static/lr_scheduler.svg
"""

from .cosine import CosineScheduler
from .step import MultiStepScheduler
