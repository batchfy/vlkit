import matplotlib
import os, sys
from os.path import dirname, abspath, join
TEST_DIR = dirname(__file__)
sys.path.insert(0, abspath(join(TEST_DIR, "../")))
from vlkit.lrscheduler import CosineScheduler, MultiStepScheduler
matplotlib.use("Agg")
import matplotlib.pyplot as plt

savedir = join(dirname(__file__), "../data/test")
os.makedirs(savedir, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(12, 3))
for ax in axes:
    ax.grid(alpha=0.5, linestyle='dotted', linewidth=2, color='black')

epochs = 20
loader_len = 100
warmup_epochs = 1

# multi step lr
lr_scheduler = MultiStepScheduler(
    iters=loader_len*epochs,
    milestones=[1000, 1500],
    base_lr=0.1,
    gammas=[0.5, 0.5],
    warmup_iters=100)

lr_record = []
for i in range(epochs*loader_len):
    lr_record.append(lr_scheduler.step())
axes[0].plot(lr_record, color="blue", linewidth=1)
axes[0].set_xlabel("Iter", fontsize=12)
axes[0].set_ylabel("Learning Rate", fontsize=12)
axes[0].set_title("Warmup + Multistep LR")

lr_scheduler = CosineScheduler(
        max_iters=2000,
        max_lr=0.1,
        min_lr=0.01,
        warmup_iters=200)

lr_record = []
for i in range(epochs * loader_len):
    lr_record.append(lr_scheduler.step())

axes[1].plot(lr_record, color="blue", linewidth=1)
axes[1].set_xlabel("Iter", fontsize=12)
axes[1].set_title("Cosine Annealing + Warmup")

lr_scheduler = CosineScheduler(
        max_iters=2000,
        restarts=2,
        decay_factor=0.5,
        max_lr=0.1,
        min_lr=0.001,
        warmup_iters=200,
        noice_std=0.02)

lr_record = []
for i in range(epochs * loader_len):
    lr_record.append(lr_scheduler.step())

axes[2].plot(lr_record, color="blue", linewidth=1)
axes[2].set_xlabel("Iter", fontsize=12)
axes[2].set_title("Cosine Annealing + Restarts + Noice")

plt.tight_layout()
plt.savefig(join(savedir, "lr_scheduler.svg"))
