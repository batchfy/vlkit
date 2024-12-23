import time, hashlib, os, re, logging
from os.path import join, isdir
from ..array import torch, np


def str2bool(input: str):
    if input.lower in ['yes', 'good', 'right', 'y']:
        return True
    elif input.lower() in ['no', 'bad', 'wrong', 'n']:
        return False
    else:
        raise RuntimeError(f"Cannot convert {input} to boolean.")


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    copy <from https://github.com/pytorch/examples/blob/master/imagenet/main.py>
    """
    if torch == None:
        raise ModuleNotFoundError("torch not found.")
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.div(batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name='value', fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def get_logger(path="log.txt"):
    logger = logging.getLogger("Logger")
    file_handler = logging.FileHandler(path, "w")
    stdout_handler = logging.StreamHandler()
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    logformatter = logging.Formatter('[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s','%m-%d %H:%M:%S')
    stdout_handler.setFormatter(logformatter)
    file_handler.setFormatter(logformatter)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


class Logger():
    def __init__(self, path="log.txt"):
        self.logger = logging.getLogger("Logger")
        self.file_handler = logging.FileHandler(path, "w")
        self.stdout_handler = logging.StreamHandler()
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.stdout_handler)
        logformatter = logging.Formatter('[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s','%m-%d %H:%M:%S')
        self.stdout_handler.setFormatter(logformatter)
        self.file_handler.setFormatter(logformatter)
        self.logger.setLevel(logging.INFO)
    
    def info(self, txt):
        self.logger.info(txt)
    
    def close(self):
        self.file_handler.close()
        self.stdout_handler.close()


def get_workdir(prefix="work_dirs/run"):
    """
    this function returns a working directory given a prefix
    """
    if prefix.endswith(os.sep):
        prefix = prefix[:-1]
    sec = time.time()
    md5 = hashlib.md5(str(sec).encode()).hexdigest()
    timestr = time.strftime('%Y%m%d_%H%M%S', time.localtime(sec))
    workdir = "{prefix}_{timestr}_{md5:.6}".format(prefix=prefix, timestr=timestr, md5=md5)
    return workdir


def isarray(x):
    if torch is not None:
        return isinstance(x, (np.ndarray, torch.Tensor))
    else:
        return isinstance(x, (np.ndarray))
