import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.folder import pil_loader
import torchvision.datasets as datasets
import numpy as np
from PIL import Image
import vlkit.image as vlimage
import os, sys
from os.path import join, split, splitext, abspath, dirname, isfile, isdir
from collections import defaultdict

def ilsvrc2012(path, bs=256, num_workers=8, crop_size=224):
    traindir = os.path.join(path, 'train')
    valdir = os.path.join(path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=bs, shuffle=True,
        num_workers=num_workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=bs, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

def cifar10(path='data/cifar10', bs=100, num_workers=8):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = datasets.CIFAR10(root=path, train=True, download=True,
                                                 transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True,
                                               num_workers=num_workers)

    test_dataset = datasets.CIFAR10(root=path, train=False, download=True,
                                                transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False,
                                               num_workers=num_workers)

    return train_loader, test_loader

def cifar100(path='data/cifar100', bs=256, num_workers=8):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = datasets.CIFAR100(root=path, train=True, download=True,
                                                 transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True,
                                               num_workers=num_workers)

    test_dataset = datasets.CIFAR100(root=path, train=False, download=True,
                                                transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, 
                                               num_workers=num_workers)

    return train_loader, test_loader

def svhn(path='data/svhn', bs=100, num_workers=8):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = datasets.SVHN(root=path, split="train", download=True,
                                                 transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True,
                                               num_workers=num_workers)

    test_dataset = datasets.SVHN(root=path, split="test", download=True,
                                                transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False,
                                               num_workers=num_workers)

    return train_loader, test_loader

class CACDDataset(torch.utils.data.Dataset):

    def __init__(self, root, filelist):

        self.root = root
        assert isfile(filelist)

        # list files: `cacd_train_list.txt`, `cacd_test_list.txt`, `cacd_val_list.txt`
        with open(filelist) as f:
            self.items = f.readlines()

    def __getitem__(self, index):

        filename, age = self.items[index]
        age = int(age)
        im = pil_loader(join(self.root, filename))

        return im, age

    def __len__(self):
        return len(self.items)

class FileListDataset(torch.utils.data.Dataset):

    def __init__(self, filelist, root=None, transform=None, target2indice=False):

        assert isfile(filelist)
        self.root = root
        self.transform = transform

        self.items = [i.strip().split(" ") for i in open(filelist).readlines()]
        if root is not None:
            self.items = [[join(root, i[0]), i[1:]] for i in self.items]

        self.num_classes = np.unique(np.array([int(i[1]) for i in self.items])).size if len(self.items[0]) > 1 else -1

        if target2indice:
            assert self.num_classes > 0
            self.target2indice = defaultdict(list)
            for idx, i in enumerate(self.items):
                target = int(i[1])
                self.target2indice[target].append(idx)
        else:
            self.target2indice = None

    def __getitem__(self, index):

        if len(self.items[index]) >= 2:
            fpath, target = self.items[index]
            target = int(target)
        else:
            fpath, = self.items[index]
            target = -1

        assert isfile(fpath), fpath
        im = pil_loader(fpath)

        if self.transform is not None:
            im = self.transform(im)

        return {"image": im, "target": target, "path": fpath}

    def __len__(self):
        return len(self.items)
