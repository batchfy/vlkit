import torch
import os.path as osp
from torchvision.datasets.folder import pil_loader
import warnings


class FileListDataset(torch.utils.data.Dataset):
    def __init__(self, filelist, root_dir=None, transform=None):
        assert osp.isfile(filelist)
        self.root_dir = root_dir
        self.transform = transform

        with open(filelist, 'r') as f:
            self.items = [i.strip().split(" ") for i in f.readlines()]

        if self.root_dir is not None:
            self.items = tuple((osp.join(self.root_dir, i[0]), i[1]) for i in self.items)
        else:
            self.items = tuple(tuple(i) for i in self.items)

    def __getitem__(self, index):
        if len(self.items[index]) >= 2:
            path, label = self.items[index]
            label = int(label)
        else:
            path, = self.items[index]
            label = -1
            warnings.warn('Use default label = -1, check your list !')

        assert osp.isfile(path), path
        img = pil_loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return dict(
                img=img,
                label=label,
                path=path
            )

    def __len__(self):
        return len(self.items)

    @property
    def num_samples(self):
        return self.__len__()

    @property
    def num_classes(self):
        return len(set((i[1] for i in self.items)))
