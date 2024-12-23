import torch
import numpy as np
from PIL import Image


class RandomChoice(torch.nn.Module):
    """Random choose transforms

    Args:
        transforms (list): list of transforms to be selected
        n (int): number of transforms will be selected in each step
        p (list): probabilities if each transforms being selected
    """
    def __init__(self, transforms, p=0.5, n=1, p_choice=None):
        super().__init__()
        assert isinstance(transforms, (list, type(None)))
        self.transforms = transforms
        self.p = p
        self.n = n
        self.p_choice = p_choice
        if isinstance(self.p_choice, list):
            assert len(self.p_choice) == len(self.transforms)
            self.p_choice = np.array(self.p_choice)
            assert self.p_choice.min() >= 0
            self.p_choice /= self.p_choice.sum()

    def forward(self, x:Image.Image) -> Image.Image:
        if np.random.uniform() < self.p:
            transforms = np.random.choice(self.transforms, size=self.n, p=self.p_choice)
            for t in transforms:
                x = t(x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(transforms={0}, p={1}, n={2}, p_choice={3})'.format(
            self.transforms, self.p, self.n, self.p_choice)