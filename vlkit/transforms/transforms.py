import torch
import  torchvision
import numpy as np


class CoordCrop(torch.nn.Module):

    def __init__(self, x1, y1, x2, y2):
        super().__init__()
        assert x2 > x1 and y2 > y1
        
        self.x1, self.y1 = x1, y1
        self.h = y2-y1
        self.w = x2 - x1


    def forward(self, img):

        return torchvision.transforms.functional.crop(img, self.y1, self.x1, self.h, self.w)

    def __repr__(self):
        return self.__class__.__name__
