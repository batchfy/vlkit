"""`Non-Photorealistic Rendering <https://docs.opencv.org/4.5.2/df/dac/group__photo__render.html>`_

"""
import cv2
import torch, random
from PIL import Image
import numpy as np


type2func = {'pencilsketch': cv2.pencilSketch,
             'stylization': cv2.stylization,
             'detailEnhance': cv2.detailEnhance,
             'edgePreservingFilter': cv2.edgePreservingFilter}


class NPR(torch.nn.Module):
    """`Non-Photorealistic Rendering <https://docs.opencv.org/4.5.2/df/dac/group__photo__render.html>`_

    Args:
        transform (str): type of transformation, should be one of `pencilsketch`, `stylization`, `detailEnhance` or `edgePreservingFilter`.
        sigma_s (int or list of ints): see <https://docs.opencv.org/4.5.2/df/dac/group__photo__render.html>.
        sigma_r (float or list of float): see <https://docs.opencv.org/4.5.2/df/dac/group__photo__render.html>.
    """
    def __init__(self, transform='stylization', sigma_s=60, sigma_r=0.001):
        super().__init__()
        assert transform in type2func, 'transform must be one of {0} but' \
            '\'{1}\' was given'.format(list(type2func.keys()), transform)
        self.transform = transform
        self.f = type2func[transform]
        assert isinstance(sigma_s, (int, list))
        assert isinstance(sigma_r, (float, list))

        if isinstance(sigma_s, list):
            assert len(sigma_s) == 2 and min(sigma_s) > 0
        else:
            assert sigma_s > 0
        if isinstance(sigma_r, list):
            assert len(sigma_r) == 2 and min(sigma_r) > 0
        else:
            assert sigma_r > 0
        self.sigma_s = sigma_s
        self.sigma_r = sigma_r


    def forward(self, x:Image.Image) -> Image.Image:
        if isinstance(self.sigma_s, list):
            sigma_s = random.uniform(*self.sigma_s)
        else:
            sigma_s = self.sigma_s
        if isinstance(self.sigma_r, list):
            sigma_r = random.uniform(*self.sigma_r)
        else:
            sigma_r = self.sigma_r

        x = np.array(x)
        x = self.f(x, sigma_s=sigma_s, sigma_r=sigma_r)
        if self.transform == 'pencilsketch':
            x = x[1]
        return Image.fromarray(x)

    def __repr__(self):
        return self.__class__.__name__ + '(transform={0}, sigma_s={1}, sigma_r={2})'.format(
            self.transform, self.sigma_s, self.sigma_r)
