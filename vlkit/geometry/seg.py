import cv2
import numpy as np
from skimage import morphology


def seg2edge(seg, thin=False):
    assert isinstance(seg, np.ndarray)
    assert seg.ndim == 2 or seg.ndim == 3, seg.shape

    def s2e(s, thin=False):
        g = np.gradient(s.astype(np.float32))
        g = np.sqrt(g[0]**2 + g[1]**2)
        e = g != 0
        if thin:
            e = morphology.thin(e)
        return e

    if seg.ndim == 2:
        edge = s2e(seg, thin=thin)
    else:
        edge = np.zeros_like(seg, dtype=bool)
        for i in range(edge.shape[0]):
            edge[i, ] = s2e(seg[i, ], thin=thin)

    return edge


def sobel(x, kernel_size=3):
    sobelx = cv2.Sobel(x, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobely = cv2.Sobel(x, cv2.CV_64F, 0, 1, ksize=kernel_size)

    sobel = np.sqrt(sobelx**2 + sobely**2)
    return sobel, sobely, sobelx


def flux2angle(flux):
    """
    computing orientation angle for each pixel given a flux
    input:
      flux: a [2, H, W] tensor represents the flux vector of each position
    return:
      angle: a [H, W] matrix representing angle of each location
    """
    C, H, W = flux.shape
    assert C == 2
    top_half = flux[0,...] >= 0 # y >= 0, \theta <= \pi
    bottom_half = flux[0,...] < 0 # y < 0, \theta > \pi

    unit = np.zeros((2, H, W), dtype=np.float32)
    unit[1,...] = 1 # unit vector: (y=0, x=1)
    cos = (flux * unit).sum(axis=0)
    acos = np.arccos(cos)
    angle = np.zeros((H, W), dtype=np.float32)
    angle[top_half] = acos[top_half]
    angle[bottom_half] = 2*np.pi - acos[bottom_half]

    return angle


def angle2flux(angle):
    """
    The inverse of `flux2angle`
    """
    return np.dstack((np.sin(angle), np.cos(angle))).transpose((2,0,1))


def dense2flux(density, kernel_size=9, normalize=True):
    """
    Compute flux field of a density map
    input:
        density: density map with shape [H, W]
    output:
        flux: flux field of shape [2, H, W]
      
    """
    sobelx = cv2.Sobel(density, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobely = cv2.Sobel(density, cv2.CV_64F, 0, 1, ksize=kernel_size)

    flux = np.dstack((sobely, sobelx)).transpose((2,0,1))

    if normalize:
        norm = np.expand_dims(np.sqrt((flux**2).sum(axis=0)), axis=0)
        norm[norm==0] = 1
        flux /= norm

    return flux


def quantize_angle(angle, num_bins=8):
    """
    angle: angle map with shape [H, W]
    num_bins: number of quantization bins
    """
    # clamp angle
    angle[angle>=np.pi*2] = np.pi*2 - 1e-5
    q = np.round(angle / (np.pi*2/num_bins)).astype(np.uint8)
    q[q==num_bins] = 0
    return q


def dequantize_angle(q, num_bins=8):
    """
    q: quantized angles (0~num_bins-1)
    num_bins: number of quantized levels
    """
    assert q.min() >= 0 and q.max() < num_bins
    angle = q * (np.pi*2 / num_bins)
    return angle
