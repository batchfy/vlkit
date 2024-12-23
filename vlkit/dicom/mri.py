import numpy as np

def kspace_resample(im, scale):
    assert im.ndim == 2, im.shape
    h, w = im.shape[0:2]
    h_new, w_new = round(scale * h), round(scale * w)
    f = np.fft.fftshift(np.fft.fft2(im))
    dh, dw = ((h_new - h) // 2, (w_new - w) // 2)
    info = np.iinfo(im.dtype)
    if dh >= 0 and dw >= 0:
        f_resized = np.zeros((h_new, w_new), dtype=np.complex64)
        f_resized[dh:dh+h, dw:dw+w] = f
        im1 = np.fft.ifft2(np.fft.ifftshift(f_resized))
        im1 = np.real(im1).clip(info.min, info.max)
    else:
        f_resized = np.zeros((h, w), dtype=np.complex64)
        f_resized[(h//2 - h_new//2):(h//2 + h_new//2), 
                  (w//2 - w_new//2):(w//2 + w_new//2)] = f[(h//2 - h_new//2):(h//2 + h_new//2), 
                                                           (w//2 - w_new//2):(w//2 + w_new//2)]
        im1 = np.fft.ifft2(np.fft.ifftshift(f_resized))
        im1 = np.real(im1).clip(info.min,info.max) 
    im1 = im1.astype(im.dtype)
    return im1
