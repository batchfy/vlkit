from PIL import Image
import cv2
from . import __all_backends__
import random


pil_interpolations = {"bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC, "lanczos": Image.LANCZOS}
cv2_interpolations = {"inter_linear": cv2.INTER_LINEAR, "inter_cubic": cv2.INTER_CUBIC,
                      "inter_area": cv2.INTER_AREA, "inter_lanczos": cv2.INTER_LANCZOS4}
__all_interpolations__ = list(pil_interpolations) + list(cv2_interpolations)


def get_interp(interpolation="bilinear", backend="pil"):

    assert backend.lower() in __all_backends__,\
            'backend (\'{backend}\') should be one of {all_backends}'.format(
                    backend=backend, all_backends=__all_backends__)

    if backend == "cv2":
        assert interpolation.lower() in cv2_interpolations,\
               'given backend=\'{backend}\', interpolation should be one of {all_interp}, '\
               'but \'{interpolation}\' was given.'.format(
                       backend=backend,
                       all_interp=cv2_interpolations,
                       interpolation=interpolation)
        return cv2_interpolations[interpolation]
    else:
        assert interpolation.lower() in pil_interpolations,\
               'given backend=\'{backend}\', interpolation should be one of {all_interp}, '\
               'but \'{interpolation}\' was given.'.format(
                       backend=backend,
                       all_interp=pil_interpolations,
                       interpolation=interpolation)
        return pil_interpolations[interpolation]

def get_random_interp(backend="pil"):
    if backend == "pil":
        interp = pil_interpolations
    elif backend == "cv2":
        interp = cv2_interpolations
    return random.choice(list(interp.values()))
