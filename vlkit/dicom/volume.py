import numpy as np
from .transform_matrix import (
    get_pixel_to_patient_transformation_matrix,
    apply_transformation_to_3d_points)


class Volume(object):
    __coords: np.ndarray
    __values: np.ndarray
    def __init__(self, dicoms):
        mat = get_pixel_to_patient_transformation_matrix(dicoms)
        self.__values = np.concatenate(([i.pixel_array[None, :, :] for i in dicoms]), axis=0)
        h, w = dicoms[0].pixel_array.shape
        d = len(dicoms)
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x, y = x.reshape(-1, 1), y.reshape(-1, 1)
        points = np.concatenate((x, y), axis=1)
        self.__coords = np.zeros((d, h, w, 3))
        for i in range(d):
            p = np.concatenate((points, np.full((len(points), 1), i)), axis=1)
            v = apply_transformation_to_3d_points(p, mat)
            v = v.reshape(h, w, v.shape[-1])
            self.__coords[i, :, :, :] = v
    @property
    def coords(self):
        return self.__coords
    @property
    def values(self):
        return self.__values
    
    def __repr__(self):
        return f"Volume of shape {self.coords.shape[:-1]}"