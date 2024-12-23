import os, datetime
import os.path as osp
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import UID


def write_array_to_dicom(array, fn, sample_ds=None, **kwargs):
    assert array.ndim == 2 and array.dtype == np.uint16
    assert fn.endswith('dcm') or fn.endswith('dicom'), fn

    if sample_ds is None:
        # Populate required values for file meta information
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = UID('1.2.840.10008.5.1.4.1.1.2')
        file_meta.MediaStorageSOPInstanceUID = UID("1.2.3")
        file_meta.ImplementationClassUID = UID("1.2.3.4")

        ds = FileDataset(fn, {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.PatientName = "Test^Firstname"
        ds.PatientID = "123456"

        # Set creation date/time
        dt = datetime.datetime.now()
        ds.ContentDate = dt.strftime('%Y%m%d')
        # long format with micro seconds
        timeStr = dt.strftime('%H%M%S.%f')
        ds.ContentTime = timeStr
    else:
        ds = sample_ds
    
    ds.PixelData = array.tobytes()

    for k, v in kwargs.items():
        setattr(ds, k, v)

    ds.save_as(fn)


def read_dicoms(input, sort_by=None):
    """
    read dicom(s) from a directory or a list of dicom files

    input: dicom file path, a directory, list of dicom files
    """
    if isinstance(input, str):
        assert osp.isdir(input) or osp.isfile(input), input
    elif isinstance(input, list):
        for i in input:
            assert osp.isfile(i), i
    else:
        raise ValueError(input)
    if isinstance(input, str):
        if osp.isdir(input):
            dicoms = sorted([osp.join(input, i) for i in os.listdir(input) if i.endswith('dcm') or i.endswith('dicom')])
        elif osp.isfile(input):
            dicoms = [input]
        else:
            raise ValueError(input)
    else:
        dicoms = input
    dicoms = [pydicom.dcmread(d) for d in dicoms]
    if sort_by is not None:
        dicoms = sorted(dicoms, key=lambda x: getattr(x, sort_by))
    return dicoms


def read_dicom_data(input, cat=False):
    """
    read dicom pixel array from a directory or a list of dicom files

    input: dicom file path, a directory, list of dicom files
    """
    if cat:
        results = tuple(ds.pixel_array[None,] for ds in read_dicoms(input))
        results = np.concatenate(results, axis=0)
    else:
        results = tuple(ds.pixel_array for ds in read_dicoms(input))
    return results
