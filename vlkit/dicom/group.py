from glob import glob
from ..common import Dotdict

try:
    import pydicom
    from pydicom.dataset import Dataset
    from pydicom import dcmread
except ImportError:
    pydicom = None


def group_study(dicoms: list):
    """
    group dicom files by study instance UID
    """
    if pydicom is None:
        raise ImportError("This function requires pydicom to be installed. Please install it using 'pip install pydicom'")
    assert isinstance(dicoms, list), "dicoms must be a list"
    assert all([hasattr(ds, "StudyInstanceUID") for ds in dicoms]), "All items in dicoms must have 'StudyInstanceUID' attribute"
    study_dict = Dotdict()
    if len(dicoms) == 0:
        return study_dict

    for ds in dicoms:
        study_uid = ds.StudyInstanceUID
        if study_uid not in study_dict:
            study_dict[study_uid] = []
        study_dict[study_uid].append(ds)

    return study_dict


def group_series(dicoms: list, remove_duplicates: bool = True):
    """
    group dicom files by series instance UID
    """
    if pydicom is None:
        raise ImportError("This function requires pydicom to be installed. Please install it using 'pip install pydicom'")
    assert isinstance(dicoms, list), "dicoms must be a list"
    assert all([hasattr(ds, "SeriesInstanceUID") for ds in dicoms]), "All items in dicoms must have 'SeriesInstanceUID' attribute"

    series_dict = Dotdict()
    if len(dicoms) == 0:
        return series_dict

    for ds in dicoms:
        series_uid = ds.SeriesInstanceUID
        if series_uid not in series_dict:
            series_dict[series_uid] = []
        series_dict[series_uid].append(ds)
    
    if remove_duplicates:
        for k in series_dict.keys():
            # Remove duplicate dicom files based on SOP Instance UID
            unique_ds = {}
            for ds in series_dict[k]:
                sop_uid = ds.SOPInstanceUID
                if sop_uid not in unique_ds:
                    unique_ds[sop_uid] = ds
            series_dict[k] = list(unique_ds.values())
    return series_dict
