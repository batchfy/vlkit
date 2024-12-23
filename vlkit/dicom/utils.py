import pydicom
from ..common import dotdict
from warnings import warn
from typing import List, Dict
from glob import glob

def load_dicom_files(dicom_dir: str, **kwargs) -> List[pydicom.dataset.FileDataset]:
    """
    Loads DICOM files from a directory.

    Parameters:
    dicom_dir (str): The directory containing DICOM files.
    recursive (bool): If True, will search for DICOM files recursively.

    Returns:
    list: A list of DICOM objects.
    """
    dicom_files = glob(f"{dicom_dir}/**/*.dcm", recursive=True)
    return [pydicom.dcmread(dcm, **kwargs) for dcm in dicom_files]


def group_dicoms_into_studies(dicoms: List[pydicom.dataset.FileDataset]) -> Dict[str, List[pydicom.dataset.FileDataset]]:
    """
    Groups a list of DICOM objects into studies based on their StudyInstanceUID.

    Parameters:
    dicoms (list): A list of DICOM objects. Each DICOM object should have a 'StudyInstanceUID' attribute.

    Returns:
    dict: A dictionary where the keys are StudyInstanceUIDs and the values are lists of DICOM objects that belong to the corresponding study.

    Notes:
    - If a DICOM object does not have a 'StudyInstanceUID' attribute, a warning will be issued and the object will be skipped.
    """
    studies = dict()
    for dcm in dicoms:
        if not hasattr(dcm, "StudyInstanceUID"):
            warn(f"{dcm.fullpath} does not have 'StudyInstanceUID' attribute")
            continue
        if dcm.StudyInstanceUID in studies:
            studies[dcm.StudyInstanceUID].append(dcm)
        else:
            studies[dcm.StudyInstanceUID] = [dcm]
    return studies


def group_dicoms_into_series(
    dicoms: List[pydicom.dataset.FileDataset],
    remove_duplicates: bool=False) -> dict:
    """
    Groups DICOM files into series based on their SeriesInstanceUID.

    Args:
        dicoms (list): A list of DICOM files, where each file is represented by a dataset object.

    Returns:
        dict: A dictionary where the keys are SeriesInstanceUIDs and the values are lists of DICOM files belonging to that series.
    """
    series = dotdict()
    for ds in dicoms:
        if ds.SeriesInstanceUID not in series:
            series[ds.SeriesInstanceUID] = [ds]
        else:
            series[ds.SeriesInstanceUID].append(ds)
    if remove_duplicates:
        for k in series.keys():
            sop_instance_uids = set()
            new_series = []
            for s in series[k]:
                if s.SOPInstanceUID not in sop_instance_uids:
                    sop_instance_uids.add(s.SOPInstanceUID)
                    new_series.append(s)
            series[k] = new_series
    return series


def build_sop_instance_uid_lookup_table(dicoms: List[pydicom.dataset.FileDataset]) -> Dict[str, pydicom.dataset.FileDataset]:
    """
    Builds a lookup table for SOPInstanceUIDs.

    Parameters:
    dicoms (list): A list of DICOM objects. Each DICOM object should have a 'SOPInstanceUID' attribute.

    Returns:
    dict: A dictionary where the keys are SOPInstanceUIDs and the values are the corresponding DICOM objects.
    """
    sop_instance_uid_lookup_table = dotdict()
    for ds in dicoms:
        sop_instance_uid_lookup_table[ds.SOPInstanceUID] = ds
    return sop_instance_uid_lookup_table