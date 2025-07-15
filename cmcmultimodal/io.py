import os
import numpy as np
from pathlib import Path
from fsl.data.image import Image
from scipy.io import loadmat

def mat2nii(mat_file, nii_file=None, nii_lowres_file=None, downsample=0):
    '''Convert and store a PSOCT .mat file to a NIFTI image.
    Optionally also create a low-resolution version.
    '''
    # Load matfile and identify data
    # TODO add checks if file exists and can be read
    mat_contents = loadmat(mat_file)
    select       = [not (x[:2]+x[-2:])=='____' for x in list(mat_contents.keys())]
    key          = np.array(list(mat_contents.keys()))[select][0]
    # Create array
    X = np.array(mat_contents[key])
    X = np.fliplr(X).T
    # Deal with orientation
    if X.dtype == np.complex128:
        X = np.angle(X)

    # reduce size
    X = np.asarray(X, dtype=np.float32)

    # deal with nans
    X = np.nan_to_num(X)
    # save high-res NIFTI
    if nii_file is None:
        nii_file = mat_file.replace('.mat', '.nii.gz')
    save_nifti(X, nii_file)

    # Also output low res
    if downsample > 0 and nii_lowres_file is not None:
        save_nifti(X[::downsample, ::downsample], nii_lowres_file)

    return Path(nii_file), Path(nii_lowres_file)

def save_nifti(data, filename):
    out_fd = Path(filename).parent
    os.makedirs(out_fd, exist_ok=True)
    Image(data).save(filename)


def zeropad(filename, length=3, save=False):
    '''Zero pad the slice number in a filename of format: PREFIX_<slice>_SUFFIX.ext
    
    Parameters:
    - filename: str or Path, the file path to modify.
    - length: int, number of digits to pad to.
    - save: bool, if True, renames the file on disk.

    Returns:
    - The new filename as a string (whether renamed or not).
    '''
    filename = Path(filename)
    fileparts = filename.name.split('_')
    # TODO consider expanding it to more parts if only one is numeric
    if len(fileparts) != 3:
        raise ValueError(f"Unexpected filename format: {filename}. Expected format: PREFIX_<slice>_SUFFIX.ext")

    try:
        fileparts[1] = str(int(fileparts[1])).zfill(length)
    except ValueError:
        raise ValueError(f"Expected numeric slice number, got: {fileparts[1]} in {filename}")

    out_filename = filename.parent / '_'.join(fileparts)

    if save and out_filename != filename:
        filename.rename(out_filename)

    return out_filename
