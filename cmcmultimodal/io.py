from fsl.data.image import Image
from scipy.io import loadmat
import numpy as np
from pathlib import Path

def mat2nii(mat_file, nii_file, nii_lowres_file = None, downsample = 0):
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

    Image(X).save(nii_file)

    # Also output low res
    if downsample > 0 and nii_lowres_file is not None:
        Image(X[::downsample, ::downsample]).save(nii_lowres_file)


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
        fileparts[-2] = str(int(fileparts[-2])).zfill(length)
    except ValueError:
        raise ValueError(f"Expected numeric slice number, got: {fileparts[-2]} in {filename}")

    out_filename = filename.parent / '_'.join(fileparts)

    if save and out_filename != filename:
        filename.rename(out_filename)

    return out_filename
