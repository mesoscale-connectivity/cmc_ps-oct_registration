#!/usr/bin/env python
'''
Input/output functions for PSOCT data

Authors: Saad Jbabdi            <saad.jbabdi@ndcn.ox.ac.uk>
         Vasilis Karlaftis      <vasilis.karlaftis@ndcn.ox.ac.uk>

Copyright (C) 2025 University of Oxford
'''

import os
import numpy as np
from pathlib import Path
from fsl.data.image import Image


def mat2nii(mat_file, seq_params, nii_file=None, nii_lowres_file=None, downsample=0):
    '''Convert and store a PSOCT .mat file to a NIFTI image.
    Optionally also create a low-resolution version.
    '''
    from scipy.io import loadmat

    # Load matfile and identify data
    # TODO add checks if file exists and can be read
    mat_contents = loadmat(mat_file)
    select       = [not (x[:2]+x[-2:]) == '____'
                    for x in list(mat_contents.keys())]
    key          = np.array(list(mat_contents.keys()))[select][0]
    # Create array
    X = np.array(mat_contents[key])
    # TODO add 'orientation' if statements
    # # Rotate image by 90 degrees counterclockwise and the LR flop - ONLY FOR MOE
    # X = np.fliplr(X).T
    # Rotate image by 90 degrees clockwise - ONLY FOR VLAD
    X = np.flipud(X).T
    # This is common for all orientations (for now)
    X = np.flip(X, axis=0)
    # Deal with orientation
    if X.dtype == np.complex128:
        X = np.angle(X)

    # reduce size
    X = np.asarray(X, dtype=np.float32)

    # deal with nans
    X = np.nan_to_num(X)
    # save high-res NIFTI
    if nii_file is None:
        nii_file = Path(str(mat_file).replace('.mat', '.nii.gz'))
    else:
        nii_file = Path(nii_file)
    os.makedirs(nii_file.parent, exist_ok=True)
    Image(X).save(nii_file)
    update_header_info(nii_file, nii_file, seq_params)

    # Also output low res
    if downsample > 0 and nii_lowres_file is not None:
        nii_lowres_file = Path(str(nii_lowres_file).replace('.nii.gz', f'_downsample_{downsample}.nii.gz'))
        os.makedirs(nii_lowres_file.parent, exist_ok=True)
        Image(X[::downsample, ::downsample]).save(nii_lowres_file)
        update_header_info(nii_lowres_file, nii_lowres_file, seq_params, downsample)
    else:
        nii_lowres_file = None

    return nii_file, nii_lowres_file


def update_header_info(filename, out_filename, seq_params, downsample=1):
    import nibabel as nib
    from cmcmultimodal.core.utils import check_seq_params

    # Read input information
    seq_params = check_seq_params(seq_params)
    orig_img = Image(filename)

    # Create pixel dimension information
    orig_pixel      = seq_params['in-plane resolution']
    lr_pixel        = orig_pixel * downsample
    slice_thickness = seq_params['out-of-plane resolution']
    # Create voxel dimension matrix (2D images should have the first two non-zero dims)
    voxdim = [lr_pixel, lr_pixel, slice_thickness]
    matrix = np.diag([*voxdim, 1])
    # Create appropriate Nifti header
    hdr = nib.Nifti1Header()
    hdr.set_xyzt_units(xyz='mm', t='sec')
    hdr.set_sform(matrix, code=2)
    # Save image file
    os.makedirs(out_filename.parent, exist_ok=True)
    Image(orig_img.data, xform=matrix, header=hdr).save(out_filename)


def pad_all_slides(inp_path, out_path=None):
    from cmcmultimodal.core.utils import pad_image

    inp_path = Path(inp_path)
    if out_path is None:
        out_path = inp_path
    else:
        os.makedirs(out_path, exist_ok=True)

    # find all valid files in input folder
    image_files = sorted(inp_path.glob('Slice_*_En*.nii.gz'))
    slide_numbers = [int(Path(f).name.split('_')[1]) for f in image_files]

    slides_dict = {}
    for sl, f in zip(slide_numbers, image_files):
        # slide_range is inclusive
        slides_dict[sl] = f  # slides_dict contains file names, not data

    # find max shape of slides
    ref_shape = find_max_shape(slides_dict)
    print(f"New slide shape for zero-padding: {ref_shape}")
    
    # zero-pad all slides to max shape
    for f in image_files:
        img   = Image(f)
        data  = pad_image(img.data[...,0], ref_shape)
        Image(data[:,:,None], header=img.header).save(out_path / f.name)

def find_max_shape(slides_dict):
    # Independent function to find the max shape across slides, irrespective of zeroes
    # Find the size of each slide
    all_slides = np.sort(list(slides_dict.keys()))
    max_size = 0
    idx = None
    for i, slide in enumerate(all_slides):
        size = np.prod(Image(slides_dict[slide]).shape)
        if size > max_size:
            max_size = size
            idx = i
    if idx is None:
        raise ValueError("All input images have zero size!")
    else:
        max_slide_shape = Image(slides_dict[all_slides[idx]]).shape
    return max_slide_shape
    

def zeropad(filename, length=3, save=False):
    '''Zero pad the slice number in a filename of format:
    PREFIX_<slice>_SUFFIX.ext

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
        raise ValueError(f"Unexpected filename format: {filename}.\
                         Expected format: PREFIX_<slice>_SUFFIX.ext")

    try:
        fileparts[1] = str(int(fileparts[1])).zfill(length)
    except ValueError:
        raise ValueError(f"Expected numeric slice number, got:\
                         {fileparts[1]} in {filename}")

    out_filename = filename.parent / '_'.join(fileparts)

    if save and out_filename != filename:
        filename.rename(out_filename)

    return out_filename
