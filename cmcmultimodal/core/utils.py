#!/usr/bin/env python
'''
Utility functions for CMC PS-OCT analysis

Authors: Saad Jbabdi            <saad.jbabdi@ndcn.ox.ac.uk>
         Vasilis Karlaftis      <vasilis.karlaftis@ndcn.ox.ac.uk>

Copyright (C) 2025 University of Oxford
'''

# Helper Functions
import numpy as np
from matplotlib import pyplot as plt
from fsl.data.image import Image
from pathlib import Path
from fsl.wrappers import flirt, LOAD


def check_seq_params(seq_params):
    import json
    # check if file exists and has a valid format
    seq_file = Path(seq_params)
    if not seq_file.is_file():
        raise FileNotFoundError(f"{seq_file} file not found.")
    try:
        with open(seq_file, "r") as f:
            seq_params = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format for {seq_file}: {e}")
    # check if the file has the required fields
    mandatory_keys = {'orientation', 'slice_order', 'in-plane resolution', 'out-of-plane resolution'}
    if not mandatory_keys.issubset(seq_params):
        raise ValueError(f"{seq_file} file does not contain all mandatory keys: {mandatory_keys}")
    return seq_params


def crop(x, s, axis):
    start = (x.shape[axis]-s)//2
    if axis == 0:
        return x[start:start+s, :]
    else:
        return x[:, start:start+s]


def pad_image(x_template, shape):
    """Zero-pad image to fit shape of a target image
    """
    # If pad_shape is smaller, crop
    if x_template.shape[0] > shape[0]:
        x_template = crop(x_template, shape[0], axis=0)
    if x_template.shape[1] > shape[1]:
        x_template = crop(x_template, shape[1], axis=1)
    pad_vert1  = (shape[0] - x_template.shape[0]) // 2
    pad_vert0  = (shape[0] - x_template.shape[0]) - pad_vert1
    pad_horiz1 = (shape[1] - x_template.shape[1]) // 2
    pad_horiz0 = (shape[1] - x_template.shape[1]) - pad_horiz1
    pad_shape = [[pad_vert0, pad_vert1], [pad_horiz0, pad_horiz1]]

    x_template_padded = np.pad(x_template, pad_shape)

    return x_template_padded


def save_padded_slides(slides, shape, hdr, out_fd):
    """Run and store zero-pad images for all slides
    """
    for sl in slides.keys():
        # TODO review if we want the get_image header
        img, _ = get_image(slides, sl)
        img_padded = pad_image(img, shape)
        out_filename = Path(out_fd) / Path(slides[sl]).name
        Image(img_padded, xform=hdr.get_sform(), header=hdr).save(out_filename)
        slides[sl] = out_filename

    return slides


def calc_flirt(src, tgt, cost='corratio'):
    """Calculate 2D registration that best aligns two images
    """
    # Run flirt 2D registration
    out = flirt(src,
                tgt,
                omat=LOAD,
                cost=cost,
                twod=True)
    
    return out['omat']


def plot_overlay(bckg, fore):
    """Overlay two images with foreground as contours
    """
    plt.imshow(bckg, cmap='gray')
    levels = np.quantile(fore, [.7, .8, .99])
    plt.contour(fore, levels=levels, linewidths=1, colors='r')
    plt.xticks([])
    plt.yticks([])


def get_image(D, sl):
    """Get 2D image from slide dictionary
    If filename provided, load it, otherwise, return the array
    """
    if isinstance(D[sl], str) or isinstance(D[sl], Path):
        img = Image(D[sl])
        return img.data.squeeze(), img.header #[..., 0]
    elif isinstance(D[sl], np.ndarray):
        return D[sl], None
    else:
        raise ValueError(f"Unsupported data type: Image should be either an array or file, not {type(D[sl])}")

