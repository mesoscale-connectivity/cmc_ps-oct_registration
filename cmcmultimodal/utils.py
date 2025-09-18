#!/usr/bin/env python
'''
Utility functions for CMC multimodal analysis

Authors: Saad Jbabdi            <saad.jbabdi@ndcn.ox.ac.uk>
         Vasilis Karlaftis      <vasilis.karlaftis@ndcn.ox.ac.uk>

Copyright (C) 2025 University of Oxford
'''

# Helper Functions
import numpy as np
from matplotlib import pyplot as plt
from numpy.fft import fft, ifft, fft2, ifft2, ifftshift
from scipy.ndimage import shift
from fsl.data.image import Image
from pathlib import Path

def cross_correlate_2d(x, h):
    """Calculate cross-correlation between 2D images using Fourier
    """
    h = ifftshift(ifftshift(h, axes=0), axes=1)
    return ifft2(fft2(x) * np.conj(fft2(h))).real

def crop(x, s, axis):
    start = (x.shape[axis]-s)//2
    if axis==0:
        return x[start:start+s,:]
    else:
        return x[:,start:start+s]

def pad_image(x_template, shape):
    """Zero-pad image to fit shape of a target image
    """
    # Undo MariPenn's bad padding
    # x_template = crop_zeros(x_template)
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

def calc_shift(src, tgt, shape):
    """Calculate 2D translation that best aligns two images
    """
    src_padded = pad_image(src, shape)
    tgt_padded = pad_image(tgt, shape)
    CC = cross_correlate_2d(src_padded, tgt_padded)
    peak = np.unravel_index(np.argmax(CC, axis=None), CC.shape)
    t    = -peak[0]+CC.shape[0]/2, -peak[1]+CC.shape[1]/2
    return np.array(t)

def plot_overlay(bckg, fore):
    """Overlay two images with foreground as contours
    """
    plt.imshow(bckg, cmap='gray')
    levels = np.quantile(fore, [.7, .8, .99])
    plt.contour(fore, levels=levels, linewidths=1, colors='r')
    plt.xticks([])
    plt.yticks([])

def plot_shifts(slides, shifts, format = '-'):
    plt.figure()
    plt.plot(slides, shifts, format)
    plt.xlabel('Slide number (#)')
    plt.ylabel('Shift [in pixels]')
    plt.legend(['x-shift', 'y-shift'])
    plt.show()

def get_image(D, sl):
    """Get image from slide dictionary
    If filename provided, load it, otherwise, return the array
    """
    if type(D[sl]) == str or isinstance(D[sl], Path):
        return Image(D[sl]).data[:,:,0]
    else:
        # TODO add if D[sl] in np.array, otherwise raise an error
        return D[sl]

def crop_zeros(X):
    ymin = np.min(np.where(X[X.shape[0]//2,:]>0)[0])
    ymax = np.max(np.where(X[X.shape[0]//2,:]>0)[0])
    xmin = np.min(np.where(X[:,X.shape[1]//2]>0)[0])
    xmax = np.max(np.where(X[:,X.shape[1]//2]>0)[0])
    return X[xmin:xmax+1,ymin:ymax+1]

def get_total_shift(all_shifts, sl, central_slide, first_slide=1):
    """Add shifts all the way to central slide
    """
    if sl < central_slide:
        return np.sum(all_shifts[sl-first_slide:central_slide-first_slide+1,:], axis=0)
    else:
        return np.sum(all_shifts[central_slide-first_slide:sl-first_slide+1,:], axis=0)
