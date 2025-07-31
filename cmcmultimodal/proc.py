#!/usr/bin/env python
'''
PSOCT processing functions for CMC multimodal analysis

Authors: Saad Jbabdi            <saad.jbabdi@ndcn.ox.ac.uk>
         Vasilis Karlaftis      <vasilis.karlaftis@ndcn.ox.ac.uk>

Copyright (C) 2025 University of Oxford
'''

import os
import numpy as np
from pathlib import Path
from cmcmultimodal.utils import get_image, calc_shift, get_total_shift, \
                                plot_shifts, pad_image
from scipy.ndimage import shift
from fsl.data.image import Image


class psoct:

    def __init__(self, inp_path, slide_range=None, lowres=True):
        self.inp_path = Path(inp_path)
        self.image_files = None
        self._slide_range = None
        self.slide_numbers = None
        self.missing_slides = []
        self.bad_slides = []
        self.slides_dict = {}
        self.interpolated_slides = []
        self.ref_slide = 0

        self._find_all_slides(lowres=lowres)
        # run slide_range setter after finding all the slides
        self.slide_range = slide_range

    @property
    def slide_range(self):
        return self._slide_range

    @slide_range.setter
    def slide_range(self, value):
        if value is None:
            self._slide_range = None
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            if all(isinstance(v, int) for v in value) and value[0] <= value[1]:
                self._slide_range = tuple(value)
            else:
                raise ValueError("slide_range must be a tuple/list of two integers (start <= end)")
        else:
            raise TypeError("slide_range must be a tuple or list of two integers")
        # update missing slides and load slides
        self._find_missing_slides()
        self._load_slides()
        # TODO check if values exceed the min and max slide number and print a warning
        
    def _find_all_slides(self, lowres=False):
        # TODO this should get the 'lowres' folder and the filenames from the io.py functions
        if lowres:
            # TODO do we need sorted here?
            self.image_files   = sorted(self.inp_path.glob('lowres/' + 'Slice_*_En*.nii.gz'))
        else:
            self.image_files   = sorted(self.inp_path.glob('Slice_*_En*.nii.gz'))
        # TODO: the specificity of the file format is interlinked with the io.py
        self.slide_numbers = [int(Path(f).name.split('_')[1]) for f in self.image_files]

    def _load_slides(self):
        if self.slide_range is not None:
            # TODO optimise the performance by looping through the shortest range
            # i.e. slide_range[1]-slide_range[0] vs slide_numbers[-1]-slide_numbers[0]
            # slides_dict contains file names, not data
            for s, f in zip(self.slide_numbers, self.image_files):
                # slide_range is inclusive
                if (s>=self.slide_range[0])&(s<=self.slide_range[1]):
                    self.slides_dict[s] = f
                    # TODO should we add a slides_dict_numbers to keep the indices of the original selection?
        else:
            self.slides_dict = self.image_files

    def _find_missing_slides(self):
        '''Get list of missing slides.'''
        self.missing_slides = list(set(np.arange(self.slide_range[0], self.slide_range[1]+1)) - set(self.slide_numbers))

    def label_bad_slides(self, indices=None):
        ''' List of bad slides as defined by visual assessment.'''
        if indices is not None:
            self.bad_slides = [i for i in indices if i>=self.slide_range[0] and i<=self.slide_range[1]]

    def _ignore_slides(self):
        # A list of bad and missing slides
        self.interpolated_slides = np.sort(np.unique(self.missing_slides+self.bad_slides)).tolist()
        return self.interpolated_slides

    def interpolate_missing_slides(self):
        slide_arr = np.array(self.slide_numbers)
        for m in self._ignore_slides():
            # nearest slide before
            before = slide_arr[(slide_arr - m)<0]
            if before.size == 0:
                before = np.inf
            else:
                before = before[np.argmin(np.abs(before-m))]
            # nearest slide after
            after = slide_arr[(slide_arr - m)>0]
            if after.size == 0:
                after = np.inf
            else:
                after = after[np.argmin(np.abs(after-m))]
            # If both are Inf (logically impossible but could happen if slide_numbers is empty), raise an error
            if np.isinf(before) and np.isinf(after):
                raise ValueError(f"No available slide before or after missing slide {m}")
            # weights for averaging - TODO not in use
            if not np.isinf(before) and not np.isinf(after) and before != after:
                weights = np.array([m-before, after-m]) / (after-before)
            else:
                weights = np.array([1.0, 0.0]) if np.abs(m-before) < np.abs(after-m) else np.array([0.0, 1.0])
            # change weights to getting closest
            # weights = np.round(weights)
            # create average image (assumes they are the same shape!)
            # TODO also assumes that these indices are part of the slides_dict
            # img_before = Image( self.slides_dict[before] ).data[:,:,0]
            # img_after  = Image( self.slides_dict[after] ).data[:,:,0]
            # Slides_dict[m] = weights[0]*img_before + weights[1]*img_after
            if np.abs(m-before) < np.abs(after-m):
                self.slides_dict[m] = self.image_files[np.where(slide_arr == before)[0][0]]
            else:
                self.slides_dict[m] = self.image_files[np.where(slide_arr == after)[0][0]]

    def _find_central_slide(self):
        # Find the size of each slide (excluding the interpolated ones)
        all_slides = np.sort(list(set(self.slides_dict.keys()) - set(self.interpolated_slides)))
        all_sizes = np.zeros(len(all_slides))
        for slide in range(len(all_slides)):
            all_sizes[slide] = np.count_nonzero(get_image(self.slides_dict, all_slides[slide]))
        # Find all slides that have max size and take the median as the central slide
        max_indices = np.where(all_sizes == np.max(all_sizes))[0]
        central_slide_num = all_slides[round(np.median(max_indices))]
        # Get the shape of the central slide
        central_slide_shape = get_image(self.slides_dict, central_slide_num).shape
        return central_slide_num, central_slide_shape
    
    def _get_ref_slide(self, ref):
        if ref == 'centre':
            ref_slide, ref_shape = self._find_central_slide()
        elif ref == 'first':
            ref_slide = np.min(list(self.slides_dict.keys()))
            ref_shape = get_image(self.slides_dict, ref_slide).shape
        elif ref == 'last':
            ref_slide = np.max(list(self.slides_dict.keys()))
            ref_shape = get_image(self.slides_dict, ref_slide).shape
        else:
            raise ValueError(f'Unexpected reference method {ref} for alignment.')
        return ref_slide, ref_shape

    def align(self, ref='centre', thr=0, verbose=False):
        ''' This method calculates the shifts between each slide and its neighbour.
        If the slide is before the central slide, it looks at the neighbour in front,
        otherwise look at the neighbour behind

        Parameters:
        - ref: reference mode for alignment ('centre' for using the central slide)
        - thr: shift threshold. Any shifts lower than this are ignored (to minimize drifts)
        '''

        self.ref_slide, self.ref_shape = self._get_ref_slide(ref)
        # Use all slides for alignment (including interpolated ones)
        # TODO change this to a dict to have them paired?
        all_slides = np.sort(list(self.slides_dict.keys()))
        all_shifts = np.zeros((len(all_slides), 2))
        # for slide in tqdm(all_slides):
        for slide in range(len(all_slides)):
        # Get image from dataframe
            img = get_image(self.slides_dict, all_slides[slide]) 
            if all_slides[slide] == self.ref_slide:
                t = [0, 0] # no shift if it is central slide
            else:
                if all_slides[slide] < self.ref_slide:
                    tgt = get_image(self.slides_dict, all_slides[slide]+1)
                else:
                    tgt = get_image(self.slides_dict, all_slides[slide]-1)
                t  = calc_shift(img, tgt, self.ref_shape)
                # don't worry about small shifts
                t[0] = t[0] if np.abs(t[0])>thr else 0.
                t[1] = t[1] if np.abs(t[1])>thr else 0.
            # Store shifts
            all_shifts[slide] = t
            if verbose:
                print(all_slides[slide], all_shifts[slide])
        return all_slides, all_shifts

    def calc_total_shift(self, ref_slides, rel_shifts):
        # TODO consider changing the outputs to attributes
        abs_shifts = np.zeros(rel_shifts.shape)
        for slide in range(len(ref_slides)):
            abs_shifts[slide] = get_total_shift(rel_shifts, ref_slides[slide], 
                                                self.ref_slide, 
                                                first_slide=self.slide_range[0])

        return abs_shifts

    # Function to be called and "automate" the registration steps
    def run_registration(self, bad_slides=None, align_ref='centre', align_thr=0, plot_alignment=False, verbose=False):
        self.label_bad_slides(indices=bad_slides)
        self.interpolate_missing_slides()
        slides, rel_shifts = self.align(ref=align_ref, thr=align_thr, verbose=verbose)
        if plot_alignment:
            plot_shifts(slides, rel_shifts, '-o')
        abs_shifts = self.calc_total_shift(slides, rel_shifts)
        if plot_alignment:
            plot_shifts(slides, abs_shifts, '-')
        return slides, rel_shifts, abs_shifts
    

    def create_slide_deck(self, ref_slides, shifts, orientation, downsample=1, applyshift=True):
        slide_deck = []
        for slide in range(len(shifts)):
            img_padded = pad_image(get_image(self.slides_dict, ref_slides[slide]), self.ref_shape)
            if applyshift:
                img_padded = shift(img_padded, shifts[slide])
            slide_deck.append(img_padded[::downsample,::downsample])
        slide_deck = np.stack(slide_deck, axis=2)
        # Reorient slide deck to make coronal
        if orientation == 'coronal':
            slide_deck = np.transpose(slide_deck,(0,2,1)).copy()
            slide_deck = np.flip(slide_deck, axis=1)
        # TODO complete the following cases
        elif orientation == 'axial':
            return 
        elif orientation == 'sagittal':
            return
        else:
            raise ValueError
        return slide_deck
    
    def apply_registration(self, slide_deck, output_name=None, downsample=1, verbose=False):

        # TODO add this in a separate function and make it more data-driven
        # Include pixel dimension in the header
        orig_pixel      = 0.006
        lr_pixel        = orig_pixel * 10 # TODO change 10 to downsample from io
        lr_pixel_down   = lr_pixel * downsample
        slice_thickness = 0.25
        voxdim          = [lr_pixel_down, slice_thickness, lr_pixel_down]
        matrix = np.eye(4)
        for i in range(3):
            matrix[i,i] = voxdim[i]

        if verbose:
            print('Creating slide deck image...', end=' ')
        # TODO consider using io.save_nifti instead
        slide_deck_img = Image(slide_deck, xform=matrix)
        if output_name is not None:
            slide_deck_img.save(output_name)
        if verbose:
            print('Done.')
        return slide_deck_img

    def align_dti_to_psoct(self, slide_deck_img, output_path, dti_ref, verbose=False):
        # Register DTI to slide_deck
        from fsl.wrappers import flirt
        dti_img = Image(dti_ref)
        matfile = Path(output_path) / 'dti_to_slides.mat'
        outfile = Path(output_path) / 'fa_to_slides'
        if verbose:
            print('Running flirt...', end=' ')
        flirt(src=dti_img, ref=slide_deck_img, out=outfile, omat=matfile, dof=12, interp='spline')
        if verbose:
            print('Done.')
        return matfile, outfile

    def run_slide_deck_creation(self, slides, abs_shifts, orientation, output_path, dti_ref, downsample = 1):
        output_path = Path(output_path)
        os.makedirs(output_path, exist_ok=True)
        slide_deck = self.create_slide_deck(slides, abs_shifts, orientation, downsample, applyshift=True)
        slide_deck_img = self.apply_registration(slide_deck, output_path / 'slide_deck', downsample)
        matfile, outfile = self.align_dti_to_psoct(slide_deck_img, output_path, dti_ref)
    