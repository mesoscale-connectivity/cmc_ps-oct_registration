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

from cmcmultimodal.utils    import get_image, calc_shift, get_total_shift, \
                                   plot_shifts, pad_image
from scipy.ndimage          import shift
from fsl.data.image         import Image
from fsl.wrappers           import flirt
from fsl.transform.flirt    import flirtMatrixToSform
from fsl.wrappers.avwutils  import fslsplit
from fsl.wrappers           import LOAD
import fsl.transform.affine as affine

# create sentinel object for slide_range
_UNSET = object()

class psoct:

    def __init__(self, inp_path, slide_range=None, lowres=True, reg_modality='Retardance'):
        self.inp_path = Path(inp_path)
        self.reg_modality = reg_modality
        self.image_files = None
        self.slide_res = None
        self.downsample = 1
        self._slide_range = _UNSET
        self.slide_numbers = None
        self.output_path = None

        # run some "processing" during initialisation
        self.__check_input_folder()
        self._find_all_slides(lowres=lowres)
        # run slide_range setter after finding all the slides
        self.slide_range = slide_range

    @property
    def slide_range(self):
        return self._slide_range

    @slide_range.setter
    def slide_range(self, value):
        # if the new range is same as current then skip
        if value == self._slide_range:
            return
        if value is None:
            self._slide_range = tuple([min(self.slide_numbers),max(self.slide_numbers)])
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            if all(isinstance(v, int) for v in value) and value[0] <= value[1]:
                if value[0] < min(self.slide_numbers):
                    value[0] = min(self.slide_numbers)
                    print('WARNING: slide_range exceeds minimum slide number! Changing to min...')
                if value[1] > max(self.slide_numbers):
                    value[1] = max(self.slide_numbers)
                    print('WARNING: slide_range exceeds maximum slide number! Changing to max...')
                self._slide_range = tuple(value)
            else:
                raise ValueError("slide_range must be a tuple/list of two integers (start <= end)")
        else:
            raise TypeError("slide_range must be a tuple or list of two integers")
        # reset attributes that are affected by slide_range changes
        self.__reset_attributes()
        # update missing slides and load slides
        self._find_missing_slides()
        self._load_slides()

    def __reset_attributes(self):
        ''' Initialise or reset attributes that depend on selected slide_range
        '''
        self.missing_slides = []
        self.slides_dict = {}
        self.bad_slides = []
        self.interpolated_slides = []
        self.ref_slide = 0
        self.ref_shape = 0
        self.slide_deck_img = None

    def __check_input_folder(self):
        folders = [p.name for p in self.inp_path.iterdir() if p.is_dir()]
        if not {'MRI', 'PSOCT'}.issubset(folders):
            raise FileNotFoundError(f"Input folder {self.inp_path} does not contain 'MRI' or 'PSOCT' folders.")
        modalities = [p.name for p in (self.inp_path / 'PSOCT').iterdir() if p.is_dir()]
        if not {self.reg_modality}.issubset(modalities):
            raise FileNotFoundError(f"PSOCT folder does not contain a {self.reg_modality} folder.")
        self.inp_path = self.inp_path / 'PSOCT' / self.reg_modality

    def _find_all_slides(self, lowres=False):
        # TODO this should get the 'lowres' folder and the filenames from the io.py functions
        if lowres:
            self.image_files = sorted(self.inp_path.glob('lowres/' + 'Slice_*_En*.nii.gz'))
            self.slide_res = 'lowres'
            self.downsample = 10
        else:
            self.image_files = sorted(self.inp_path.glob('Slice_*_En*.nii.gz'))
            self.slide_res = 'highres'
        # TODO: the specificity of the file format is interlinked with the io.py
        self.slide_numbers = [int(Path(f).name.split('_')[1]) for f in self.image_files]

    def _load_slides(self):
        # TODO optimise the performance by looping through the shortest range
        # i.e. slide_range[1]-slide_range[0] vs slide_numbers[-1]-slide_numbers[0]
        # slides_dict contains file names, not data
        for sl, f in zip(self.slide_numbers, self.image_files):
            # slide_range is inclusive
            if (sl>=self.slide_range[0])&(sl<=self.slide_range[1]):
                self.slides_dict[sl] = f
                # TODO should we add a slides_dict_numbers to keep the indices of the original selection?

    def _find_missing_slides(self):
        '''Get list of missing slides.'''
        if self.slide_range is not None and self.slide_numbers is not None:
            self.missing_slides = list(set(np.arange(self.slide_range[0], self.slide_range[1]+1)) - set(self.slide_numbers))

    def label_bad_slides(self, indices=None):
        ''' List of bad slides as defined by visual assessment.'''
        if indices is not None and self.slide_range is not None:
            self.bad_slides = [sl for sl in indices if sl>=self.slide_range[0] and sl<=self.slide_range[1]]

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
            # weights for averaging - not in use
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
        if verbose:
            print('Reference slide for alignment: ', self.ref_slide)
        # Use all slides for alignment (including interpolated ones)
        # TODO change this to a dict to have them paired?
        slides = np.sort(list(self.slides_dict.keys()))
        rel_shifts = np.zeros((len(slides), 2))
        # TODO for interpolated_slides the alignment could be skipped?
        for sl in range(len(slides)):
        # Get image from dataframe
            img = get_image(self.slides_dict, slides[sl]) 
            if slides[sl] == self.ref_slide:
                t = [0, 0] # no shift if it is central slide
            else:
                if slides[sl] < self.ref_slide:
                    tgt = get_image(self.slides_dict, slides[sl]+1)
                else:
                    tgt = get_image(self.slides_dict, slides[sl]-1)
                t  = calc_shift(img, tgt, self.ref_shape)
                # don't worry about small shifts
                t[0] = t[0] if np.abs(t[0])>thr else 0.
                t[1] = t[1] if np.abs(t[1])>thr else 0.
            # Store shifts
            rel_shifts[sl] = t
            if verbose:
                print(slides[sl], rel_shifts[sl])
        # Calculate absolute shifts
        abs_shifts = self._calc_total_shift(slides, rel_shifts)

        return slides, rel_shifts, abs_shifts

    def _calc_total_shift(self, slides, rel_shifts):
        # TODO consider changing the outputs to attributes
        abs_shifts = np.zeros(rel_shifts.shape)
        for sl in range(len(slides)):
            abs_shifts[sl] = get_total_shift(rel_shifts, slides[sl], 
                                                self.ref_slide, 
                                                first_slide=self.slide_range[0])
        return abs_shifts

    # Function to be called and "automate" the registration steps
    def run_registration(self, bad_slides=None, align_ref='centre', align_thr=0, plot_alignment=False, verbose=False):
        self.label_bad_slides(indices=bad_slides)
        self.interpolate_missing_slides()
        slides, rel_shifts, abs_shifts = self.align(ref=align_ref, thr=align_thr, verbose=verbose)
        if plot_alignment:
            plot_shifts(slides, rel_shifts, '-o')
            plot_shifts(slides, abs_shifts, '-')
        return slides, rel_shifts, abs_shifts
    

    def _create_slide_deck(self, slides, shifts, orientation, downsample=1, applyshift=True):
        slide_deck = []
        for sl in range(len(slides)):
            img_padded = pad_image(get_image(self.slides_dict, slides[sl]), self.ref_shape)
            if applyshift:
                img_padded = shift(img_padded, shifts[sl])
            slide_deck.append(img_padded[::downsample,::downsample])
        slide_deck = np.stack(slide_deck, axis=2)
        # Reorient slide deck to make coronal
        # TODO orientation should become an attribute?
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
    
    def apply_registration(self, slides, shifts, orientation, output_name=None, downsample=1, verbose=False):
        # TODO add this in a separate function and make it more data-driven
        # TODO read this from the json file with sequence details
        # Include pixel dimension in the header
        orig_pixel      = 0.006
        lr_pixel        = orig_pixel * self.downsample
        lr_pixel_down   = lr_pixel * downsample
        slice_thickness = 0.25
        voxdim          = [lr_pixel_down, slice_thickness, lr_pixel_down]
        matrix = np.eye(4)
        for i in range(3):
            matrix[i,i] = voxdim[i]

        if verbose:
            print('Creating slide deck image...', end=' ')
        slide_deck = self._create_slide_deck(slides, shifts, orientation, downsample, applyshift=True)
        # TODO consider using io.save_nifti instead
        self.slide_deck_img = Image(slide_deck, xform=matrix)
        if output_name is not None:
            self.slide_deck_img.save(self.output_path / output_name)
        if verbose:
            print('Done.')

    # TODO add fnirt option
    def align_mri_to_psoct(self, mri_ref, verbose=False):
        # Register MRI to slide_deck
        # TODO does this need to be an image or could it be a filename?
        mri_img = Image(mri_ref)
        matfile = self.output_path / 'mri_to_slides.mat'
        outfile = self.output_path / Path(mri_ref).name.replace('.nii.gz','_to_slides')
        if verbose:
            print('Running flirt...', end=' ')
        flirt(src=mri_img, ref=self.slide_deck_img, out=outfile, omat=matfile, dof=12, interp='spline')
        if verbose:
            print('Done.')
        return matfile, outfile
    
    def align_psoct_to_mri(self, matfile, mri_ref):
        mat     = np.loadtxt(matfile)
        mat_inv = np.linalg.inv(mat)
        outfile = self.output_path / 'slide_deck_to_mri'

        xform = flirtMatrixToSform(mat_inv,srcImage=self.slide_deck_img,refImage=Image(mri_ref))
        slide_img_hdr = Image(self.slide_deck_img.data, xform = xform)
        slide_img_hdr.save(outfile)
        np.savetxt(self.output_path / 'slides_to_mri.mat', mat_inv, fmt='%.10f', delimiter=' ')
        return outfile

    def update_nifti_headers(self, slide_deck, orientation):
        # Add header information to single slides
        # This will be useful for visualising high resolution slides on top of the MRI

        # First split the slide deck to get individual sides "correct" header
        # TODO add other cases
        if orientation == 'coronal':
            indiv_slides = fslsplit(src=slide_deck, out=LOAD, dim='y')
        else:
            raise ValueError("Only 'coronal' orientation is currently supported!")
        
        # Careful here: slides are now going in the opposite direction
        # I.e. idx:0->N-1 and sl:last->first
        # first, last = self.slide_range
        # lookup = dict( zip(np.arange(first, last+1)[::-1], range(last-first+1)) )
        # TODO read this from the json file with sequence details
        slide_numbers = sorted(list(self.slides_dict.keys()), reverse=True)
        split_numbers = sorted(list(indiv_slides.keys()))

        for sl, idx in zip(slide_numbers, split_numbers):#lookup:
            # idx = lookup[sl]
            # print(idx, sl)
            img = indiv_slides[idx]#[f'out{str(idx).zfill(4)}']
            # Get the relative path and update the slide number for the interpolated slides
            rel_path = self.slides_dict[sl].relative_to(self.inp_path)
            fileparts = rel_path.name.split('_')
            fileparts[1] = str(sl).zfill(3)
            rel_path = rel_path.parent / '_'.join(fileparts)
            filename = self.output_path / self.reg_modality / str(rel_path).replace('.nii.gz', '_hdr.nii.gz')
            os.makedirs(filename.parent, exist_ok=True)
            Image(img.get_fdata(), header=img.header).save(filename)
            indiv_slides[idx] = filename
        
        return indiv_slides

    def apply_to_highres_images(self, indiv_slides, shifts, orientation, other_images=['Retardance']):
        # Now apply this header to the high res images
        # Note: they need to be zero-padded first and shifted!!

        # convert other_images to a list if not already
        if not isinstance(other_images, list):
            other_images = [other_images]
        
        # run alignment across all modalities
        for mod in other_images:
            if self.slide_res == 'highres' and mod == self.reg_modality:
                return

            data_path = self.inp_path.parent / mod
            data_files = sorted(data_path.glob('Slice_*_En*.nii.gz'))
            # TODO make this more versatile
            mod_slide_numbers = np.array([int(Path(f).name.split('_')[1]) for f in data_files])

            # Careful here: slides are now going in the opposite direction
            # I.e. idx:0->N-1 and sl:last->first
            slide_numbers = sorted(list(self.slides_dict.keys()), reverse=True)
            split_numbers = sorted(list(indiv_slides.keys()))
            
            for sl, idx, sft in zip(slide_numbers, split_numbers, shifts):
                # skip bad_slides
                if sl in self.bad_slides:
                    continue
                
                img_lowres = Image(indiv_slides[idx])
                
                # Load Image
                mod_idx = np.where(mod_slide_numbers == sl)[0]
                if len(mod_idx) != 1:
                    print(f"Unexpected number of matching files for '{mod}', file number {sl}. Skipping this file.")
                    continue
                
                highres_file = data_files[mod_idx[0]]
                filename = self.output_path / mod / str(highres_file.name).replace('.nii.gz', '_hdr.nii.gz')
                img_highres = Image(highres_file).data[:,:,0]

                # Zero-pad and shift
                highres_shape = [x * self.downsample for x in self.ref_shape]
                img_padded    = pad_image(img_highres, highres_shape)
                img_zp_shift  = shift(img_padded, sft * self.downsample)
                
                # TODO update for other orientations
                if orientation == 'coronal':
                    newShape = [img_lowres.shape[0]*self.downsample, img_lowres.shape[1], img_lowres.shape[2]*self.downsample]
                else:
                    raise ValueError("Only 'coronal' orientation is currently supported!")
                newShape = np.array(np.round(newShape), dtype=int)

                matrix = affine.rescale(img_lowres.shape, newShape, 'centre')
                matrix = affine.concat(img_lowres.voxToWorldMat, matrix)

                if orientation == 'coronal':
                    img_highres = Image(img_zp_shift[:,None,:], xform=matrix, header=img_lowres.header)
                else:
                    raise ValueError("Only 'coronal' orientation is currently supported!")
                os.makedirs(filename.parent, exist_ok=True)
                img_highres.save(filename)
    
    def run_slide_deck_creation(self, slides, abs_shifts, orientation, other_images, output_path, mri_ref, downsample=1):
        # TODO consider moving the next two lines into apply_registration
        self.output_path = Path(output_path)
        os.makedirs(self.output_path, exist_ok=True)
        self.apply_registration(slides, abs_shifts, orientation, output_name='slide_deck', downsample=downsample)
        matfile, _ = self.align_mri_to_psoct(mri_ref)
        psoct_to_mri_file = self.align_psoct_to_mri(matfile, mri_ref)
        indiv_slides = self.update_nifti_headers(psoct_to_mri_file, orientation)
        self.apply_to_highres_images(indiv_slides, abs_shifts, orientation, other_images)
        return indiv_slides
    