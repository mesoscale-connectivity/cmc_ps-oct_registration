#!/usr/bin/env python
'''
PSOCT processing functions for CMC multimodal analysis

Authors: Saad Jbabdi            <saad.jbabdi@ndcn.ox.ac.uk>
         Vasilis Karlaftis      <vasilis.karlaftis@ndcn.ox.ac.uk>

Copyright (C) 2025 University of Oxford
'''

import os
import json
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

# Lookup table for orientation information
OrientationLookup = {'sagittal': ' x', 'coronal':  'y', 'axial':  'z'}
# FSL convention for orientation
FSLconvention     = {'sagittal': 'LR', 'coronal': 'PA', 'axial': 'IS'}

class psoct:

    def __init__(self, inp_path, seq_params, slide_range=None, lowres=True, reg_modality='Retardance', verbose=False):
        self.inp_path       = Path(inp_path)
        self.reg_modality   = reg_modality
        self.image_files    = None
        self.slide_res      = None
        self.seq_params     = None
        self.orientation    = None
        self.reverse_slides = False
        self.downsample     = 1
        self._slide_range   = _UNSET
        self.slide_numbers  = None
        self.output_path    = None
        self.verbose        = verbose

        if self.verbose:
            print(f"\nReading input information for '{self.inp_path}' ...")
        # check validity of input folder
        self.__check_input_folder()
        # check validity of seq_params file
        self._read_seq_params(seq_params)
        # run some "processing" during initialisation
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
        self.rel_shifts = {}
        self.abs_shifts = {}
        self.slide_deck_img = None

    def _find_missing_slides(self):
        '''Get list of missing slides.'''
        if self.slide_range is not None and self.slide_numbers is not None:
            self.missing_slides = list(set(np.arange(self.slide_range[0], self.slide_range[1]+1)) - set(self.slide_numbers))
            self.missing_slides = list(map(int, self.missing_slides))
            if self.verbose and len(self.missing_slides) > 0:
                print(f"\tFound {len(self.missing_slides)} missing slides: {self.missing_slides}")

    def _load_slides(self):
        for sl, f in zip(self.slide_numbers, self.image_files):
            # slide_range is inclusive
            if (sl>=self.slide_range[0])&(sl<=self.slide_range[1]):
                self.slides_dict[sl] = f  # slides_dict contains file names, not data

    def __check_input_folder(self):
        # check the MRI & PSOCT folders
        folders = [p.name for p in self.inp_path.iterdir() if p.is_dir()]
        if not {'MRI', 'PSOCT'}.issubset(folders):
            raise FileNotFoundError(f"Input folder {self.inp_path} does not contain 'MRI' or 'PSOCT' folders.")
        # check the modality folders
        modalities = [p.name for p in (self.inp_path / 'PSOCT').iterdir() if p.is_dir()]
        if not {self.reg_modality}.issubset(modalities):
            raise FileNotFoundError(f"PSOCT folder does not contain a {self.reg_modality} folder.")
        self.inp_path = self.inp_path / 'PSOCT' / self.reg_modality
        if self.verbose:
            print('\tInput folder read successfully.')

    def __check_seq_params(self, seq_params):
        import json
        # check if file exists and has a valid format
        seq_file = Path(seq_params)
        if not seq_file.is_file():
            raise FileNotFoundError(f"{seq_file} file not found.")
        try:
            with open(seq_file, "r") as f:
                self.seq_params = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format for {seq_file}: {e}")
        # check if the file has the required fields
        mandatory_keys = {'orientation', 'slice_order', 'in-plane resolution', 'out-of-plane resolution'}
        if not mandatory_keys.issubset(self.seq_params):
            raise ValueError(f"{seq_file} file does not contain all mandatory keys: {mandatory_keys}")
    
    def _read_seq_params(self, seq_params):
        # check if JSON file is of a valid format
        self.__check_seq_params(seq_params)
        # if all correct, read data orientation
        self.orientation = self.seq_params['orientation']
        # make orientation lowercase to account for potentially capitalising first letter
        if isinstance(self.orientation, str):
            self.orientation = self.orientation.lower()
        else:
            raise ValueError(f"Unexpected orientation datatype: {type(self.orientation)}. Expected a string.")
        # read slice_order and check if it is compatible with the orientation
        slice_order = self.seq_params['slice_order']
        if self.orientation == 'sagittal':
            if slice_order not in {'LR', 'RL'}:
                raise ValueError(f"Unexpected 'slice_order' value: {slice_order}. Expected 'LR' or 'RL'.")
        elif self.orientation == 'coronal':
            if slice_order not in {'AP', 'PA'}:
                raise ValueError(f"Unexpected 'slice_order' value: {slice_order}. Expected 'AP' or 'PA'.")
        elif self.orientation == 'axial':
            if slice_order not in {'SI', 'IS'}:
                raise ValueError(f"Unexpected 'slice_order' value: {slice_order}. Expected 'SI' or 'IS'.")
        else:
            raise ValueError(f"Unexpected orientation value: {self.orientation}. Expected 'sagittal', 'coronal' or 'axial'.")
        if self.verbose:
            print('\tPSOCT sequence parameters read successfully.')
        # check if slice_order is compatible with FSL convention, otherwise reverse slides during processing
        if slice_order != FSLconvention[self.orientation]:
            self.reverse_slides = True

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
        if self.verbose:
            print('\tMissing slides have been interpolated successfully.')

    def _find_central_slide(self):
        # Find the size of each slide (excluding the interpolated ones)
        all_slides = np.sort(list(set(self.slides_dict.keys()) - set(self.interpolated_slides)))
        all_sizes = np.zeros(len(all_slides))
        for slide in range(len(all_slides)):
            all_sizes[slide] = np.count_nonzero(get_image(self.slides_dict, all_slides[slide]))
        # Find all slides that have max size and take the median as the central slide
        max_indices = np.where(all_sizes == np.max(all_sizes))[0]
        central_slide_num = all_slides[round(np.median(max_indices))]
        return central_slide_num
    
    def _find_max_shape(self):
        # Independent function to find the max shape across slides, irrespective of zeroes
        # Find the size of each slide (excluding the interpolated ones)
        all_slides = np.sort(list(set(self.slides_dict.keys()) - set(self.interpolated_slides)))
        max_size = 0
        idx = None
        for slide in range(len(all_slides)):
            size = np.prod(get_image(self.slides_dict, all_slides[slide]).shape)
            if size > max_size:
                max_size = size
                idx = slide
        if idx is None:
            raise ValueError("All input images have zero size!")
        else:
            max_slide_shape = get_image(self.slides_dict, all_slides[idx]).shape
        return max_slide_shape
    
    def _get_ref_slide(self, ref):
        if ref == 'centre':
            ref_slide = self._find_central_slide()
        elif ref == 'first':
            ref_slide = np.min(list(self.slides_dict.keys()))
        elif ref == 'last':
            ref_slide = np.max(list(self.slides_dict.keys()))
        else:
            raise ValueError(f'Unexpected reference method {ref} for alignment.')
        ref_shape = self._find_max_shape()
        return ref_slide, ref_shape

    def align(self, ref='centre', thr=0):
        ''' This method calculates the shifts between each slide and its neighbour.
        If the slide is before the central slide, it looks at the neighbour in front,
        otherwise look at the neighbour behind

        Parameters:
        - ref: reference mode for alignment ('centre' for using the central slide)
        - thr: shift threshold. Any shifts lower than this are ignored (to minimize drifts)
        '''

        self.ref_slide, self.ref_shape = self._get_ref_slide(ref)
        if self.verbose:
            print(f"\tReference slide for alignment: {self.ref_slide}")
        # Use all slides for alignment (including interpolated ones)
        slides = sorted(list(self.slides_dict.keys()))
        # TODO for interpolated_slides the alignment could be skipped?
        if self.verbose:
            print('\tRelative alignment values:')
        for sl in slides:
        # Get image from dataframe
            img = get_image(self.slides_dict, sl) 
            if sl == self.ref_slide:
                t = np.array([0, 0]) # no shift if it is central slide
            else:
                if sl < self.ref_slide:
                    tgt = get_image(self.slides_dict, sl+1)
                else:
                    tgt = get_image(self.slides_dict, sl-1)
                t  = calc_shift(img, tgt, self.ref_shape)
                # don't worry about small shifts
                t[0] = t[0] if np.abs(t[0])>thr else 0.
                t[1] = t[1] if np.abs(t[1])>thr else 0.
            # Store shifts
            self.rel_shifts[sl] = t
            if self.verbose:
                print('\t\t', sl, self.rel_shifts[sl])

        # Calculate absolute shifts
        self._calc_total_shift(self.rel_shifts)

    def _calc_total_shift(self, rel_shifts_dict):
        for sl in rel_shifts_dict.keys():
            self.abs_shifts[sl] = get_total_shift(np.array(list(rel_shifts_dict.values())), sl, 
                                                  self.ref_slide, first_slide=self.slide_range[0])

    # Function to be called and "automate" the registration steps
    def run_registration(self, bad_slides=None, align_ref='centre', align_thr=0, plot_alignment=False):
        if self.verbose:
            print('\nStarting slide registration process ...')
        self.label_bad_slides(indices=bad_slides)
        self.interpolate_missing_slides()
        self.align(ref=align_ref, thr=align_thr)
        if plot_alignment:
            plot_shifts(self.rel_shifts.keys(), self.rel_shifts.values(), '-o')
            plot_shifts(self.abs_shifts.keys(), self.abs_shifts.values(), '-')
        if self.verbose:
            print('Slide registration completed.')
    

    def _create_slide_deck(self, downsample=1, applyshift=True):
        slide_deck = []
        for sl in self.abs_shifts.keys():
            img_padded = pad_image(get_image(self.slides_dict, sl), self.ref_shape)
            if applyshift:
                img_padded = shift(img_padded, self.abs_shifts[sl])
            slide_deck.append(img_padded[::downsample,::downsample])
        # Stack images into a 3D array
        slide_deck = np.stack(slide_deck, axis=2)
        # Reorient slide deck to make coronal
        if self.orientation == 'sagittal':
            slide_deck = np.transpose(slide_deck,(2,0,1)).copy()
            if self.reverse_slides:
                slide_deck = np.flip(slide_deck, axis=0)
        elif self.orientation == 'coronal':
            slide_deck = np.transpose(slide_deck,(0,2,1)).copy()
            if self.reverse_slides:
                slide_deck = np.flip(slide_deck, axis=1)
        elif self.orientation == 'axial':
            # order of indices is already correct
            if self.reverse_slides:
                slide_deck = np.flip(slide_deck, axis=2)
        else:
            raise ValueError(f"Unexpected orientation value: {self.orientation}")
        return slide_deck
    
    def apply_registration(self, output_name=None, downsample=1):
        # TODO add this in a separate function?
        # Include pixel dimension in the header
        orig_pixel      = self.seq_params['in-plane resolution']
        lr_pixel        = orig_pixel * self.downsample
        lr_pixel_down   = lr_pixel * downsample
        slice_thickness = self.seq_params['out-of-plane resolution']
        # Create voxel dimension matrix based on orientation
        if self.orientation == 'sagittal':
            voxdim = [slice_thickness, lr_pixel_down, lr_pixel_down]
        elif self.orientation == 'coronal':
            voxdim = [lr_pixel_down, slice_thickness, lr_pixel_down]
        elif self.orientation == 'axial':
            voxdim = [lr_pixel_down, lr_pixel_down, slice_thickness]
        else:
            raise ValueError(f"Unexpected orientation value: {self.orientation}")
        matrix = np.eye(4)
        for i in range(3):
            matrix[i,i] = voxdim[i]

        if self.verbose:
            print('\tCreating slide deck image ...', end=' ')
        slide_deck = self._create_slide_deck(downsample, applyshift=True)
        # TODO consider using io.save_nifti instead
        self.slide_deck_img = Image(slide_deck, xform=matrix)
        if output_name is not None:
            self.slide_deck_img.save(self.output_path / output_name)
        if self.verbose:
            print('Done.')

    # TODO add fnirt option
    def align_mri_to_psoct(self, mri_ref):
        # Register MRI to slide_deck
        # TODO does this need to be an image or could it be a filename?
        if mri_ref.is_absolute():
            mri_ref_fullpath = mri_ref
        else:
            mri_ref_fullpath = self.inp_path.parent.parent / mri_ref
        mri_img = Image(mri_ref_fullpath)
        matfile = self.output_path / 'mri_to_slides.mat'
        outfile = self.output_path / Path(mri_ref).name.replace('.nii.gz','_to_slides')
        if self.verbose:
            print('\tRunning MRI-to-PSOCT registration ...', end=' ')
        flirt(src=mri_img, ref=self.slide_deck_img, out=outfile, omat=matfile, dof=12, interp='spline')
        if self.verbose:
            print('Done.')
        return matfile, outfile
    
    def align_psoct_to_mri(self, matfile, mri_ref):
        mat     = np.loadtxt(matfile)
        mat_inv = np.linalg.inv(mat)
        outfile = self.output_path / 'slide_deck_to_mri'
        if mri_ref.is_absolute():
            mri_ref_fullpath = mri_ref
        else:
            mri_ref_fullpath = self.inp_path.parent.parent / mri_ref

        xform = flirtMatrixToSform(mat_inv,srcImage=self.slide_deck_img,refImage=Image(mri_ref_fullpath))
        slide_img_hdr = Image(self.slide_deck_img.data, xform = xform)
        slide_img_hdr.save(outfile)
        np.savetxt(self.output_path / 'slides_to_mri.mat', mat_inv, fmt='%.10f', delimiter=' ')
        
        return outfile

    def update_nifti_headers(self, slide_deck):
        # Add header information to single slides
        # This will be useful for visualising high resolution slides on top of the MRI

        # First split the slide deck to get individual sides "correct" header
        if self.orientation in OrientationLookup.keys():
            indiv_slides = fslsplit(src=slide_deck, out=LOAD, dim=OrientationLookup[self.orientation])
        else:
            raise ValueError(f"Unexpected orientation value: {self.orientation}")
        
        slide_numbers = sorted(list(self.slides_dict.keys()), reverse=self.reverse_slides)
        split_numbers = sorted(list(indiv_slides.keys()))

        if self.verbose:
            print('\tUpdating headers for slides:', end=' ')
        for sl, idx in zip(slide_numbers, split_numbers):
            img = indiv_slides[idx]
            # Get the relative path and update the slide number for the interpolated slides
            rel_path = self.slides_dict[sl].relative_to(self.inp_path)
            fileparts = rel_path.name.split('_')
            fileparts[1] = str(sl).zfill(3)
            rel_path = rel_path.parent / '_'.join(fileparts)
            filename = self.output_path / self.reg_modality / str(rel_path).replace('.nii.gz', '_hdr.nii.gz')
            os.makedirs(filename.parent, exist_ok=True)
            Image(img.get_fdata(), header=img.header).save(filename)
            indiv_slides[idx] = filename
            if self.verbose:
                print(sl, end=',')
        if self.verbose:
            print(' Done.')
        return indiv_slides

    def apply_to_highres_images(self, indiv_slides, other_images=['Retardance']):
        # Now apply this header to the high res images
        # Note: they need to be zero-padded first and shifted!!

        # convert other_images to a list if not already
        if not isinstance(other_images, list):
            other_images = [other_images]
        
        # run alignment across all modalities
        for mod in other_images:
            if self.verbose:
                print(f"\tApplying registration matrix to '{mod}' slides ...")
            if self.slide_res == 'highres' and mod == self.reg_modality:
                return

            data_path = self.inp_path.parent / mod
            data_files = sorted(data_path.glob('Slice_*_En*.nii.gz'))
            # TODO make this more versatile
            mod_slide_numbers = np.array([int(Path(f).name.split('_')[1]) for f in data_files])

            slide_numbers = sorted(list(self.slides_dict.keys()), reverse=self.reverse_slides)
            split_numbers = sorted(list(indiv_slides.keys()))
            
            for sl, idx in zip(slide_numbers, split_numbers):
                # skip bad_slides
                if sl in self.bad_slides:
                    continue
                
                img_lowres = Image(indiv_slides[idx])
                
                # Load Image
                mod_idx = np.where(mod_slide_numbers == sl)[0]
                if len(mod_idx) != 1 and self.verbose:
                    print(f"\t\tUnexpected number of matching files: file number {sl}. Skipping this file.")
                    continue
                
                highres_file = data_files[mod_idx[0]]
                filename = self.output_path / mod / str(highres_file.name).replace('.nii.gz', '_hdr.nii.gz')
                img_highres = Image(highres_file).data[:,:,0]

                # Zero-pad and shift
                highres_shape = [x * self.downsample for x in self.ref_shape]
                img_padded    = pad_image(img_highres, highres_shape)
                img_zp_shift  = shift(img_padded, self.abs_shifts[sl] * self.downsample)
                
                if self.orientation == 'sagittal':
                    newShape = [img_lowres.shape[0], img_lowres.shape[1]*self.downsample, img_lowres.shape[2]*self.downsample]
                elif self.orientation == 'coronal':
                    newShape = [img_lowres.shape[0]*self.downsample, img_lowres.shape[1], img_lowres.shape[2]*self.downsample]
                elif self.orientation == 'axial':
                    newShape = [img_lowres.shape[0]*self.downsample, img_lowres.shape[1]*self.downsample, img_lowres.shape[2]]
                else:
                    raise ValueError(f"Unexpected orientation value: {self.orientation}")
                newShape = np.array(np.round(newShape), dtype=int)

                matrix = affine.rescale(img_lowres.shape, newShape, 'centre')
                matrix = affine.concat(img_lowres.voxToWorldMat, matrix)

                if self.orientation == 'sagittal':
                    img_highres = Image(img_zp_shift[None,:,:], xform=matrix, header=img_lowres.header)
                elif self.orientation == 'coronal':
                    img_highres = Image(img_zp_shift[:,None,:], xform=matrix, header=img_lowres.header)
                elif self.orientation == 'axial':
                    img_highres = Image(img_zp_shift[:,:,None], xform=matrix, header=img_lowres.header)
                else:
                    raise ValueError(f"Unexpected orientation value: {self.orientation}")
                os.makedirs(filename.parent, exist_ok=True)
                img_highres.save(filename)
            if self.verbose:
                print(f"\tRegistration of '{mod}' slides completed.")
    
    def _save_shifts(self):
        with open(self.output_path / "abs_shifts.json", "w") as f:
            json.dump({int(k): v.tolist() for k, v in self.abs_shifts.items()}, f)
        with open(self.output_path / "rel_shifts.json", "w") as f:
            json.dump({int(k): v.tolist() for k, v in self.rel_shifts.items()}, f)
        if self.verbose:
            print('\tRelative and absolute shifts saved.')

    def run_slide_deck_creation(self, other_images, output_path, mri_ref, downsample=1):
        if self.verbose:
            print('\nStarting slide deck creation ...')
        mri_ref = Path(mri_ref)
        # TODO consider moving the next two lines into apply_registration
        self.output_path = Path(output_path)
        os.makedirs(self.output_path, exist_ok=True)
        # save shifts to a text file
        self._save_shifts()
        # save slide decks and header information
        self.apply_registration(output_name='slide_deck', downsample=downsample)
        matfile, _ = self.align_mri_to_psoct(mri_ref)
        psoct_to_mri_file = self.align_psoct_to_mri(matfile, mri_ref)
        indiv_slides = self.update_nifti_headers(psoct_to_mri_file)
        self.apply_to_highres_images(indiv_slides, other_images)
        if self.verbose:
            print(f"\nPSOCT pipeline completed and results saved to {self.output_path}")
        return indiv_slides
    