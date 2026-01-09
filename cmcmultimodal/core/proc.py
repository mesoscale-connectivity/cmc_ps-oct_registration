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
import nibabel as nib
# import tempfile
# import glob

from cmcmultimodal.core.utils    import get_image, pad_image, calc_flirt#, save_padded_slides
from fsl.data.image              import Image
from fsl.wrappers                import flirt, fnirt, applyxfm, invwarp#, LOAD
from fsl.transform.flirt         import flirtMatrixToSform#, readFlirt, fromFlirt
# from fsl.wrappers.avwutils       import fslsplit
import fsl.transform.affine as affine
# from cmcmultimodal.core.io       import save_nifti
import dask.multiprocessing
import multiprocessing

# set cores for dask parallel processing
NUM_CORES = min(8, multiprocessing.cpu_count() - 1)

# create sentinel object for slide_range
_UNSET = object()

# Lookup table for orientation information
OrientationLookup = {'sagittal': ' x', 'coronal':  'y', 'axial':  'z'}
# FSL convention for orientation
FSLconvention     = {'sagittal': 'LR', 'coronal': 'PA', 'axial': 'IS'}

# set relative path to fnirt config file
fnirt_config = Path(os.path.dirname(__file__)).parent / 'config/fnirt_config'


class psoct:

    def __init__(self, inp_path, seq_params, slide_range=None, lowres=True,
                 psoct_reg_mod='Cross', mri_reg_mod='Retardance', verbose=False):
        self.inp_path       = Path(inp_path)
        self.psoct_reg_mod  = psoct_reg_mod
        self.mri_reg_mod    = mri_reg_mod
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
        self.mri_ref        = None

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
            self._slide_range = tuple([min(self.slide_numbers), max(self.slide_numbers)])
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
        self.rel_mat = {}
        self.abs_mat = {}
        self.slide_deck = None
        self.slide_deck_img = None

    def _find_missing_slides(self):
        '''Get list of missing slides.'''
        if (self.slide_range is not None) and (self.slide_numbers is not None):
            self.missing_slides = list(set(np.arange(min(self.slide_range), max(self.slide_range)+1)) - set(self.slide_numbers))
            self.missing_slides = list(map(int, self.missing_slides))
            if self.verbose and len(self.missing_slides) > 0:
                print(f"\tFound {len(self.missing_slides)} missing slides: {self.missing_slides}")

    def _load_slides(self):
        for sl, f in zip(self.slide_numbers, self.image_files):
            # slide_range is inclusive
            if (sl >= self.slide_range[0]) and (sl <= self.slide_range[1]):
                self.slides_dict[sl] = f  # slides_dict contains file names, not data

    def __check_input_folder(self):
        # check the MRI & PSOCT folders
        folders = [p.name for p in self.inp_path.iterdir() if p.is_dir()]
        if not {'MRI', 'PSOCT'}.issubset(folders):
            raise FileNotFoundError(f"Input folder {self.inp_path} does not contain 'MRI' or 'PSOCT' folders.")
        # check the modality folders
        modalities = [p.name for p in (self.inp_path / 'PSOCT').iterdir() if p.is_dir()]
        if not {self.psoct_reg_mod}.issubset(modalities):
            raise FileNotFoundError(f"PSOCT folder does not contain a {self.psoct_reg_mod} folder.")
        if not {self.mri_reg_mod}.issubset(modalities):
            raise FileNotFoundError(f"PSOCT folder does not contain a {self.mri_reg_mod} folder.")
        self.inp_path = self.inp_path / 'PSOCT' / self.psoct_reg_mod
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
            self.bad_slides = [sl for sl in indices if sl >= self.slide_range[0] and sl <= self.slide_range[1]]

    def _ignore_slides(self):
        # A list of bad and missing slides
        self.interpolated_slides = np.sort(np.unique(self.missing_slides+self.bad_slides)).tolist()
        return self.interpolated_slides

    def interpolate_missing_slides(self):
        slide_arr = np.array(self.slide_numbers)
        for m in self._ignore_slides():
            # nearest slide before
            before = slide_arr[(slide_arr - m) < 0]
            if before.size == 0:
                before = np.inf
            else:
                before = before[np.argmin(np.abs(before-m))]
            # nearest slide after
            after = slide_arr[(slide_arr - m) > 0]
            if after.size == 0:
                after = np.inf
            else:
                after = after[np.argmin(np.abs(after-m))]
            # If both are Inf (logically impossible but could happen if slide_numbers is empty), raise an error
            if np.isinf(before) and np.isinf(after):
                raise ValueError(f"No available slide before or after missing slide {m}")
            # # weights for averaging - not in use
            # if not np.isinf(before) and not np.isinf(after) and before != after:
            #     weights = np.array([m-before, after-m]) / (after-before)
            # else:
            #     weights = np.array([1.0, 0.0]) if np.abs(m-before) < np.abs(after-m) else np.array([0.0, 1.0])
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

    def align(self, ref='centre'):
        ''' This method calculates the registration matrices between each slide and its neighbour.
        If the slide is before the central slide, it looks at the neighbour in front,
        otherwise look at the neighbour behind

        Parameters:
        - ref: reference mode for alignment ('centre' for using the central slide)
        '''
        self.ref_slide, self.ref_shape = self._get_ref_slide(ref)
        if self.verbose:
            print(f"\tReference slide for alignment: {self.ref_slide}, size={self.ref_shape}")
        # Use all slides for alignment (including interpolated ones)
        slides = sorted(list(self.slides_dict.keys()))

        # create image default header
        hdr = nib.Nifti1Header()
        hdr.set_xyzt_units(xyz='mm', t='sec')
        orig_pixel      = self.seq_params['in-plane resolution']
        # TODO add if statement for lowres vs highres
        lr_pixel        = orig_pixel * self.downsample
        slice_thickness = self.seq_params['out-of-plane resolution']
        voxdim = [lr_pixel, lr_pixel, slice_thickness]
        matrix = np.diag([*voxdim, 1])
        hdr.set_sform(matrix, code=2)

        # # Pad slides
        # os.makedirs(self.output_path / 'padded_slices', exist_ok=True)
        # self.slides_dict = save_padded_slides(self.slides_dict, self.ref_shape, hdr, self.output_path / 'padded_slices')

        # TODO for interpolated_slides the alignment could be skipped?
        dask.config.set(scheduler='processes', num_workers=NUM_CORES)
        jobs = []
        for sl in slides:
            # Get image from dataframe
            if sl == self.ref_slide:
                jobs.append(np.eye(4))
            else:
                img = get_image(self.slides_dict, sl)
                if sl < self.ref_slide:
                    tgt = get_image(self.slides_dict, sl+1)
                else:
                    tgt = get_image(self.slides_dict, sl-1)
                # cost was 'leastsq' or 'normcorr' for Retardance reference
                jobs.append(dask.delayed(calc_flirt)(img, tgt, self.ref_shape, hdr, cost='corratio'))
                # jobs.append(dask.delayed(flirt)(img, tgt, omat=LOAD, cost='corratio', twod=True))
        tmp_results = dask.compute(jobs)[0]
        # tmp_results = dask.compute(jobs)['omat'][0]
        self.rel_mat = dict(zip(slides, tmp_results))
        # Calculate absolute transformation matrices
        self._calc_total_mat(self.rel_mat)

    def _calc_total_mat(self, rel_mat_dict):
        slides = sorted(list(self.slides_dict.keys()))
        self.abs_mat = {self.ref_slide: np.eye(4)}
        # left side: propagate forward
        for sl in range(self.ref_slide-1, slides[0]-1, -1):
            self.abs_mat[sl] = self.abs_mat[sl+1] @ self.rel_mat[sl]
        # right side: propagate backward
        for sl in range(self.ref_slide+1, slides[-1]+1):
            self.abs_mat[sl] = self.abs_mat[sl-1] @ self.rel_mat[sl]
        # sort the matrices by slide number
        self.abs_mat = dict(sorted(self.abs_mat.items()))

    # def _create_slide_deck(self, downsample=1):
    #     slide_deck = []
    #     # we assume that abs_mat are sorted by slide number
    #     for sl in self.abs_mat.keys():
    #         img_padded = pad_image(get_image(self.slides_dict, sl), self.ref_shape)
    #             _, src_filename = tempfile.mkstemp(suffix=".nii.gz", prefix="source_")
    #             _, tgt_filename = tempfile.mkstemp(suffix=".nii.gz", prefix="target_")
    #             tgt_padded = pad_image(get_image(self.slides_dict, self.ref_slide), self.ref_shape)
    #             # save_nifti(img_padded, src_filename)
    #             # save_nifti(tgt_padded, tgt_filename)
    #             hdr = nib.Nifti1Header()
    #             hdr.set_xyzt_units(xyz='mm', t='sec')
    #             orig_pixel      = self.seq_params['in-plane resolution']
    #             # TODO add if statement for lowres vs highres
    #             lr_pixel        = orig_pixel * self.downsample
    #             slice_thickness = self.seq_params['out-of-plane resolution']
    #             voxdim = [lr_pixel, lr_pixel, slice_thickness]
    #             matrix = np.diag([*voxdim, 1])
    #             hdr.set_sform(matrix, code=2)
    #             Image(img_padded, xform=matrix, header=hdr).save(src_filename)
    #             Image(tgt_padded, xform=matrix, header=hdr).save(tgt_filename)
    #             img_padded = applyxfm(src_filename,
    #                                   tgt_filename,
    #                                   self.abs_mat[sl],
    #                                   LOAD,
    #                                   twod=True)
    #             img_padded = img_padded['out'].get_fdata()
    #             Path.unlink(src_filename)
    #             Path.unlink(tgt_filename)
    #         slide_deck.append(img_padded[::downsample, ::downsample])
    #     # Stack images into a 3D array
    #     slide_deck = np.stack(slide_deck, axis=2)
    #     # Reorient slide deck to make coronal
    #     if self.orientation == 'sagittal':
    #         slide_deck = np.transpose(slide_deck, (2, 0, 1)).copy()
    #         if self.reverse_slides:
    #             slide_deck = np.flip(slide_deck, axis=0)
    #     elif self.orientation == 'coronal':
    #         slide_deck = np.transpose(slide_deck, (0, 2, 1)).copy()
    #         if self.reverse_slides:
    #             slide_deck = np.flip(slide_deck, axis=1)
    #     elif self.orientation == 'axial':
    #         # order of indices is already correct
    #         if self.reverse_slides:
    #             slide_deck = np.flip(slide_deck, axis=2)
    #     else:
    #         raise ValueError(f"Unexpected orientation value: {self.orientation}")
    #     return slide_deck

    def apply_registration(self, downsample=1):
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
        matrix = np.diag([*voxdim, 1])
        # Create appropriate Nifti header
        hdr = nib.Nifti1Header()
        hdr.set_xyzt_units(xyz='mm', t='sec')
        hdr.set_sform(matrix, code=2)

        if self.verbose:
            print('\tCreating slide deck image ...', end=' ')
        # slide_deck = self._create_slide_deck(downsample)
        dask.config.set(scheduler='processes', num_workers=NUM_CORES)
        jobs = []
        os.makedirs(self.output_path / 'raw_slices', exist_ok=True)
        os.makedirs(self.output_path / 'resampled_slices', exist_ok=True)

        # re-read input data if mri_reg_mod != psoct_reg_mod
        if self.mri_reg_mod != self.psoct_reg_mod:
            self.inp_path = self.inp_path / '..' / self.mri_reg_mod
            if self.verbose:
                print("")
            self._find_all_slides(self.slide_res=='lowres')
            # self._find_missing_slides() # missing slides should be the same
            self._load_slides()
            self.interpolate_missing_slides()

        def save_image(slides_dict, sl, ref_shape, ref_slide, tmp_matrix, header, abs_mat,
                       slice_thickness, slide_range, output_path):
            img_padded = pad_image(get_image(slides_dict, sl), ref_shape)
            tgt_padded = pad_image(get_image(slides_dict, ref_slide), ref_shape)
            # TODO remove header and re-check
            src_filename = Image(img_padded, xform=tmp_matrix)
            tgt_filename = Image(tgt_padded, xform=tmp_matrix)
            xform = flirtMatrixToSform(abs_mat, srcImage=src_filename, refImage=tgt_filename)
            P = np.array([[1,0,0,0],
                        [0,0,1,0],
                        [0,1,0,0],
                        [0,0,0,1]])
            xform = P @ xform @ P.T
            xform[1,-1] = slice_thickness * (slide_range[-1] - sl)
            # TODO store the raw_filename in a class var to be reusable
            raw_filename = output_path / 'raw_slices' / f'slide_{str(sl).zfill(3)}'
            # TODO add orientation check
            Image(img_padded[:,None,:], xform=xform, header=header).save(raw_filename)
            res_filename = output_path / 'resampled_slices' / f'slide_{str(sl).zfill(3)}'
            # TODO change this to fsl.utils.image.resample.resampleToReference?
            applyxfm(src_filename,
                     tgt_filename,
                     abs_mat,
                     res_filename,
                     twod=True,
                     usesqform=True)

        for sl in self.abs_mat.keys():
            tmp_matrix = np.diag([*[lr_pixel, lr_pixel, slice_thickness], 1])
            temp_hdr = hdr.copy()
            # TODO review hdr vs temp_hdr usage
            hdr.set_sform(tmp_matrix, code=2)
            jobs.append(dask.delayed(save_image)(self.slides_dict, sl, self.ref_shape, self.ref_slide,
                                                 tmp_matrix, temp_hdr, self.abs_mat[sl], slice_thickness,
                                                 self.slide_range, self.output_path))
        dask.compute(jobs)
        slide_deck = []
        for sl in self.abs_mat.keys():
            out_filename = self.output_path / 'resampled_slices' / f'slide_{str(sl).zfill(3)}'
            slide_deck.append(Image(out_filename).data[:,:,0])
        # Stack images into a 3D array
        slide_deck = np.stack(slide_deck, axis=2)
        # Reorient slide deck to make coronal
        if self.orientation == 'sagittal':
            slide_deck = np.transpose(slide_deck, (2, 0, 1)).copy()
            if self.reverse_slides:
                slide_deck = np.flip(slide_deck, axis=0)
        elif self.orientation == 'coronal':
            slide_deck = np.transpose(slide_deck, (0, 2, 1)).copy()
            if self.reverse_slides:
                slide_deck = np.flip(slide_deck, axis=1)
        elif self.orientation == 'axial':
            # order of indices is already correct
            if self.reverse_slides:
                slide_deck = np.flip(slide_deck, axis=2)
        else:
            raise ValueError(f"Unexpected orientation value: {self.orientation}")
        # TODO consider using io.save_nifti instead
        self.slide_deck_img = Image(slide_deck, xform=matrix, header=hdr)
        self.slide_deck = self.output_path / (self.mri_reg_mod[:3] + '_slide_deck')
        self.slide_deck_img.save(self.slide_deck)
        # inp_files = sorted(glob.glob(str(out_filename.parent / 'slide*')), reverse=True)
        # fslmerge('y', self.slide_deck, *inp_files)
        if self.verbose:
            print('Done.')

    def align_mri_to_psoct(self):
        # Register MRI to slide_deck
        # TODO does this need to be an image or could it be a filename?
        if self.mri_ref.is_absolute():
            mri_ref_fullpath = self.mri_ref
        else:
            mri_ref_fullpath = self.inp_path.parent.parent / self.mri_ref
        # mri_img = Image(mri_ref_fullpath)
        matfile = self.output_path / 'MRI_to_PSOCT.mat'
        # TODO this assumes the input has the .nii.gz suffix
        outfile = self.output_path / self.mri_ref.name.replace('.nii.gz', '_in_PSOCT')
        if self.verbose:
            print('\tRunning MRI-to-PSOCT registration ...', end=' ')
        flirt(src=mri_ref_fullpath, ref=self.slide_deck, out=outfile, omat=matfile, dof=12, interp='spline')
        if self.verbose:
            print('Done.')
        return matfile, outfile

    def align_psoct_to_mri(self, matfile, nonlinear=False):
        # invert and save tranformation matrix
        mat      = np.loadtxt(matfile)
        mat_inv  = np.linalg.inv(mat)
        mat_file = self.output_path / 'PSOCT_to_MRI.mat'
        np.savetxt(mat_file, mat_inv, fmt='%.10f', delimiter=' ')

        if self.mri_ref.is_absolute():
            mri_ref_fullpath = self.mri_ref
        else:
            mri_ref_fullpath = self.inp_path.parent.parent / self.mri_ref

        if self.verbose:
            print('\tRunning PSOCT-to-MRI registration ...', end=' ')

        # perform alignment
        outfile = self.output_path / (self.mri_reg_mod[:3] + '_slide_deck_in_MRI')
        if nonlinear:
            field_file = self.output_path / 'PSOCT_to_MRI_warpfield.nii.gz'
            fnirt(src=self.slide_deck_img, ref=mri_ref_fullpath, iout=outfile, fout=field_file, aff=mat_file, config=fnirt_config, verbose=self.verbose)
        else:
            xform = flirtMatrixToSform(mat_inv, srcImage=self.slide_deck_img, refImage=Image(mri_ref_fullpath))
            Image(self.slide_deck_img.data, xform=xform).save(outfile)

        if self.verbose:
            print('Done.')

        return outfile
    
    def invert_warpfield(self):
        if self.verbose:
            print('\tInverting warpfield ...', end=' ')
        invwarp(self.output_path / 'PSOCT_to_MRI_warpfield.nii.gz',
                self.slide_deck,
                self.output_path / 'MRI_to_PSOCT_warpfield.nii.gz',
                verbose=self.verbose,
                noconstraint=True)
        if self.verbose:
            print('Done.')

    # def update_nifti_headers(self, slide_deck):
    #     # Add header information to single slides
    #     # This will be useful for visualising high resolution slides on top of the MRI

    #     # First split the slide deck to get individual sides "correct" header
    #     if self.orientation in OrientationLookup.keys():
    #         indiv_slides = fslsplit(src=slide_deck, out=LOAD, dim=OrientationLookup[self.orientation])
    #     else:
    #         raise ValueError(f"Unexpected orientation value: {self.orientation}")

    #     slide_numbers = sorted(list(self.slides_dict.keys()), reverse=self.reverse_slides)
    #     split_numbers = sorted(list(indiv_slides.keys()))

    #     def save_image(data, header, filename):
    #         Image(data, header=header).save(filename)

    #     dask.config.set(scheduler='processes', num_workers=NUM_CORES)
    #     jobs = []
    #     for sl, idx in zip(slide_numbers, split_numbers):
    #         img = indiv_slides[idx]
    #         # Get the relative path and update the slide number for the interpolated slides
    #         rel_path = self.slides_dict[sl].relative_to(self.inp_path)
    #         fileparts = rel_path.name.split('_')
    #         fileparts[1] = str(sl).zfill(3)
    #         rel_path = rel_path.parent / '_'.join(fileparts)
    #         filename = self.output_path / self.mri_reg_mod / str(rel_path).replace('.nii.gz', '_hdr.nii.gz')
    #         os.makedirs(filename.parent, exist_ok=True)
    #         jobs.append(dask.delayed(save_image)(img.get_fdata(), img.header, filename))
    #         indiv_slides[idx] = filename
    #     if self.verbose:
    #         print('\tUpdating headers for registration slides ...', end=' ')
    #     dask.compute(jobs)
    #     if self.verbose:
    #         print(' Done.')
    #     return indiv_slides

    def apply_to_lowres_images(self, other_images=['Retardance']):
        # Now apply this header to the low res images
        # Note: they need to be zero-padded first and shifted!!

        # convert other_images to a list if not already
        if not isinstance(other_images, list):
            other_images = [other_images]

        if self.slide_res == 'highres':
            if self.verbose:
                print("\tRegistration image is highres. Cannot apply to lowres images. Exiting...")
            return

        # run alignment across all modalities
        def image_proc(file, filename, ref_shape, ref_slide, abs_mat, header, orientation,
                        tmp_matrix, slice_thickness, slide_range, sl,
                        output_path, slide_deck, mri_ref):
            img = Image(file).data[:, :, 0]
            # Zero-pad and shift
            img_padded = pad_image(img, ref_shape)
            tgt_padded = pad_image(Image(ref_slide).data[:, :, 0], ref_shape)
            src_filename = Image(img_padded, xform=tmp_matrix)
            tgt_filename = Image(tgt_padded, xform=tmp_matrix)
            # TODO add if self.slide_res=='lowres'?
            xform = flirtMatrixToSform(abs_mat, srcImage=src_filename, refImage=tgt_filename)
            P = np.array([[1,0,0,0],
                        [0,0,1,0],
                        [0,1,0,0],
                        [0,0,0,1]])
            xform = P @ xform @ P.T
            xform[1,-1] = slice_thickness * (slide_range[-1] - sl)

            if orientation == 'sagittal':
                tgt_img = Image(img_padded[None, :, :], xform=xform, header=header)
            elif orientation == 'coronal':
                tgt_img = Image(img_padded[:, None, :], xform=xform, header=header)
            elif orientation == 'axial':
                tgt_img = Image(img_padded[:, :, None], xform=xform, header=header)
            else:
                raise ValueError(f"Unexpected orientation value: {orientation}")
            
            # readFlirt is just np.loadtxt
            # mat = readFlirt(output_path / 'PSOCT_to_MRI.mat')
            # # convert flirt matrix into a world->world transformation
            # mat = fromFlirt(mat, Image(slide_deck), Image(mri_ref), from_='world', to='world')
            # # concat is just matmul, i.e. '@'
            # xform = affine.concat(mat, tgt_img.getAffine('voxel', 'world'))

            # tgt_img.header.set_sform(matrix, code=2)

            os.makedirs(filename.parent, exist_ok=True)
            Image(tgt_img.data, xform=xform, header=header).save(filename)

        for mod in other_images:

            data_path = self.inp_path.parent / mod / 'lowres'
            data_files = sorted(data_path.glob('Slice_*_En*.nii.gz'))
            # TODO make this more versatile
            mod_slide_numbers = np.array([int(Path(f).name.split('_')[1]) for f in data_files])

            slide_numbers = sorted(list(self.slides_dict.keys()), reverse=self.reverse_slides)

            dask.config.set(scheduler='processes', num_workers=NUM_CORES)
            jobs = []
            if self.verbose:
                print(f"\tApplying registration matrix to lowres '{mod}' slides ...")
            for sl in slide_numbers:
                # skip bad_slides
                if sl in self.bad_slides:
                    continue

                # Load Image
                mod_idx = np.where(mod_slide_numbers == sl)[0]
                if len(mod_idx) != 1:
                    if self.verbose:
                        print(f"\t\tUnexpected number of matching files: file number {sl}. Skipping this file.")
                    continue

                ref_img = Image(self.output_path / 'raw_slices' / f'slide_{str(sl).zfill(3)}')
                file = data_files[mod_idx[0]]
                filename = self.output_path / mod / 'lowres' / str(file.name).replace('.nii.gz', '_hdr.nii.gz')

                orig_pixel = self.seq_params['in-plane resolution']
                lr_pixel = orig_pixel * self.downsample
                slice_thickness = self.seq_params['out-of-plane resolution']
                tmp_matrix = np.diag([*[lr_pixel, lr_pixel, slice_thickness], 1])
                
                jobs.append(dask.delayed(image_proc)(file, filename, self.ref_shape, self.slides_dict[self.ref_slide], self.abs_mat[sl],
                                                     ref_img.header, self.orientation,
                                                     tmp_matrix, slice_thickness, self.slide_range, sl,
                                                     self.output_path, self.slide_deck, self.mri_ref))
            dask.compute(jobs)
            if self.verbose:
                print(f"\tRegistration of '{mod}' slides completed.")

    def apply_to_highres_images(self, other_images=['Retardance']):
        # TODO make this work even if slide_res=='highres'
        # Now apply this header to the high res images

        # convert other_images to a list if not already
        if not isinstance(other_images, list):
            other_images = [other_images]

        # run alignment across all modalities
        def image_proc(file, filename, ref_shape, ref_file, downsample, orientation):
            img = Image(file).data[:, :, 0]
            # TODO add if self.slide_res=='lowres'?
            ref_shape = [x * downsample for x in ref_shape]
            img_padded = pad_image(img, ref_shape)

            ref_img = Image(ref_file)
            if orientation == 'sagittal':
                newShape = [ref_img.shape[0], ref_img.shape[1]*downsample, ref_img.shape[2]*downsample]
            elif orientation == 'coronal':
                newShape = [ref_img.shape[0]*downsample, ref_img.shape[1], ref_img.shape[2]*downsample]
            elif orientation == 'axial':
                newShape = [ref_img.shape[0]*downsample, ref_img.shape[1]*downsample, ref_img.shape[2]]
            else:
                raise ValueError(f"Unexpected orientation value: {orientation}")

            newShape = np.array(np.round(newShape), dtype=int)
            matrix   = affine.rescale(ref_img.shape, newShape, 'centre')
            matrix   = affine.concat(ref_img.voxToWorldMat, matrix)

            if orientation == 'sagittal':
                img_highres = Image(img_padded[None, :, :], xform=matrix, header=ref_img.header)
            elif orientation == 'coronal':
                img_highres = Image(img_padded[:, None, :], xform=matrix, header=ref_img.header)
            elif orientation == 'axial':
                img_highres = Image(img_padded[:, :, None], xform=matrix, header=ref_img.header)
            else:
                raise ValueError(f"Unexpected orientation value: {orientation}")
            os.makedirs(filename.parent, exist_ok=True)
            img_highres.save(filename)

        # run alignment across all modalities
        for mod in other_images:

            data_path     = self.inp_path.parent / mod
            data_files    = sorted(data_path.glob('Slice_*_En*.nii.gz'))
            # TODO add if self.slide_res=='lowres'?
            ls_data_files = sorted((self.output_path / mod / 'lowres').glob('Slice_*_En*.nii.gz'))
            # TODO make this more versatile
            mod_slide_numbers = np.array([int(Path(f).name.split('_')[1]) for f in data_files])
            ls_mod_slide_numbers = np.array([int(Path(f).name.split('_')[1]) for f in ls_data_files])

            slide_numbers = sorted(list(self.slides_dict.keys()), reverse=self.reverse_slides)

            dask.config.set(scheduler='processes', num_workers=NUM_CORES)
            jobs = []
            if self.verbose:
                print(f"\tApplying registration matrix to highres '{mod}' slides ...")
            for sl in slide_numbers:
                # skip bad_slides
                if sl in self.bad_slides:
                    continue

                # Load Image
                mod_idx = np.where(mod_slide_numbers == sl)[0]
                if len(mod_idx) != 1:
                    if self.verbose:
                        print(f"\t\tUnexpected number of matching files: file number {sl}. Skipping this file.")
                    continue

                file = data_files[mod_idx[0]]
                mod_idx = np.where(ls_mod_slide_numbers == sl)[0]
                ref_file = ls_data_files[mod_idx[0]]

                filename = self.output_path / mod / str(file.name).replace('.nii.gz', '_hdr.nii.gz')

                jobs.append(dask.delayed(image_proc)(file, filename, self.ref_shape, ref_file,
                                                     self.downsample, self.orientation))
            dask.compute(jobs)
            if self.verbose:
                print(f"\tRegistration of '{mod}' slides completed.")

    def _save_matrices(self):
        with open(self.output_path / "abs_mat.json", "w") as f:
            json.dump({int(k): v.tolist() for k, v in self.abs_mat.items()}, f)
        with open(self.output_path / "rel_mat.json", "w") as f:
            json.dump({int(k): v.tolist() for k, v in self.rel_mat.items()}, f)
        if self.verbose:
            print('\tRelative and absolute transformation matrices saved.')

    # Function to be called for flirt version of the pipeline
    def run_pipeline(self,
                     other_images,
                     output_path,
                     mri_ref,
                     downsample=1,
                     bad_slides=None,
                     fnirt=False,
                     align_ref='centre',
                     ref_copy=True,
                     inv_warp=False):
        if self.verbose:
            print('\nStarting slide registration process ...')
        self.label_bad_slides(indices=bad_slides)
        self.interpolate_missing_slides()
        self.align(ref=align_ref)
        if self.verbose:
            print('Slide registration completed.')
        if self.verbose:
            print('\nStarting slide deck creation ...')
        # TODO consider moving the next two lines into apply_registration
        self.output_path = Path(output_path)
        os.makedirs(self.output_path, exist_ok=True)
        # save matrices to a text file
        self._save_matrices()
        # save a copy of the mri_ref in the output folder
        self.mri_ref = Path(mri_ref)
        if ref_copy:
            import shutil
            shutil.copyfile(self.mri_ref, self.output_path / self.mri_ref.name)
            self.mri_ref = self.output_path / self.mri_ref.name
        # save slide decks and header information
        self.apply_registration(downsample=downsample)
        matfile, _ = self.align_mri_to_psoct()
        psoct_to_mri_file = self.align_psoct_to_mri(matfile, fnirt)
        if inv_warp and fnirt:
            self.invert_warpfield()
        # indiv_slides = self.update_nifti_headers(psoct_to_mri_file)
        # TODO make sure these work for highres reference too!
        self.apply_to_lowres_images(other_images)
        self.apply_to_highres_images(other_images)
        if self.verbose:
            print(f"\nPSOCT pipeline completed and results saved to {self.output_path}")
