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
import shutil

from cmcmultimodal.core.utils    import check_seq_params, get_image, calc_flirt
from fsl.data.image              import Image
from fsl.wrappers                import flirt, fnirt, applyxfm, invwarp
from fsl.transform.flirt         import flirtMatrixToSform
from fsl.wrappers.fnirt          import applywarp
import fsl.transform.affine as affine
import dask.multiprocessing
import multiprocessing

# set cores for dask parallel processing
NUM_CORES = min(8, multiprocessing.cpu_count() - 1)

# create sentinel object for slide_range
_UNSET = object()

# Lookup table for orientation information
ORIENTATION_LOOKUP = {'sagittal': 'x', 'coronal': 'y', 'axial': 'z'}
# FSL convention for orientation
FSLCONVENTION      = {'sagittal': 'LR', 'coronal': 'PA', 'axial': 'IS'}

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
        self._find_all_slides(self.inp_path, lowres=lowres)
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

    def _read_seq_params(self, seq_params):
        # check if JSON file is of a valid format
        self.seq_params = check_seq_params(seq_params)
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
        if slice_order != FSLCONVENTION[self.orientation]:
            self.reverse_slides = True

    def _find_all_slides(self, inp_path, lowres=False):
        # TODO this should get the 'lowres' folder and the filenames from the io.py functions
        if lowres:
            self.image_files = sorted(inp_path.glob('lowres/' + 'Slice_*_En*.nii.gz'))
            self.slide_res = 'lowres'
            self.downsample = 10
        else:
            self.image_files = sorted(inp_path.glob('Slice_*_En*.nii.gz'))
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
        for i, slide in enumerate(all_slides):
            all_sizes[i] = np.count_nonzero(get_image(self.slides_dict, slide)[0])
        # Find all slides that have max size and take the median as the central slide
        max_indices = np.where(all_sizes == np.max(all_sizes))[0]
        central_slide_num = all_slides[round(np.median(max_indices))]
        return central_slide_num

    def _get_ref_slide(self, ref):
        if ref == 'centre':
            ref_slide = self._find_central_slide()
        elif ref == 'first':
            ref_slide = np.min(list(self.slides_dict.keys()))
        elif ref == 'last':
            ref_slide = np.max(list(self.slides_dict.keys()))
        else:
            raise ValueError(f'Unexpected reference method {ref} for alignment.')
        return ref_slide

    def align(self, ref='centre'):
        ''' This method calculates the registration matrices between each slide and its neighbour.
        If the slide is before the central slide, it looks at the neighbour in front,
        otherwise look at the neighbour behind

        Parameters:
        - ref: reference mode for alignment ('centre' for using the central slide)
        '''
        self.ref_slide = self._get_ref_slide(ref)
        if self.verbose:
            print(f"\tReference slide for alignment: {self.ref_slide}")
        # Use all slides for alignment (including interpolated ones)
        slides = sorted(list(self.slides_dict.keys()))
        dask.config.set(scheduler='processes', num_workers=NUM_CORES)
        jobs = []
        for sl in slides:
            # Get image from dataframe
            if sl == self.ref_slide:
                jobs.append(np.eye(4))
            else:
                img = self.slides_dict[sl]
                if sl < self.ref_slide:
                    tgt = self.slides_dict[sl+1]
                else:
                    tgt = self.slides_dict[sl-1]
                if tgt == img:
                    jobs.append(np.eye(4))
                else:
                    # cost was 'leastsq' or 'normcorr' for Retardance reference
                    jobs.append(dask.delayed(calc_flirt)(img, tgt, cost='corratio'))
        tmp_results = dask.compute(jobs)[0]
        assert len(slides) == len(tmp_results)
        self.rel_mat = dict(zip(slides, tmp_results))
        # Calculate absolute transformation matrices
        self._calc_total_mat()
        # save matrices to a text file
        self._save_matrices()

    def _calc_total_mat(self):
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

    def apply_registration(self):
        if self.verbose:
            print('\tCreating slide deck image ...', end=' ')
        dask.config.set(scheduler='processes', num_workers=NUM_CORES)
        jobs = []
        # create new 'raw_slices' and 'resampled_slices' folders if already exist
        for fd in [self.output_path / 'raw_slices', self.output_path / 'resampled_slices']:
            if fd.exists():
                shutil.rmtree(fd)
            os.makedirs(fd)

        # re-read input data if mri_reg_mod != psoct_reg_mod
        if self.mri_reg_mod != self.psoct_reg_mod:
            self.inp_path = self.inp_path / '..' / self.mri_reg_mod
            if self.verbose:
                print("")
            self.slides_dict = {}
            self._find_all_slides(self.inp_path, self.slide_res=='lowres')
            self._find_missing_slides() # missing slides should be the same
            self._load_slides()
            self.interpolate_missing_slides()

        def save_image(slides_dict, sl, ref_slide, abs_mat,
                       orientation, slide_range, output_path):
            src_filename = Image(slides_dict[sl])
            tgt_filename = Image(slides_dict[ref_slide])
            xform = flirtMatrixToSform(abs_mat, srcImage=src_filename, refImage=tgt_filename)
            if orientation == 'sagittal':
                P = np.array([[0,0,1,0],
                            [1,0,0,0],
                            [0,1,0,0],
                            [0,0,0,1]])
            elif orientation == 'coronal':
                P = np.array([[1,0,0,0],
                            [0,0,1,0],
                            [0,1,0,0],
                            [0,0,0,1]])
            elif orientation == 'axial':
                P = np.eye(4)
            xform = P @ xform @ P.T
            if orientation == 'sagittal':
                dim = 0
            elif orientation == 'coronal':
                dim = 1
            elif orientation == 'axial':
                dim = 2
            if self.reverse_slides:
                xform[dim,-1] = src_filename.pixdim[2] * (slide_range[-1] - sl)
            else:
                xform[dim,-1] = src_filename.pixdim[2] * (sl - slide_range[0])
            # update header with the updated xform
            hdr = src_filename.header
            hdr.set_sform(xform, code=2)
            raw_filename = output_path / 'raw_slices' / f'slide_{str(sl).zfill(3)}'
            data = Image(src_filename).data[...,0]
            if orientation == 'sagittal':
                Image(data[None,:,:], xform=xform, header=hdr).save(raw_filename)
            elif self.orientation == 'coronal':
                Image(data[:,None,:], xform=xform, header=hdr).save(raw_filename)
            elif self.orientation == 'axial':
                Image(data[:,:,None], xform=xform, header=hdr).save(raw_filename)
            else:
                raise ValueError(f"Unexpected orientation value: {orientation}")
            # register and resample each slice for creating the slide deck
            res_filename = output_path / 'resampled_slices' / f'slide_{str(sl).zfill(3)}'
            applyxfm(src_filename,
                     tgt_filename,
                     abs_mat,
                     res_filename,
                     twod=True,
                     usesqform=True)

        for sl in self.abs_mat.keys():
            jobs.append(dask.delayed(save_image)(self.slides_dict, sl, self.ref_slide,
                                                 self.abs_mat[sl], self.orientation,
                                                 self.slide_range, self.output_path))
        dask.compute(jobs)
        slide_deck = []
        for sl in self.abs_mat.keys():
            out_filename = self.output_path / 'resampled_slices' / f'slide_{str(sl).zfill(3)}'
            slide_deck.append(Image(out_filename).data[...,0])
        # Stack images into a 3D array
        slide_deck = np.stack(slide_deck, axis=2)
        # Reorient slide deck to make coronal
        hdr = Image(out_filename).header
        voxdim = Image(out_filename).pixdim
        if self.orientation == 'sagittal':
            slide_deck = np.transpose(slide_deck, (2, 0, 1)).copy()
            voxdim = (voxdim[2], voxdim[0], voxdim[1])
            if self.reverse_slides:
                slide_deck = np.flip(slide_deck, axis=0)
        elif self.orientation == 'coronal':
            slide_deck = np.transpose(slide_deck, (0, 2, 1)).copy()
            voxdim = (voxdim[0], voxdim[2], voxdim[1])
            if self.reverse_slides:
                slide_deck = np.flip(slide_deck, axis=1)
        elif self.orientation == 'axial':
            # order of indices is already correct
            if self.reverse_slides:
                slide_deck = np.flip(slide_deck, axis=2)
        else:
            raise ValueError(f"Unexpected orientation value: {self.orientation}")
        xform = np.diag([*voxdim, 1])
        hdr.set_sform(xform, code=2)
        self.slide_deck_img = Image(slide_deck, xform=xform, header=hdr)
        self.slide_deck = self.output_path / (self.mri_reg_mod + '_slide_deck')
        self.slide_deck_img.save(self.slide_deck)
        # save slice position-to-number mapping
        self._save_slice_mapping()
        if self.verbose:
            print('Done.')

    def align_mri_to_psoct(self):
        # Register MRI to slide_deck
        # TODO does this need to be an image or could it be a filename?
        if self.mri_ref.is_absolute():
            mri_ref_fullpath = self.mri_ref
        else:
            mri_ref_fullpath = self.inp_path.parent.parent / self.mri_ref
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
        outfile = self.output_path / (self.mri_reg_mod + '_slide_deck_in_MRI')
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

    def apply_to_lowres_images(self, other_images=['Retardance']):
        # Now apply this header to the low res images
        # Note: they need to be zero-padded first and shifted!!

        # TODO move this check in run_pipeline?
        if self.slide_res == 'highres':
            if self.verbose:
                print("\tRegistration image is highres. Cannot apply to lowres images. Exiting...")
            return

        # run alignment across all modalities
        def image_proc(file, filename, ref_slide, abs_mat, 
                       orientation, reverse_slides, slide_range, sl):
            src_filename = Image(file)
            tgt_filename = Image(ref_slide)
            # TODO add if self.slide_res=='lowres'?
            xform = flirtMatrixToSform(abs_mat, srcImage=src_filename, refImage=tgt_filename)
            if orientation == 'sagittal':
                P = np.array([[0,0,1,0],
                            [1,0,0,0],
                            [0,1,0,0],
                            [0,0,0,1]])
            elif orientation == 'coronal':
                P = np.array([[1,0,0,0],
                            [0,0,1,0],
                            [0,1,0,0],
                            [0,0,0,1]])
            elif orientation == 'axial':
                P = np.eye(4)
            xform = P @ xform @ P.T
            if orientation == 'sagittal':
                dim = 0
            elif orientation == 'coronal':
                dim = 1
            elif orientation == 'axial':
                dim = 2
            if reverse_slides:
                xform[dim,-1] = src_filename.pixdim[2] * (slide_range[-1] - sl)
            else:
                xform[dim,-1] = src_filename.pixdim[2] * (sl - slide_range[0])
            # update header with the updated xform
            hdr = src_filename.header
            hdr.set_sform(xform, code=2)
            
            os.makedirs(filename.parent, exist_ok=True)
            if orientation == 'sagittal':
                Image(src_filename.data[None, :, :], xform=xform, header=hdr).save(filename)
            elif orientation == 'coronal':
                Image(src_filename.data[:, None, :], xform=xform, header=hdr).save(filename)
            elif orientation == 'axial':
                Image(src_filename.data[:, :, None], xform=xform, header=hdr).save(filename)
            else:
                raise ValueError(f"Unexpected orientation value: {orientation}")

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

                file = data_files[mod_idx[0]]
                filename = self.output_path / mod / 'lowres' / str(file.name).replace('.nii.gz', '_hdr.nii.gz')

                jobs.append(dask.delayed(image_proc)(file, filename, self.slides_dict[self.ref_slide], self.abs_mat[sl],
                                                     self.orientation, self.reverse_slides, self.slide_range, sl))
            dask.compute(jobs)
            if self.verbose:
                print(f"\tRegistration of '{mod}' slides completed.")

    def create_lowres_slidedeck(self, other_images=['Retardance'], nonlinear=False):
        import subprocess
        from fsl.wrappers.avwutils  import fslmerge

        # split reg slide deck for creating reference volumes for resampling
        dim = ORIENTATION_LOOKUP[self.orientation]
        split_folder = self.output_path / 'Ref_deck_split'
        os.makedirs(split_folder, exist_ok=True)
        subprocess.run(['fslsplit',
                        self.slide_deck,
                        split_folder / 'vol',
                        '-' + dim])
        ref_files = sorted(split_folder.glob('*.nii.gz'), reverse=True)

        # create slidedeck in PSOCT space
        for mod in other_images:
            self.inp_path = self.inp_path / '..' / mod
            out_file = self.output_path / (mod[0:3] + '_slide_deck')

            if self.verbose:
                print("")
                print(f"Creating slide deck for {mod}...")

            self.slides_dict = {}
            self._find_all_slides(self.output_path / mod, self.slide_res=='lowres')
            self._find_missing_slides()
            self._load_slides()
            self.interpolate_missing_slides()

            # reorder slides and extract the filenames
            inp_files = [v for _, v in sorted(self.slides_dict.items())]

            os.makedirs(out_file.parent / (mod[0:3] +'_temp_files'), exist_ok=True)
            jobs = []
            # TODO need to determine when self.reverse_slides value matters
            for idx, (inp, ref) in enumerate(zip(inp_files, ref_files), 1):
                filename = 'vol_' + str(idx).zfill(3) + '.nii.gz'
                out_filename = out_file.parent / (mod[0:3] + '_temp_files') / filename
                # if slide is interpolated, then the target should match the original slide
                if idx in self.missing_slides:
                    orig_idx = int(Path(inp).name.split('_')[1]) - 1
                    ref = ref_files[orig_idx]
                jobs.append(dask.delayed(flirt)(inp, ref, out=out_filename, usesqform=True, applyxfm=True, twod=True))
            dask.compute(jobs)
            out_folder = out_file.parent / (mod[0:3] + '_temp_files')
            out_files = sorted(out_folder.glob('*.nii.gz'), reverse=True)
            fslmerge(ORIENTATION_LOOKUP[self.orientation], out_file, *out_files)
            shutil.rmtree(out_folder)
        
        shutil.rmtree(split_folder)

        # create slidedeck in MRI space
        for mod in other_images:
            inp_file = self.output_path / (mod[0:3] + '_slide_deck')
            out_file = self.output_path / (mod[0:3] + '_slide_deck_in_MRI')
            if nonlinear:
                warp = self.output_path / 'PSOCT_to_MRI_warpfield.nii.gz'
                applywarp(inp_file, self.mri_ref, out_file, warp=warp)
            else:
                mat = self.output_path / 'PSOCT_to_MRI.mat'
                applyxfm(inp_file, self.mri_ref, mat, out_file)

        if self.verbose:
            print("Slide decks successfully created.")

    def apply_to_highres_images(self, other_images=['Retardance']):
        # TODO make this work even if slide_res=='highres'
        # Now apply this header to the high res images

        # run alignment across all modalities
        def image_proc(file, filename, ref_file, downsample, orientation):
            src_img = Image(file)
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

            # update header with the updated xform
            hdr = ref_img.header
            hdr.set_sform(matrix, code=2)

            os.makedirs(filename.parent, exist_ok=True)
            if orientation == 'sagittal':
                Image(src_img.data[None, :, :], xform=matrix, header=hdr).save(filename)
            elif orientation == 'coronal':
                Image(src_img.data[:, None, :], xform=matrix, header=hdr).save(filename)
            elif orientation == 'axial':
                Image(src_img.data[:, :, None], xform=matrix, header=hdr).save(filename)
            else:
                raise ValueError(f"Unexpected orientation value: {orientation}")

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

                jobs.append(dask.delayed(image_proc)(file, filename, ref_file,
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
    
    def _save_slice_mapping(self):
        if not self.slides_dict:
            return {}
        # create new dict
        new_dict = {}
        for k, v in self.slides_dict.items():
            # keep only first two segments of filename + _hdr extension
            val = "_".join(v.name.split('_')[0:2]) + '_*_hdr.nii.gz'
            # change dict keys to have the smallest value equal to 0
            new_dict[k - min(self.slides_dict.keys())] = val
        # reverse key order if slides are opposite to FSL convention
        if self.reverse_slides:
            new_dict = {max(new_dict.keys()) - k: v for k, v in new_dict.items()}
        # reorder keys to be in increasing order
        new_dict = dict(sorted(new_dict.items()))
        # save new dict in file
        with open(self.output_path / "slidedeck_slice_mapping.json", "w") as f:
            json.dump({k: v for k, v in new_dict.items()}, f, indent=2)
        if self.verbose:
            print('\tSlice position-to-number mapping saved.')

    # Function to be called for flirt version of the pipeline
    def run_pipeline(self,
                     other_images,
                     output_path,
                     mri_ref,
                     bad_slides=None,
                     fnirt=False,
                     align_ref='centre',
                     ref_copy=True,
                     inv_warp=False):
        if self.verbose:
            print('\nStarting slide registration process ...')
        self.label_bad_slides(indices=bad_slides)
        self.interpolate_missing_slides()
        self.output_path = Path(output_path)
        self.mri_ref = Path(mri_ref)
        os.makedirs(self.output_path, exist_ok=True)
        self.align(ref=align_ref)
        if self.verbose:
            print('Slide registration completed.')
        if self.verbose:
            print('\nStarting slide deck creation ...')
        # save slide decks and header information
        self.apply_registration()
        # save a copy of the mri_ref in the output folder
        if ref_copy:
            shutil.copyfile(self.mri_ref, self.output_path / self.mri_ref.name)
            self.mri_ref = self.output_path / self.mri_ref.name
        matfile, _ = self.align_mri_to_psoct()
        _ = self.align_psoct_to_mri(matfile, fnirt)
        if inv_warp and fnirt:
            self.invert_warpfield()
        # TODO make sure these work for highres reference too!
        # convert other_images to a list if not already
        if not isinstance(other_images, list):
            other_images = [other_images]
        self.apply_to_lowres_images(other_images)
        self.create_lowres_slidedeck(other_images, fnirt)
        self.apply_to_highres_images(other_images)
        if self.verbose:
            print(f"\nPSOCT pipeline completed and results saved to {self.output_path}")
