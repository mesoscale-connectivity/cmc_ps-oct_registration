
# from glob import glob
# import os
import numpy as np
from pathlib import Path
# from cmcmultimodal import io
from fsl.data.image import Image


class psoct:

    def __init__(self, inp_path, slide_range=None, lowres=True):
        self.inp_path = Path(inp_path)
        self.image_files = None
        self._slide_range = None
        self.slide_range = slide_range
        self.slide_numbers = None
        self.missing_slides = []
        self.bad_slides = []
        self.slides_dict = {}

        self.__find_all_slides(lowres=lowres)

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
        
    def __find_all_slides(self, lowres=False):
        # TODO this should get the 'lowres' folder and the filenames from the io.py functions
        if lowres:
            # TODO do we need sorted here?
            self.image_files   = sorted(self.inp_path.glob('lowres/' + 'Slice_*_En*.nii.gz'))
        else:
            self.image_files   = sorted(self.inp_path.glob('Slice_*_En*.nii.gz'))
        # TODO: the specificity of the file format is interlinked with the io.py
        self.slide_numbers = [int(Path(f).name.split('_')[1]) for f in self.image_files]

    def load_slides(self):
        if self.slide_range is not None:
            # TODO optimise the performance by looping through the shortest range
            # i.e. slide_range[1]-slide_range[0] vs slide_numbers[-1]-slide_numbers[0]
            # slides_dict contains file names, not data
            for s, f in zip(self.slide_numbers, self.image_files):
                # slide_range is inclusive
                if (s>=self.slide_range[0])&(s<=self.slide_range[1]):
                    self.slides_dict[s] = f
        else:
            # TODO check with Saad
            self.slides_dict = self.image_files

    def find_missing_slides(self):
        # Get list of missing slides
        self.missing_slides = list(set(np.arange(self.slide_range[0], self.slide_range[1]+1)) - set(self.slide_numbers))
        self.missing_slides = np.sort(np.unique(self.missing_slides+self.bad_slides)).tolist()
        return self.missing_slides

    def label_bad_slides(self, indices=None):
        ''' List of bad slides as defined by visual assessment.'''
        if indices is not None:
            self.bad_slides = list(indices)

    def interpolate_missing_slides(self):
        slide_arr = np.array(self.slide_numbers)
        for m in self.missing_slides:
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

    def run(self, bad_slides=None):
        self.load_slides()
        self.label_bad_slides(indices=bad_slides)
        self.find_missing_slides()
        self.interpolate_missing_slides()
