
import numpy as np
from pathlib import Path
from cmcmultimodal.utils import get_image, calc_shift


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
        self.interpolated_slides = []

        self._find_all_slides(lowres=lowres)
        self._find_missing_slides()
        self._load_slides()

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

    def ignore_slides(self):
        # A list of bad and missing slides
        self.interpolated_slides = np.sort(np.unique(self.missing_slides+self.bad_slides)).tolist()
        return self.interpolated_slides

    def interpolate_missing_slides(self):
        slide_arr = np.array(self.slide_numbers)
        for m in self.ignore_slides():
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

    def find_central_slide(self):
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
    
    def align(self, ref='centre', thr=0, verbose=False):
        '''
        Parameters:
        - ref: reference mode for alignment ('centre' for using the central slide)
        - thr: shift threshold. Any shifts lower than this are ignored (to minimize drifts)
        '''
        # This is the main loop that calculates the shifts between each slide
        # and its neighbour. If it is before central slide, look at neighbour in front
        # otherwise look at neighbour behind
        if ref == 'centre':
            ref_slide, ref_shape = self.find_central_slide()
        elif ref == 'first':
            ref_slide = np.min(list(self.slides_dict.keys()))
            ref_shape = get_image(self.slides_dict, ref_slide).shape
        elif ref == 'last':
            ref_slide = np.max(list(self.slides_dict.keys()))
            ref_shape = get_image(self.slides_dict, ref_slide).shape
        else:
            raise ValueError(f'Unexpected reference method {ref} for alignment.')
        # Use all slides for alignment (including interpolated ones)
        all_slides = np.sort(list(self.slides_dict.keys()))
        all_shifts = np.zeros((len(all_slides), 2))
        # for slide in tqdm(all_slides):
        for slide in range(len(all_slides)):
        # Get image from dataframe
            img = get_image(self.slides_dict, all_slides[slide]) 
            if all_slides[slide] == ref_slide:
                t = [0, 0] # no shift if it is central slide
            else:
                if all_slides[slide] < ref_slide:
                    tgt = get_image(self.slides_dict, all_slides[slide]+1)
                else:
                    tgt = get_image(self.slides_dict, all_slides[slide]-1)
                t  = calc_shift(img, tgt, ref_shape)
                # don't worry about small shifts
                t[0] = t[0] if np.abs(t[0])>thr else 0.
                t[1] = t[1] if np.abs(t[1])>thr else 0.
            # Store shifts
            all_shifts[slide] = t
            if verbose:
                print(all_slides[slide], all_shifts[slide])

    def run_registration(self, bad_slides=None):
        self.label_bad_slides(indices=bad_slides)
        self.interpolate_missing_slides()
        self.align(verbose=True)

