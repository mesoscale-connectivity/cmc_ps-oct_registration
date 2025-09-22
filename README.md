# CMCmultimodal
Codebase to process PSOCT data for the CMC multimodal project.

The code provides a CLI wrapper function (see psoct_wrapper.py), as well as, a python library for more flexibility.

## Installation
Clone the repository from Gitlab: <https://git.fmrib.ox.ac.uk/saad/cmcmultimodal.git>

## Usage
### CLI wrapper
Example usage of the CLI wrapper function.
> Paths and filenames need to be updated to valid local paths.

```bash
python psoct_wrapper.py --inp_path <your_subject_folder>
                        --out_path <your_results_folder>
                        --seq_params <psoct_seq_params_json>
                        --mri_ref <mri_image_for_registration>
                        --reg_modality Retardance
                        --other_images Retardance Orientation
                        --slide_range 98 200
                        --bad_slides 140
                        --verbose
```

<pre> 
Compulsory arguments:
  -in, --inp_path INP_PATH
                        Input path to PSOCT dataset
  -out, --out_path OUT_PATH
                        Output directory for results
  --seq_params SEQ_PARAMS
                        Path to PSOCT sequence parameters' JSON file
  --mri_ref MRI_REF     Reference MRI NIfTI file for alignment
  --reg_modality {Retardance,Cross,Orientation}
                        The PSOCT modality to be used for alignment
  --other_images [{Retardance,Cross,Orientation} ...]
                        One or more PSOCT modalities to apply the registration to

Optional arguments:
  --highres             Use high-resolution data for alignment (default: False)
  --slide_range SLIDE_RANGE SLIDE_RANGE
                        Range of slides to process (start end)
  --bad_slides [BAD_SLIDES ...]
                        List of bad slide numbers to skip
  --align_ref {centre,first,last}
                        Reference slide for alignment
  --align_thr ALIGN_THR
                        Ignore shifts smaller than this threshold
  --reg_downsample REG_DOWNSAMPLE
                        Downsample factor for the slide deck
  -v, --verbose         Print diagnostic information while running
  -h, --help            show this help message and exit
</pre>

## Authors and acknowledgment
(tbd)
