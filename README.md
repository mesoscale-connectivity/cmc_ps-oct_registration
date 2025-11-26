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
psoct_wrapper --inp_path INP_PATH
              --out_path OUT_PATH
              --seq_params SEQ_PARAMS
              --mri_ref MRI_REF
              --reg_modality Retardance
              --reg_method 'flirt'
              --other_images Retardance Orientation
              --slide_range 98 200
              --bad_slides 140
              --non-linear
              --verbose
```

## Authors and acknowledgment
(tbd)
