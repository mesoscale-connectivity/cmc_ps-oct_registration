# CMCmultimodal
Codebase to process PSOCT data for the CMC multimodal project. This package performs within-slice registration of PSOCT data and registration to an MRI reference volume. 2D slice and 3D slidedeck images are stored, along with the transformation matrices or warps.

The code provides a CLI wrapper function (see psoct_wrapper.py), as well as, a python library for more flexibility.

## Installation
Clone the repository from Gitlab: <https://git.fmrib.ox.ac.uk/saad/cmcmultimodal.git>

## Usage
### CLI wrapper
Example usage of the CLI wrapper function.
> *Paths and filenames need to be updated to valid local paths.*

```bash
psoct_wrapper --inp_path INP_PATH
              --out_path OUT_PATH
              --seq_params SEQ_PARAMS
              --mri_ref MRI_REF
              --psoct_reg_modality Cross
              --mri_reg_modality Retardance
              --other_images Retardance Orientation
              --slide_range 98 200
              --bad_slides 140
              --nonlinear
              --invwarp
              --verbose
```

## Authors and acknowledgment
The code for this package is developed by: 
- Saad Jbabdi&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<saad.jbabdi@ndcn.ox.ac.uk> 
- Vasilis Karlaftis&nbsp;&nbsp;<vasilis.karlaftis@ndcn.ox.ac.uk>

Copyright (C) 2026 University of Oxford
