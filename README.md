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
                        --reg_method 'flirt'
                        --other_images Retardance Orientation
                        --slide_range 98 200
                        --bad_slides 140
                        --verbose
```

## Authors and acknowledgment
(tbd)
