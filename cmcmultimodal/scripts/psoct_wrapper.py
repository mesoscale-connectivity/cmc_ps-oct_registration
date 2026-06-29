#!/usr/bin/env python
'''
PSOCT CLI wrapper for CMC multimodal analysis

Authors: Saad Jbabdi            <saad.jbabdi@ndcn.ox.ac.uk>
         Vasilis Karlaftis      <vasilis.karlaftis@ndcn.ox.ac.uk>

Copyright (C) 2025 University of Oxford
'''

import argparse
import sys
import time
from datetime import datetime
import textwrap


def run_psoct_pipeline(
        inp_path,
        out_path,
        seq_params,
        mri_ref,
        lowres=True,
        slide_range=None,
        bad_slides=None,
        fnirt=False,
        invwarp=False,
        align_ref='centre',
        psoct_reg_mod='Cross',
        mri_reg_mod='Retardance',
        other_images=['Retardance', 'Cross'],
        verbose=False):

    from cmcmultimodal.core.proc import psoct

    # Initialize the object
    ps = psoct(inp_path=inp_path,
               seq_params=seq_params,
               lowres=lowres,
               slide_range=slide_range,
               psoct_reg_mod=psoct_reg_mod,
               mri_reg_mod=mri_reg_mod,
               verbose=verbose)

    # Run pipeline
    ps.run_pipeline(other_images=other_images,
                    output_path=out_path,
                    mri_ref=mri_ref,
                    bad_slides=bad_slides,
                    fnirt=fnirt,
                    align_ref=align_ref,
                    invwarp=invwarp)


def parse_cli_args():
    parser = argparse.ArgumentParser(
                description=textwrap.dedent("""PSOCT registration and slide deck creation pipeline"""),
                formatter_class=argparse.RawTextHelpFormatter)

    # ---- Required arguments group ----
    required = parser.add_argument_group("Compulsory arguments")
    required.add_argument('-in',  '--inp_path',   type=str, required=True,
                          help="Input path to PSOCT dataset")
    required.add_argument('-out', '--out_path',   type=str, required=True,
                          help="Output directory for results")
    required.add_argument('--seq_params',         type=str, required=True,
                          help="Path to PSOCT sequence parameters' JSON file")
    required.add_argument('--mri_ref',            type=str, required=True,
                          help="Reference MRI NIfTI file for alignment")
    required.add_argument('--psoct_reg_modality', type=str, required=True,
                          help="The PSOCT modality to be used for between-slide alignment")
    required.add_argument('--mri_reg_modality',   type=str, required=True,
                          help="The PSOCT modality to be used for alignment to MRI")
    required.add_argument('--other_images',       type=str, required=True, nargs='*',
                          choices=['Retardance', 'Cross', 'Orientation', 'Reflectivity'],
                          help="One or more PSOCT modalities to apply the registration to")

    # ---- Optional arguments group ----
    optional = parser.add_argument_group("Optional arguments")
    optional.add_argument('--highres', action='store_true',
                          help="Use high-resolution data for alignment (default: False)")
    optional.add_argument('--nonlinear', action='store_true',
                          help='Apply non-linear (FNIRT) registration to MRI reference (default: False)')
    optional.add_argument('--invwarp',    action='store_true',
                          help='For non-linear registration, invert the warp field (default: False)')
    optional.add_argument('--slide_range', type=int, nargs=2, default=None,
                          metavar=('START', 'END'), help="Range of slides to process (start end)")
    optional.add_argument('--bad_slides',  type=int, nargs='*', default=None,
                          metavar='SLIDE_NO', help="List of bad slide numbers to skip")

    optional.add_argument('--align_ref', type=str, default='centre',
                          choices=['centre', 'first', 'last'], help="Reference slide for alignment")
    optional.add_argument('-v', '--verbose',  action='store_true',
                          help="Print diagnostic information while running")

    return parser.parse_args()


def main():
    start_time = time.time()
    start_dt = datetime.now()
    # read argument and execute code
    args = parse_cli_args()
    run_psoct_pipeline(
        inp_path        = args.inp_path,
        out_path        = args.out_path,
        seq_params      = args.seq_params,
        mri_ref         = args.mri_ref,
        lowres          = (not args.highres),
        slide_range     = tuple(args.slide_range) if args.slide_range else None,
        bad_slides      = args.bad_slides,
        fnirt           = args.nonlinear,
        invwarp         = args.invwarp,
        align_ref       = args.align_ref,
        psoct_reg_mod   = args.psoct_reg_modality,
        mri_reg_mod     = args.mri_reg_modality,
        other_images    = args.other_images,
        verbose         = args.verbose
    )
    # store command line in txt file
    end_time = time.time()
    cmd = " ".join(sys.argv)
    with open(args.out_path + "/command_log.txt", "a") as f:
        f.write(f"[{start_dt.isoformat()}] {cmd}\n"
                f"→ Duration: {(end_time-start_time):.2f}s\n")


if __name__ == "__main__":
    main()
