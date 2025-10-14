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
from pathlib import Path
from cmcmultimodal.proc import psoct

def run_psoct_pipeline(
    inp_path,
    out_path,
    seq_params,
    mri_ref,
    lowres=True,
    slide_range=None,
    bad_slides=None,
    reg_method='flirt',
    fnirt=False,
    align_ref='centre',
    align_thr=0,
    plot_alignment=False,
    reg_modality='Retardance',
    reg_downsample=1,
    other_images=['Retardance', 'Cross'],
    verbose=False):

    # Initialize the object
    ps = psoct(inp_path=Path(inp_path), seq_params=seq_params, lowres=lowres, slide_range=slide_range, reg_modality=reg_modality, verbose=verbose)

    # Run pipeline
    _ = ps.run_pipeline(other_images=other_images,
                        output_path=out_path,
                        mri_ref=mri_ref,
                        downsample=reg_downsample,
                        bad_slides=bad_slides,
                        reg_method=reg_method,
                        fnirt=fnirt,
                        align_ref=align_ref,
                        align_thr=align_thr,
                        plot_alignment=plot_alignment
    )

def parse_cli_args():
    parser = argparse.ArgumentParser(description=textwrap.dedent("""
                                                                 PSOCT registration and slide deck creation pipeline

                                                                 Usage:
                                                                 python psoct_wrapper.py --inp_path <your_subject_folder> --out_path <your_results_folder> --seq_params <psoct_seq_params_json> --mri_ref <mri_image_for_registration> --reg_modality Retardance --other_images Retardance Orientation
                                                                 python psoct_wrapper.py --inp_path <your_subject_folder> --out_path <your_results_folder> --seq_params <psoct_seq_params_json> --mri_ref <mri_image_for_registration> --reg_modality Retardance --other_images Retardance Orientation --slide_range 98 200 --bad_slides 140 --verbose
                                                                 """),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     usage=argparse.SUPPRESS)

    # ---- Required arguments group ----
    required = parser.add_argument_group("Compulsory arguments")
    required.add_argument('-in',  '--inp_path', type=str, required=True, help="Input path to PSOCT dataset")
    required.add_argument('-out', '--out_path', type=str, required=True, help="Output directory for results")
    required.add_argument('--seq_params',       type=str, required=True, help="Path to PSOCT sequence parameters' JSON file")
    required.add_argument('--mri_ref',          type=str, required=True, help="Reference MRI NIfTI file for alignment")
    required.add_argument('--reg_modality',     type=str, required=True, choices=['Retardance', 'Cross', 'Orientation'], help="The PSOCT modality to be used for alignment")
    required.add_argument('--other_images',     type=str, required=True, nargs='*', choices=['Retardance', 'Cross', 'Orientation'], help="One or more PSOCT modalities to apply the registration to")

    # ---- Optional arguments group ----
    optional = parser.add_argument_group("Optional arguments")
    optional.add_argument('--highres', action='store_true', help="Use high-resolution data for alignment (default: False)")
    required.add_argument('--reg_method',  type=str, default='flirt', choices=['flirt', 'cc'], help="The registration method for within-slide alignment")
    optional.add_argument('--non_linear', action='store_true', help='Apply non-linear (FNIRT) registration to MRI reference (default: False)')
    optional.add_argument('--slide_range', type=int, nargs=2, default=None, help="Range of slides to process (start end)")
    optional.add_argument('--bad_slides',  type=int, nargs='*', default=None, help="List of bad slide numbers to skip")

    optional.add_argument('--align_ref', type=str, default='centre', choices=['centre', 'first', 'last'], help="Reference slide for alignment")
    optional.add_argument('--align_thr', type=float, default=0.0,  help="Ignore shifts smaller than this threshold")
    optional.add_argument('--reg_downsample', type=int, default=1, help="Downsample factor for the slide deck")
    # TODO add plot save function for this to be sensible option
    # optional.add_argument('--plot_alignment', action='store_true', help="Plot relative and absolute shifts (default: False)")
    optional.add_argument('-v', '--verbose',  action='store_true', help="Print diagnostic information while running")

    return parser.parse_args()


if __name__ == '__main__':
    start_time = time.time()
    start_dt = datetime.now()
    # read argument and execute code
    args = parse_cli_args()
    # TODO propagate the non-linear flag when implemented
    run_psoct_pipeline(
        inp_path=args.inp_path,
        out_path=args.out_path,
        seq_params=args.seq_params,
        mri_ref=args.mri_ref,
        lowres=(not args.highres),
        slide_range=tuple(args.slide_range) if args.slide_range else None,
        bad_slides=args.bad_slides,
        reg_method=args.reg_method,
        fnirt=args.non_linear,
        align_ref=args.align_ref,
        align_thr=args.align_thr,
        # plot_alignment=args.plot_alignment,
        reg_modality=args.reg_modality,
        reg_downsample=args.reg_downsample,
        other_images = args.other_images,
        verbose=args.verbose
    )
    # store command line in txt file
    end_time = time.time()
    cmd = " ".join(sys.argv)
    with open(args.out_path + "/command_log.txt", "a") as f:
        f.write(f"[{start_dt.isoformat()}] {cmd}\n"
                f"→ Duration: {(end_time-start_time):.2f}s\n"
        )
