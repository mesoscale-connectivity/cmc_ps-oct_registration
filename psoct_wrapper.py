#!/usr/bin/env python
'''
PSOCT CLI wrapper for CMC multimodal analysis

Authors: Saad Jbabdi            <saad.jbabdi@ndcn.ox.ac.uk>
         Vasilis Karlaftis      <vasilis.karlaftis@ndcn.ox.ac.uk>

Copyright (C) 2025 University of Oxford
'''

import argparse
from pathlib import Path
from cmcmultimodal.proc import psoct

def run_psoct_pipeline(
    inp_path,
    out_path,
    mri_ref,
    lowres=True,
    slide_range=None,
    bad_slides=None,
    align_ref='centre',
    align_thr=0,
    plot_alignment=False,
    modality='Retardance',
    orientation='coronal',
    reg_downsample=1):

    # Initialize the object
    ps = psoct(inp_path=Path(inp_path), lowres=lowres, slide_range=slide_range)

    # Run registration
    slides, rel_shifts, abs_shifts = ps.run_registration(
        bad_slides=bad_slides,
        align_ref=align_ref,
        align_thr=align_thr,
        plot_alignment=plot_alignment
    )

    # Create slide deck and apply headers
    indiv_slides = ps.run_slide_deck_creation(
        slides,
        abs_shifts,
        orientation=orientation,
        modality=modality,
        output_path=out_path,
        mri_ref=mri_ref,
        downsample=reg_downsample
    )

    print(f"\nPSOCT pipeline completed and results saved to {out_path}")


def parse_cli_args():
    parser = argparse.ArgumentParser(description="PSOCT registration and slide deck creation pipeline")

    parser.add_argument('--inp_path', type=str, required=True, help="Input path to PSOCT dataset. Select the subject-level folder.")
    parser.add_argument('--out_path', type=str, required=True, help="Output directory for results")
    parser.add_argument('--mri_ref',  type=str, required=True, help="Reference MRI NIfTI file for alignment")
    parser.add_argument('--orientation', type=str, required=True, choices=['coronal', 'axial', 'sagittal'], help='Orientation of PSOCT data acquisition')

    parser.add_argument('--modality', type=str, default='Retardance', choices=['Retardance', 'Cross', 'Orientation'], help="One or more PSOCT modalities to apply the registration to")
    # TODO allow multiple modalities to be entered here
    # parser.add_argument('--modality', type=str, nargs='*', default='Retardance', choices=['Retardance', 'Cross', 'Orientation'], help="One or more PSOCT modalities to apply the registration to")
    parser.add_argument('--highres', action='store_true', help="Use high-resolution data for alignment (default: False)")
    # parser.add_argument('--non-linear', action='store_true', help='Apply non-linear (FNIRT) registration to MRI reference (default: False)')
    parser.add_argument('--slide_range', type=int, nargs=2, default=None, help="Range of slides to process (start end)")
    parser.add_argument('--bad_slides', type=int, nargs='*', default=None, help="List of bad slide numbers to skip")

    parser.add_argument('--align_ref', type=str, default='centre', choices=['centre', 'first', 'last'], help="Reference slide for alignment")
    parser.add_argument('--align_thr', type=float, default=0.0, help="Ignore shifts smaller than this threshold")
    parser.add_argument('--plot_alignment', action='store_true', help="Plot relative and absolute shifts")

    parser.add_argument('--reg_downsample', type=int, default=1, help="Downsample factor for the slide deck")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_cli_args()
    # TODO propagate the non-linear flag when implemented
    run_psoct_pipeline(
        inp_path=args.inp_path,
        out_path=args.out_path,
        mri_ref=args.mri_ref,
        lowres=(not args.highres),
        slide_range=tuple(args.slide_range) if args.slide_range else None,
        bad_slides=args.bad_slides,
        align_ref=args.align_ref,
        align_thr=args.align_thr,
        plot_alignment=args.plot_alignment,
        modality=args.modality,
        orientation=args.orientation,
        reg_downsample=args.reg_downsample
    )
