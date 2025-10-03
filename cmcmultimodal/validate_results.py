#!/usr/bin/env python
'''
PSOCT results validator (to be used to compare newer to older versions of the pipeline)

Authors: Vasilis Karlaftis      <vasilis.karlaftis@ndcn.ox.ac.uk>

Copyright (C) 2025 University of Oxford
'''

from fsl.data.image import Image
import numpy as np
import json


def _compare_images(ref, est):
    ref_img = Image(ref)
    est_img = Image(est)

    if not np.allclose(ref_img.data, est_img.data):
        print(ref)
        print('\t', 'Images are NOT equal!')
    if ref_img.header != est_img.header:
        print(ref)
        print('\t', 'Headers are NOT equal!')

def _compare_matrices(ref, est):
    ref_mat = np.loadtxt(ref)
    est_mat = np.loadtxt(est)

    if not np.allclose(ref_mat, est_mat, atol=0.001):
        print(ref)
        print('\t', 'Matrices are NOT equal!')

def _compare_json(ref, est):
    with open(ref) as f:
        data = json.load(f)
    ref_json = {int(k): np.array(v) for k, v in data.items()}
    with open(est) as f:
        data = json.load(f)
    est_json = {int(k): np.array(v) for k, v in data.items()}

    is_equal = True
    for key in ref_json.keys():
        if not np.allclose(ref_json[key], est_json[key], atol=0.001):
            is_equal = False
            break
    if not is_equal:
        print(ref)
        print('\t', 'JSON files are NOT equal!')

def __run_subfile_code(subfile, corresponding_est_file):
    if corresponding_est_file.exists() == False:
        print(subfile)
        print('\t', 'File does not exist in estimated path!')
        return
    if subfile.suffix in ['.nii', '.gz']:
        _compare_images(subfile, corresponding_est_file)
    elif subfile.suffix == '.mat':
        _compare_matrices(subfile, corresponding_est_file)


def compare_results_folder(ref_path, est_path):
    for file in ref_path.glob('*'):
        if file.is_dir():
            for subfile in file.glob('*'):
                if subfile.is_dir():
                    for subsubfile in subfile.glob('*'):
                        corresponding_est_file = est_path / file.name / subfile.name / subsubfile.name
                        __run_subfile_code(subsubfile, corresponding_est_file)
                else:
                    corresponding_est_file = est_path / file.name / subfile.name
                    __run_subfile_code(subfile, corresponding_est_file)
        else:
            corresponding_est_file = est_path / file.name
            if corresponding_est_file.exists() == False:
                print(file)
                print('\t', 'File does not exist in estimated path!')
                continue
            if file.suffix in ['.nii', '.gz']:
                _compare_images(file, corresponding_est_file)
            elif file.suffix == '.mat':
                _compare_matrices(file, corresponding_est_file)
            elif file.suffix == '.json':
                _compare_json(file, corresponding_est_file)
