#!/usr/bin/env python

'''
Tests methods return the same result after notebook conversion

Authors: Vasilis Karlaftis    <vasilis.karlaftis@ndcn.ox.ac.uk>

Copyright (C) 2025 University of Oxford
'''

from pathlib import Path
import numpy as np
import pytest

from fsl.data.image     import Image
from cmcmultimodal.proc import psoct

datadir = Path(__file__).parent / 'benchmark'

# Run PSOCT analysis main steps as chained fixtures
# to avoid repeating the analysis steps
@pytest.fixture(scope="module")
def step1_createClass():
    data_class = psoct(Path(datadir), lowres=True, slide_range=(98,200))
    return data_class

@pytest.fixture(scope="module")
def step2_runRegistration(step1_createClass):
    data_class = step1_createClass
    slides, rel_shifts, abs_shifts = data_class.run_registration(bad_slides=[140,], align_ref='centre', align_thr=0)
    return data_class, slides, rel_shifts, abs_shifts

@pytest.fixture(scope="module")
def step3_applyRegistration(step2_runRegistration):
    data_class, slides, _, abs_shifts = step2_runRegistration
    data_class.apply_registration(slides, abs_shifts, 'coronal', None, 1)
    return data_class, abs_shifts

@pytest.fixture(scope="module")
def step4_alignMRI2PSOCT(step3_applyRegistration, tmp_path_factory):
    data_class, abs_shifts = step3_applyRegistration
    tmpdir = tmp_path_factory.mktemp("test_results")
    data_class.output_path = tmpdir
    mri_ref = datadir / 'reoriented_FA.nii.gz'
    # mri_ref = datadir / 'dti_FA.nii.gz'
    mat_file, data_file = data_class.align_mri_to_psoct(mri_ref)
    return data_class, mat_file, data_file, abs_shifts

@pytest.fixture(scope="module")
def step5_alignPSOCT2MRI(step4_alignMRI2PSOCT):
    data_class, mat_file, _, abs_shifts = step4_alignMRI2PSOCT
    mri_ref = datadir / 'reoriented_FA.nii.gz'
    # mri_ref = datadir / 'dti_FA.nii.gz'
    data_file = data_class.align_psoct_to_mri(mat_file, mri_ref)
    return data_class, data_file, abs_shifts

@pytest.fixture(scope="module")
def step6_update_headers(step5_alignPSOCT2MRI):
    data_class, data_file, abs_shifts = step5_alignPSOCT2MRI
    indiv_slides = data_class.update_nifti_headers(data_file, 'coronal')
    data_class.apply_to_highres_images(indiv_slides, abs_shifts, 'coronal', 'Retardance')
    return data_class


# Test #1: check interpolated_slides match the reference data
def test_interpolated_slides(step1_createClass):
    data_class = step1_createClass
    data_class.label_bad_slides(indices=[140,])
    data_class.interpolate_missing_slides()
    
    ref_data = [99, 100, 105, 106, 107, 108, 109, 140]

    assert data_class.slide_range == (98,200)
    assert data_class.interpolated_slides == ref_data

# Test #2: check alignment matches the reference data
def test_align(step2_runRegistration):
    data_class, slides, rel_shifts, _ = step2_runRegistration

    central_slide = 190 
    ref_data = {
        98: np.array([0.0, 0.0]),
        99: np.array([0.0, 0.0]),
        100: np.array([0.0, 0.0]),
        101: np.array([0.0, 0.0]),
        102: np.array([0.0, 0.0]),
        103: np.array([0.0, 0.0]),
        104: np.array([0.0, 0.0]),
        105: np.array([0.0, 0.0]),
        106: np.array([0.0, np.float64(2.0)]),
        107: np.array([0.0, 0.0]),
        108: np.array([0.0, 0.0]),
        109: np.array([0.0, 0.0]),
        110: np.array([0.0, 0.0]),
        111: np.array([0.0, 0.0]),
        112: np.array([0.0, 0.0]),
        113: np.array([0.0, 0.0]),
        114: np.array([0.0, 0.0]),
        115: np.array([0.0, 0.0]),
        116: np.array([0.0, np.float64(90.0)]),
        117: np.array([0.0, 0.0]),
        118: np.array([np.float64(-5.0), np.float64(96.0)]),
        119: np.array([np.float64(5.0), np.float64(-6.0)]),
        120: np.array([0.0, 0.0]),
        121: np.array([0.0, 0.0]),
        122: np.array([0.0, 0.0]),
        123: np.array([0.0, 0.0]),
        124: np.array([0.0, 0.0]),
        125: np.array([0.0, 0.0]),
        126: np.array([0.0, 0.0]),
        127: np.array([0.0, 0.0]),
        128: np.array([0.0, 0.0]),
        129: np.array([0.0, 0.0]),
        130: np.array([np.float64(-5.0), np.float64(6.0)]),
        131: np.array([np.float64(5.0), np.float64(-6.0)]),
        132: np.array([0.0, 0.0]),
        133: np.array([np.float64(-5.0), np.float64(6.0)]),
        134: np.array([0.0, np.float64(2.0)]),
        135: np.array([np.float64(-5.0), np.float64(6.0)]),
        136: np.array([0.0, 0.0]),
        137: np.array([0.0, np.float64(44.0)]),
        138: np.array([np.float64(5.0), np.float64(27.0)]),
        139: np.array([np.float64(-5.0), np.float64(-27.0)]),
        140: np.array([0.0, 0.0]),
        141: np.array([0.0, 0.0]),
        142: np.array([0.0, 0.0]),
        143: np.array([0.0, 0.0]),
        144: np.array([np.float64(5.0), np.float64(27.0)]),
        145: np.array([0.0, 0.0]),
        146: np.array([np.float64(-5.0), np.float64(-27.0)]),
        147: np.array([np.float64(5.0), np.float64(27.0)]),
        148: np.array([0.0, 0.0]),
        149: np.array([0.0, 0.0]),
        150: np.array([0.0, 0.0]),
        151: np.array([0.0, 0.0]),
        152: np.array([0.0, 0.0]),
        153: np.array([np.float64(-5.0), np.float64(-27.0)]),
        154: np.array([np.float64(5.0), np.float64(27.0)]),
        155: np.array([0.0, 0.0]),
        156: np.array([0.0, 0.0]),
        157: np.array([0.0, 0.0]),
        158: np.array([0.0, 0.0]),
        159: np.array([0.0, 0.0]),
        160: np.array([0.0, 0.0]),
        161: np.array([np.float64(-5.0), np.float64(-27.0)]),
        162: np.array([np.float64(5.0), np.float64(26.0)]),
        163: np.array([0.0, 0.0]),
        164: np.array([0.0, 0.0]),
        165: np.array([0.0, 0.0]),
        166: np.array([0.0, 0.0]),
        167: np.array([0.0, 0.0]),
        168: np.array([0.0, 0.0]),
        169: np.array([0.0, 0.0]),
        170: np.array([0.0, 0.0]),
        171: np.array([0.0, 0.0]),
        172: np.array([0.0, 0.0]),
        173: np.array([0.0, 0.0]),
        174: np.array([np.float64(40.0), np.float64(-27.0)]),
        175: np.array([np.float64(-40.0), np.float64(27.0)]),
        176: np.array([0.0, 0.0]),
        177: np.array([0.0, 0.0]),
        178: np.array([np.float64(-5.0), np.float64(-27.0)]),
        179: np.array([np.float64(5.0), np.float64(27.0)]),
        180: np.array([np.float64(-6.0), np.float64(8.0)]),
        181: np.array([0.0, 0.0]),
        182: np.array([np.float64(1.0), np.float64(-1.0)]),
        183: np.array([np.float64(-5.0), np.float64(-26.0)]),
        184: np.array([np.float64(5.0), np.float64(27.0)]),
        185: np.array([0.0, 0.0]),
        186: np.array([0.0, 0.0]),
        187: np.array([0.0, 0.0]),
        188: np.array([0.0, 0.0]),
        189: np.array([0.0, 0.0]),
        190: np.array([0, 0]),
        191: np.array([0.0, 0.0]),
        192: np.array([0.0, 0.0]),
        193: np.array([0.0, 0.0]),
        194: np.array([np.float64(-90.0), 0.0]),
        195: np.array([np.float64(90.0), 0.0]),
        196: np.array([0.0, 0.0]),
        197: np.array([0.0, 0.0]),
        198: np.array([0.0, 0.0]),
        199: np.array([0.0, 0.0]),
        200: np.array([np.float64(3.0), np.float64(-5.0)]),
    }

    assert data_class.ref_slide == central_slide
    assert np.array_equal(slides, np.array(list(ref_data.keys())))
    assert np.allclose(rel_shifts, np.array(list(ref_data.values())), atol=0.5)

# Test #3: check slide_deck image matches the reference data
def test_apply_registration(step3_applyRegistration):
    data_class, _ = step3_applyRegistration

    ref_data_file = datadir / 'slide_deck'
    ref_data = Image(ref_data_file).data
    est_data = data_class.slide_deck_img.data

    assert np.allclose(ref_data, est_data)

# Test #4: check MRI2PSOCT alignment matches the reference data
def test_align_mri_to_psoct(step4_alignMRI2PSOCT):
    _, est_mat_file, est_data_file, _ = step4_alignMRI2PSOCT

    ref_data_file = datadir / 'fa_to_slides'
    ref_data = Image(ref_data_file).data
    ref_mat_file = datadir / 'dti_to_slides.mat'
    ref_mat = np.loadtxt(ref_mat_file)

    est_data = Image(est_data_file).data
    est_mat = np.loadtxt(est_mat_file)

    assert np.allclose(ref_mat, est_mat, atol=0.001)
    assert np.allclose(ref_data, est_data)

# Test #5: check PSOCT2MRI alignment matches the reference data
def test_align_psoct_to_mri(step5_alignPSOCT2MRI):
    _, est_data_file, _ = step5_alignPSOCT2MRI

    ref_data_file = datadir / 'slide_deck_with_header'
    ref_data = Image(ref_data_file).data
    est_data = Image(est_data_file).data

    assert np.allclose(ref_data, est_data)

# Test #6: check update_nifti_headers & apply_to_highres_images results match the reference data
def test_nifti_headers(step6_update_headers):
    output_path = step6_update_headers.output_path
    highres_ref = datadir / 'highres_with_header'
    for file in highres_ref.glob('*.nii.gz'):
        ref_img = Image(file)
        est_img_file = output_path / file.name
        est_img = Image(est_img_file)
        assert np.allclose(ref_img.data, est_img.data)
        assert ref_img.header == est_img.header
