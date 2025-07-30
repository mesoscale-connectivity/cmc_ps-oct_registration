#!/usr/bin/env python

'''
Tests methods of PSOCT class in proc.py

Authors: Vasilis Karlaftis    <vasilis.karlaftis@ndcn.ox.ac.uk>

Copyright (C) 2025 University of Oxford
'''

from cmcmultimodal.proc import psoct
from pathlib import Path
import numpy as np

datadir = Path(__file__).parent / 'testdata'

# Test #1: check align functionality
def test_align():
    my_data = psoct(Path(datadir), lowres=True, slide_range=(98,200))
    slides, shifts = my_data.run_registration(bad_slides=[140,], align_ref='centre', align_thr=0)

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

    assert np.array_equal(slides, np.array(list(ref_data.keys())))
    assert np.allclose(shifts, np.array(list(ref_data.values())), atol=0.5)
