This document contains the CMCmultimodal release history in reverse chronological order.

0.3.0 (Monday 29th June 2026)
-----------------------------
- Added coefficient field in output files
- Updated initial MRI_in_PSOCT output file with correct non-linear version (if exists)
- Added `cmc_zarr_tools` as submodule for NIfTI to Zarr conversion

0.2.0 (Friday 27th March 2026)
------------------------------
- Added support to 'sagittal' orientation data
- Added slide deck creation of lowres slices
- Added option to invert warpfield
- Nifti are now created with correct header and zero-padding from the beginning
- Removed support of cross-correlation method

0.1.3 (Friday 5th December 2025)
--------------------------------
- Nifti data are NOT resampled anymore, only header is updated

0.1.2 (Wednesday 26th November 2025)
------------------------------------
- Code verified on 'coronal' dataset
- Implementation of linear and non-linear registration to MRI
- Nifti data resampled after within-slice-registration
- Backwards compatibility with cross-correlation method
