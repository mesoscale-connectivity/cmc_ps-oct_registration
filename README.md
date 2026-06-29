# CMCmultimodal
Codebase to process PSOCT data for the CMC multimodal project. This package performs within-slice registration of PSOCT data and registration to an MRI reference volume. 2D slice and 3D slidedeck images are stored, along with the transformation matrices or warps.

The code provides a CLI wrapper function (see psoct_wrapper.py), as well as, a python library for more flexibility.

## Installation
### PSOCT processing pipeline
Currently this package is only available from GitLab:
```bash
git clone --recurse-submodules https://git.fmrib.ox.ac.uk/saad/cmcmultimodal.git
cd cmcmultimodal
pip install .
```

> If you are a developer and you are planning to contribute, then you can install it like this instead:
`pip install -e ".[dev]"`

### NIfTI to Zarr conversion module
If you also want to convert the PSOCT NIfTI files to Zarr format, then you should also run:
```bash
pip install cmc_zarr_tools/
```

## Usage
### CLI wrapper
A full list of available options can be found in the wrapper's help (i.e. running `psoct_wrapper -h`). Example usage of the CLI wrapper function:

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
> *Paths and filenames need to be updated to valid local paths.*


## PSOCT in Zarr format
### Conversion to Zarr
#### Case 1: Automatic conversion
If your NIfTI file is ready to be converted, i.e. you don't want to change its data, then you can use the [`nifti-zarr-py`](https://github.com/neuroscales/nifti-zarr-py) package as follows:

`nii2zarr <input_file> <output_file>`

#### Case 2: Manual conversion
If your NIfTI require a "manipulation", e.g. you want to create a new data array from existing file(s), then [`cmc_zarr_tools`](https://github.com/VKarlaftis/cmc_zarr_tools) provides a CLI for creating a 3D zarr stack of 2D NIfTI images:

`create_3d_zarr <input_dir> <ref_slice> <output_zarr> <slice_axis>`

For the specific case of the PSOCT slices in native space, you have to first resample the slices to a common grid before you apply the `create_3d_zarr` command. This can be down with the following command:

`resampled_to_ref <input_slice> <ref_slice> <output_slice>`

> `ref_slice` in both commands above should be the reference slide used for the within-slice registration (i.e. `psoct.ref_slide`).

### Visualisation of Zarr data
Zarr data can be visualised in [Neuroglancer](https://github.com/google/neuroglancer). Here we demonstrate how to do that with the use of the [`ngtools`](https://github.com/neuroscales/ngtools) package.

1. Run `ngtools` in the terminal. That should start a python-like console in the terminal and open a webserver with Neuroglancer.
2. Run `load <input_file>` in the console.
3. If the image is not oriented as desired, then you can change it in the "layer side panel". Simply update the values in the "3x3 Source-by-Output dimensions" matrix and click 'Enter'.

#### Special case 1: Orientation images
Particular care should be taken for Orientation data, since the numerical values in each voxel is the angle of the local fibres.

To do so, we are using [`cmc_hybrid`](https://git.fmrib.ox.ac.uk/saad/cmc_hybrid) conventions on how to convert Orientation angles to 3D vectors. The following code will do that on the fly within the shader (this can be found in `Layer Side Panel` --> `Rendering` --> `Shader`). Further, it will create a slider widget to change the angle offset value (that might defer from subject to subject).

```GLSL
#uicontrol float offset slider(min=-180, max=180, step=1, default=35)
void main() {
  float pi = 3.14159265358979323846;
  float theta = getDataValue();
  theta = theta * 0.5;
  theta = pi - theta + offset * pi / 180.0;
  if (theta <= 0.0) {
    theta += pi;
  }
  if (theta > pi) {
    theta -= pi;
  }
  vec3 v = vec3(0.0, cos(theta), -sin(theta));
  v = abs(v);
  emitRGB(v);
}
```

#### Special case 2: 3D Vector images
If e.g. the Orientation data have already been converted to 3D vectors, then your input is a 4D image.

> For 4D Zarr data, make sure that the vector dimension is chunked as 3! If it isn't, then you need to re-chunk your data.

1. Check how your data are chunked. If the chunk-print is not showing a value equal to 3 for where the shape-print shows the corresponding vector dimension, then it means you need to re-chunk your data.
```python
import zarr
z = zarr.open("input_file.zarr/0", mode="r")
print(z.shape)
print(z.chunks)
```

2. Re-chunk your data.
```python
import zarr
src = zarr.open("input_file.zarr/0", mode="r")
root = zarr.open_group("output_file.zarr", mode="w")
root.attrs.update({
    "multiscales": [
        {
            "version": "0.4",
            "name": "",
            "axes": [
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space", "unit": "millimeter"},
                {"name": "y", "type": "space", "unit": "millimeter"},
                {"name": "x", "type": "space", "unit": "millimeter"},
            ],
            "datasets": [{"path": "0"}],
        }
    ]
})
dst = root.create_array("0",
                        shape=src.shape,
                        dtype=src.dtype,
                        chunks=(3, *src.chunks[1:]),
                        fill_value=src.fill_value,)
# repeat for other multiscales if needed ...
dst[:] = src[:]
```

3. To visualise your data, load them as before. Then update the name of the output dimension from `t` or `c'` (depending on if you had to do re-chunking) to `c^` (`Layer Side Panel` --> `Source` --> `Output dimensions`). This is essential so that Neuroglancer recognises this dimension as a channel that can be manipulated in the shader. Then enter the following code in the shader (`Layer Side Panel` --> `Rendering` --> `Shader`).
```GLSL
void main() {
  vec3 v = abs(vec3(
    getDataValue(0),
    getDataValue(1),
    getDataValue(2)
  ));
  emitRGB(v);
}
```

## Authors and acknowledgment
The code for this package is developed by: 
- Saad Jbabdi&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<saad.jbabdi@ndcn.ox.ac.uk> 
- Vasilis Karlaftis&nbsp;&nbsp;<vasilis.karlaftis@ndcn.ox.ac.uk>

Copyright (C) 2026 University of Oxford
