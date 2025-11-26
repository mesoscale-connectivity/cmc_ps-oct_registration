import os
from fsl.data.image import Image
from pathlib import Path

image = '/Users/Vasilis/Downloads/CMC_results/Moe_flirt_Cross/Cro_slide_deck.nii.gz'

rest_of_command = f"--scene ortho --layout horizontal \
--hideCursor --bgColour 0 0 0 {image} \
--overlayType volume --cmap greyscale --displayRange 50.0 70.0 \
--interpolation none"

for axis in ['x', 'y', 'z']:

    os.makedirs(f'my_folder_{axis}', exist_ok=True)

    image_size = {label: Image(image).shape[i] for i, label in enumerate(['x', 'y', 'z'])}

    for sl in range(0,image_size[axis],round(image_size[axis]/200)):
        sl_fill = f'{str(sl).zfill(4)}'   
        print(f"slice {sl_fill}")
        out = f'my_folder_{axis}/image_{sl_fill}.png'
        if axis == 'x':
            cmd = f"fsleyes render -of {out}  --voxelLoc {sl} {image_size['y']//2} {image_size['z']//2} --hidey --hidez {rest_of_command}"
        elif axis == 'y':
            cmd = f"fsleyes render -of {out}  --voxelLoc {image_size['x']//2} {sl} {image_size['z']//2} --hidex --hidez {rest_of_command}"
        elif axis == 'z':
            cmd = f"fsleyes render -of {out}  --voxelLoc {image_size['x']//2} {image_size['y']//2} {sl} --hidex --hidey {rest_of_command}"
        os.system(cmd)

    import imageio
    from glob import glob

    # glob does not necessarily give the files in alpha/numeric order, so we used 'sorted'
    filenames = sorted(glob(f'my_folder_{axis}/image_????.png'), reverse=True)
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(f'{Path(image).name}_{axis}.gif', images, loop=0)


    # cleanup stored images
    for filename in filenames:
        os.remove(filename)
    os.rmdir(f'my_folder_{axis}')

