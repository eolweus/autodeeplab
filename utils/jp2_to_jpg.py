import argparse
import rasterio
import numpy as np
from rasterio.plot import reshape_as_image
from PIL import Image

# define command line arguments
parser = argparse.ArgumentParser(description='Convert jp2 image to jpeg')
parser.add_argument('input', metavar='input_file', type=str,
                    help='path to input jp2 image')
parser.add_argument('output', metavar='output_file', type=str,
                    help='path to output jpeg image')

# parse command line arguments
args = parser.parse_args()

# open the jp2 file
with rasterio.open(args.input) as dataset:

    num_bands = dataset.count

    # check that the image has at least 3 bands
    if num_bands == 1:
        print('Image has only 1 band. Converting to grayscale.')
        red_band = dataset.read(1)
        green_band = dataset.read(1)
        blue_band = dataset.read(1)
    elif num_bands < 3:
        print('Image has only {} bands. Converting to grayscale.'.format(
            num_bands))
        red_band = dataset.read(1)
        green_band = dataset.read(1)
        blue_band = dataset.read(1)
    elif num_bands > 3:
        print('Image has {} bands. Using only the first 3.'.format(num_bands))
        red_band = dataset.read(4)
        green_band = dataset.read(3)
        blue_band = dataset.read(2)

    # extract the red, green, and blue bands

    # reshape the bands as a single image
    image = reshape_as_image([red_band, green_band, blue_band])

    # handle invalid pixels
    invalid_mask = np.isnan(image).any(axis=2)
    image[invalid_mask] = [0, 0, 0]

    # Normalize the image to 0-255 range
    normalized_image = ((image - image.min()) *
                        (255 / (image.max() - image.min()))).astype(np.uint8)

    # Create a PIL image from the normalized numpy array
    pil_image = Image.fromarray(normalized_image)

    # save the image as a jpeg file
    pil_image.save(args.output)
    print('Saved image to {}'.format(args.output))
