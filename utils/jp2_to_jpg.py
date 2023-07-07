import argparse
import os
import rasterio
import numpy as np
from rasterio.plot import reshape_as_image
from PIL import Image

# define command line arguments
parser = argparse.ArgumentParser(description='Convert jp2 images to jpeg')
parser.add_argument('input', type=str,
                    help='path to input folder containing jp2 images')
parser.add_argument('output', type=str,
                    help='path to output folder for saving jpeg images')

# parse command line arguments
args = parser.parse_args()

# create the output folder if it doesn't exist
os.makedirs(args.output, exist_ok=True)

# iterate over the files in the input folder
for filename in os.listdir(args.input):
    # check if the file is a jp2 image
    if filename.endswith('.jp2'):
        # construct the input and output file paths
        input_path = os.path.join(args.input, filename)
        output_filename = os.path.splitext(filename)[0] + '.jpeg'
        output_path = os.path.join(args.output, output_filename)

        # open the jp2 file
        with rasterio.open(input_path) as dataset:
            num_bands = dataset.count

            # check that the image has at least 3 bands
            if num_bands < 3:
                print('Image {} has less than 3 bands. Converting to grayscale.'.format(filename))
                red_band = dataset.read(1)
                green_band = dataset.read(1)
                blue_band = dataset.read(1)
            elif num_bands > 3:
                print('Image {} has more than 3 bands. Using only the first 3.'.format(filename))
                red_band = dataset.read(4)
                green_band = dataset.read(3)
                blue_band = dataset.read(2)
            else:
                red_band = dataset.read(1)
                green_band = dataset.read(2)
                blue_band = dataset.read(3)

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
            pil_image.save(output_path)
            print('Converted and saved {} to {}'.format(filename, output_path))
