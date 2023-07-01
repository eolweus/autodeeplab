import os
import rasterio
from rasterio.enums import Resampling


def convert_to_tiff(jp2_file, tiff_file):
    with rasterio.open(jp2_file) as src:
        tiff_profile = src.profile
        tiff_profile.update(driver='GTiff')

        data = src.read(1)  # Read the first band

        with rasterio.open(tiff_file, 'w', **tiff_profile) as dst:
            dst.write(data, 1)


def batch_convert(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".jp2"):
            jp2_file = os.path.join(directory, filename)
            tiff_file = os.path.join(directory, filename[:-4] + '.tif')
            # tiff_file = os.path.join(directory, filename[:-4] + '.tif')
            convert_to_tiff(jp2_file, tiff_file)
            print(f"Converted {jp2_file} to {tiff_file}")


# Use the function on a directory
if __name__ == '__main__':
    batch_convert('/cluster/home/erlingfo/autodeeplab/test_images/target')
