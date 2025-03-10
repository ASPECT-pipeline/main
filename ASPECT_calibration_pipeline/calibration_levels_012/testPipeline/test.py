import sys
import os
from astropy.io import fits
import numpy as np

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(parent_dir)

import calibrationMain

#Path to VIS, NIR1 and NIR2
vis = os.path.join(main_dir, "test_data/levels_012_test/test_data/vis_lo_600w_2500")
nir1 = os.path.join(main_dir, "test_data/levels_012_test/test_data/nir1_lo_600w_2500")
nir2 = os.path.join(main_dir, "test_data/levels_012_test/test_data/nir2_lo_600w_2500")
swir = os.path.join(main_dir, "test_data/levels_012_test/test_data/swir_test22")

output = os.path.join(main_dir, "test_data/levels_012_test/test_output")
outputPath = os.path.join(output, "test_1")

calibrationMain.pipeline(vis,nir1,nir2,swir,outputPath)

def read_fits_file(path):
    with fits.open(path) as hdul:
        h = hdul[0].header
        print(repr(h))
        total_size = hdul._file.size
        num_extensions = len(hdul) - 1

        #check extension 1 data:
        extension = hdul[1]
        header = extension.header
        print(repr(header))
        print(f'Total Size: {total_size}')
        print(f'Number of extensions: {num_extensions}')
        # print(extension.data)

def read_output_files():
    vis = os.path.join(os.getcwd(), "outputFiles/VIS_l/VIS_l_RC.fits")
    nir1 = os.path.join(os.getcwd(), "outputFiles/NIR1_l/NIR1_l_RC.fits")
    nir2 = os.path.join(os.getcwd(), "outputFiles/NIR2_l/NIR2_l_RC.fits")
    swir = os.path.join(os.getcwd(), "outputFiles/SWIR_/SWIR_.fits")

    # read_fits_file(vis)
    # read_fits_file(nir1)
    # read_fits_file(nir2)
    # read_fits_file(swir)

# read_output_files()

