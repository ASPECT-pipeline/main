import os
import modules.utilities as utilities
import modules.convertToFits as convertToFits
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

acq_path = os.path.join(os.getcwd(), 'test_data/ASPECT_fly_images/acqseq_101')
# acq_path = os.path.join(os.getcwd(), 'test_data/ASPECT_fly_images/acqseq_104')
decoded_binaries = os.path.join(acq_path, 'acq_000_decompressed')
meta_folder = os.path.join(acq_path, 'meta')
fits_output_dir = os.path.join(os.getcwd(), 'test_data/levels_012_test/test_output/ASPECT_fly/101')
aspect_fly_fits_vis = os.path.join(fits_output_dir, 'AS0_XXXXXX_200101T014411_1C.fits')
aspect_fly_fits_nir1 = os.path.join(fits_output_dir, 'AS1_XXXXXX_200101T014800_0A.fits')
aspect_fly_fits_nir2 = os.path.join(fits_output_dir, 'AS2_XXXXXX_200101T014800_0A.fits')
aspect_fly_fits_swir = os.path.join(fits_output_dir, 'AS3_XXXXXX_200101T014411_1B.fits')

autoseq_dir = os.path.join(os.getcwd(), 'test_data/ASPECT_Autoseq_20240809')
autoseq_encoded_vis0 = os.path.join(autoseq_dir, 'acqseq_505/acq_000/diff_encoding/dc_0_exp_001_diffEnc.bin.jp2')
autoseq_decoded_vis0 = os.path.join(autoseq_dir, 'diff_decoded/505/dc_0_decoded.dat02.img')
autoseq_decoding_ouput = os.path.join(autoseq_dir, 'pipeline_diff_decoded/505')

# simulated_cube = os.path(os.getcwd())

# Getting channels frame counts and original fiel names from acquisition folder
def test_channel_frames_names():
    acq_folder = utilities.get_acq_folder(acq_path)
    channels_and_frames = utilities.get_channel_frames_names(acq_folder)
    print(channels_and_frames)

# Getting acquisition ID and acquisition sequence ID 
def test_acqseq():
    acq_folder = utilities.get_acq_folder(acq_path)
    result = utilities.get_acqSeq(acq_folder)
    print(result)

# Read telemetry and retrive metadata
def test_primary_metadata():

    result = utilities.collect_primary_metadata(meta_folder, 'VIS')
    print(result)

def test_spice_metadata():
    telemetry_path = os.path.join(meta_folder, 'telemetry.json')
    utilities.collect_spice_metadata(telemetry_path, '', '')

# Test the fits file conversion as a whole
def test_convert_to_fits(output: str):
    convertToFits.convert_to_fits(acq_path, output)

def read_fits_file(path, visualise = True):
    with fits.open(path) as hdul:
        print(f'Fitsfile from path:')
        print({path})

        for i, hdu in enumerate(hdul):

            h = hdu.header
            print(f'Header for HDU {i}')
            print(repr(h))

            if visualise:
                if isinstance(hdu, fits.ImageHDU):
                    print("→ This is an ImageHDU")
                    print(f'data type: {type(hdu.data)}')
                    for i, frame in enumerate(hdu.data):
                        plt.imshow(frame, cmap='gray')
                        plt.title(f'frame: {i}')
                        plt.show()
                elif isinstance(hdu, fits.BinTableHDU):
                    print("→ This is a Binary Table HDU")
                    print(f'data type: {type(hdu.data)}')
                    print(f'SWIR data:')
                    print(hdu.data)
            
        print()

def read_bin_dir(dir_path: str | Path):
    dir_path = Path(dir_path)

    bin_files = sorted(dir_path.glob("*.bin"))
    print(f'bin files: {bin_files}')
    channel_map = {
        0: (1024, 1024),
        1: (518, 648),
        2: (518, 648),
        3: (1,0)
    }

    pattern = re.compile(r'^dc_(\d)_')

    for i, file in enumerate(bin_files):
        with file.open("rb") as f:
            data = f.read()
            frame = np.frombuffer(data, dtype=np.uint16)

        match = pattern.match(file.name)
        if match:
            print(f'file {i} matched')
            index = int(match.group(1))
            if index in channel_map:
                height, width = channel_map[index]
            else:
                raise ValueError(f'wrong file name. {file.name}')
       

        frame = frame.reshape((height, width))

        plt.imshow(frame, cmap='gray')
        plt.title(f'Frame {i+1}: {file.name}')
        plt.axis('off')
        plt.show()


def test_decoding(input:str, output: str, compare: str):

    print(f'decoding: {input}')
    decoded = utilities.decompress_jp2(input, output)
    print(f'decoded file: {decoded}')
    print(f'comparing to: {compare}')

    try:
        # Load binary data and convert to images
        data_1 = np.fromfile(decoded, dtype=np.uint16).reshape((1024, 1024))
        data_2 = np.fromfile(compare, dtype=np.uint16).reshape((1024, 1024))

        print(f'data shape: {data_1.shape}, {data_2.shape}')
        print(f'data match: {np.array_equal(data_1, data_2)}')

        print(f'First 20 of pipeline: {data_1.ravel()[:20]}')
        print(f'First 20 of decoded: {data_2.ravel()[:20]}')

        # Visualize both images and their difference
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(data_1, cmap='gray')
        plt.title('Pipeline Decoded')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(data_2, cmap='gray')
        plt.title('Reference Decoded')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(np.abs(data_1 - data_2), cmap='hot')
        plt.title('Absolute Difference')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        return np.array_equal(data_1, data_2)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return False

"""
Function calls after this
"""
# test_acqseq()
# test_channel_frames_names()
# test_primary_metadata()
# test_convert_to_fits(output=fits_output_dir)
# test_spice_metadata()

read_fits_file(aspect_fly_fits_vis, True)
# read_bin_dir(decoded_binaries)
# print(test_decoding(autoseq_encoded_vis0, autoseq_decoding_ouput, autoseq_decoded_vis0))



# Python3 ASPECT_calibration_pipeline/levels_012/test_level_012.py