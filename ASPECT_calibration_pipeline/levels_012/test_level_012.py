import os
import modules.utilities as utilities
import modules.convertToFits as convertToFits
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import re
import cv2
import math
from matplotlib.patches import Patch
from pprint import pprint
from collections import defaultdict


# ASPECT FLY images
acq_path = os.path.join(os.getcwd(), 'test_data/ASPECT_fly_images/acqseq_101')
acq_path = os.path.join(os.getcwd(), 'test_data/ASPECT_fly_images/acqseq_104')
decoded_binaries = os.path.join(acq_path, 'acq_000_decompressed')
decompressed = os.path.join(decoded_binaries , 'dc_1_exp_000.bin')

# ASPECT FLY output
meta_folder = os.path.join(acq_path, 'meta')
fits_output_dir = os.path.join(os.getcwd(), 'test_data/levels_012_test/test_output/ASPECT_fly/104')
aspect_fly_fits_vis = os.path.join(fits_output_dir, 'AS0_XXXXXX_200101T014411_1B.fits')
aspect_fly_fits_nir1 = os.path.join(fits_output_dir, 'AS1_XXXXXX_200101T014800_1B.fits')
aspect_fly_fits_nir2 = os.path.join(fits_output_dir, 'AS2_XXXXXX_200101T014800_1B.fits')
aspect_fly_fits_swir = os.path.join(fits_output_dir, 'AS3_XXXXXX_200101T014411_1B.fits')
aspect_fly_fits_nir1_nir2 = os.path.join(fits_output_dir, 'ASP_XXXXXX_200101T014800_2B.fits')

# ASPECT DIFF encoded images
autoseq_dir = os.path.join(os.getcwd(), 'test_data/ASPECT_Autoseq_20240809') # FOLDER
autoseq_505 = os.path.join(autoseq_dir, 'acqseq_505')
autoseq_505_diff = os.path.join(autoseq_dir, 'acqseq_505/acq_000/diff_encoding')
autoseq_505_out = os.path.join(autoseq_dir, 'pipeline_diff_decoded')
autoseq_505_in = os.path.join(autoseq_dir, 'pipeline_jp2_decoded')
autoseq_505_offsets = os.path.join(autoseq_dir, 'acqseq_505/acq_000/meta_diff_encoding/diff_encoding.json')

autoseq_encoded_vis0 = os.path.join(autoseq_dir, 'acqseq_505/acq_000/diff_encoding/dc_0_exp_001_diffEnc.bin.jp2')
autoseq_decoded_vis0 = os.path.join(autoseq_dir, 'diff_decoded/505/dc_0_decoded.dat02.img')
autoseq_decoding_ouput = os.path.join(autoseq_dir, 'pipeline_diff_decoded/505')

# SIMULATED ASPECT images
simulated_dir = os.path.join(os.getcwd(), 'test_data/ASPECT_simulated_images/2027-03-23_06_00_00-McEwen/acq_000')
simulated_vis = os.path.join(simulated_dir, 'dc_0_exp_000.bin')
simulated_nir1 = os.path.join(simulated_dir, 'dc_1_exp_000.bin')

simulated_output_dir = os.path.join(os.getcwd(), 'test_data/levels_012_test/test_output/ASPECT_simulated/2027-03-23_06_00_00')
simulated_output_vis = os.path.join(simulated_output_dir, 'AS0_XXXXXX_270323T060000_1B.fits')
simulated_output_nir1 = os.path.join(simulated_output_dir, 'AS1_XXXXXX_270323T060000_1B.fits')
simulated_output_nir2 = os.path.join(simulated_output_dir, 'AS2_XXXXXX_270323T060000_1B.fits')
simulated_output_ASP = os.path.join(simulated_output_dir, 'ASP_XXXXXX_270323T060000_2B.fits')

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
def test_convert_to_fits(input: str, output: str):
    convertToFits.convert_to_fits(acq_path, output, )

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
                        print(f'points from frame {i}')
                        print(f'(250, 250): {frame[250][250]}')
                        print(f'(10, 150): {frame[10][150]}')
                        print(f'(300, 300): {frame[300][300]}')
                        plt.imshow(frame, cmap='gray')
                        plt.title(f'frame: {i}')
                        plt.show()
                elif isinstance(hdu, fits.BinTableHDU):
                    if i == 1:
                        print("→ This is a Binary Table HDU")
                        print(f'data type: {type(hdu.data)}')
                        print(f'SWIR data:')
                        print(hdu.data)

                    else:
                        data = hdu.data
                        # Iterate over each column
                        for col_name in data.columns.names:
                            print(f"Column: {col_name}") # The 2D frame of the data cube
                            # Iterate over each row in the column (image)
                            row0_array = data[col_name][0]
                            print(f"Row 8 length: {len(row0_array)}")
        
            
        print()

def visualise_fits(fitsPath, visualise:bool = True, spect:bool = False):
    name = os.path.splitext(os.path.basename(fitsPath))[0]
    print(f'Reading file: {name}')
    # Open FITS file using astropy
    with fits.open(fitsPath) as hdul:
        print(f'info:\n {hdul.info()}') # Print the info of the hdul
        total_size = hdul._file.size # Total size of the file
        print(f"Total File Size: {total_size} bytes")
        num_extensions = len(hdul) - 1 # Number of extensions
        print(f"Number of Extensions: {num_extensions}")


        # Iterate through each extension
        for i, hdu in enumerate(hdul):
            print('')
            print(f'\nHDU number: {i}') # Extension number
            header = hdu.header
            data = hdu.data
            # print(f'Data cube shape: {data.shape}')

            print(repr(header))
            

            if isinstance(hdu, fits.ImageHDU):
                naxis1 = header.get('NAXIS1')  # Width
                naxis2 = header.get('NAXIS2')  # Height
                naxis3 = header.get('NAXIS3')  # Number of images


                # Determine grid layout
                max_images_per_row = 5
                rows = math.ceil(naxis3 / max_images_per_row)
                cols = min(max_images_per_row, naxis3)

                # utils.plot_random_spectra(data, header)
                x = 110
                y = 130
                if spect:
                    spectra = []
                    for i in range(naxis3):
                        val = data[i, y, x]
                        spectra.append(val)

                    print(spectra)
                    print(f'vis bg: {data[0, 490, 585]}')
                    print(f'nir bg: {data[12, 490, 585]}')

                    vis_wl = [int(w.strip()) for w in header['VIS_WL'].split(',')]
                    nir1_wl = [int(w.strip()) for w in header['NIR1_WL'].split(',')]
                    nir2_wl = [int(w.strip()) for w in header['NIR2_WL'].split(',')]
                    nir_wl = nir1_wl + nir2_wl
                    all_wavelengths = vis_wl + nir1_wl + nir2_wl
                    spectra = np.array(spectra)
                    spectra = spectra.astype(np.float32)
                    spectra[:11] /= 4096 # 2^12
                    spectra[11:] /= 16384 # 2^14
                    print(f'Spectra length: {len(spectra)}')
                    print(f'Spectra length: {len(all_wavelengths)}')
                    print(f'vis wl: {len(vis_wl)}')
                    print(f'nir1 wl: {len(nir1_wl)}')
                    print(f'nir2 wl: {len(nir2_wl)}')
                    vis_len = len(vis_wl)
                    nir1_len = len(nir1_wl)
                    nir2_len = len(nir2_wl)
                    nir1_s = spectra[vis_len : vis_len+nir1_len]
                    nir2_s = spectra[vis_len+nir1_len : ]

                    # plt.figure(figsize=(10, 5))
                    # plt.plot(all_wavelengths, spectra, 'ro-', label="Spectra")
                    # plt.xlabel("Wavelength (nm)")
                    # plt.ylabel("values")
                    # plt.title(f"Spectra from ({x}, {y})")
                    # plt.legend()
                    # plt.show()

                    # Display multiple spectras across the image
                    positions = [(250,250), (110, 130), (130, 300), (475, 260)]
                    spectra_list = [
                        np.concatenate([data[:11, y, x] / 4096, data[11:, y, x] / 16384])
                        for (x, y) in positions
                    ]
                    figure = utilities.plot_spectra_with_image(spectra_list,positions,data[0], all_wavelengths)
                    figure.show()

                    # corrected, _ = utils.nir2_offset_correction(nir1_wl, nir1_s, nir2_wl, nir2_s)
                    # connected = np.concatenate((spectra[: vis_len+nir1_len], corrected))
                    # print('lengths after segment correction')
                    # print(len(connected))

                    # # outliers = utils.remove_outliers(spectra, all_wavelengths)

                    # plt.figure(figsize=(10, 5))
                    # plt.plot(all_wavelengths, spectra, 'ro-', label="Original Spectra")
                    # plt.plot(all_wavelengths, connected, 'bo-', label="connected")
                    # plt.xlabel("Wavelength (nm)")
                    # plt.ylabel("values")
                    # plt.title("Spectra from (250, 250)")
                    # plt.legend()
                    # plt.show()


                    """
                    FOLLOWING PART IS MISSING CONVERSION COEFFIENTS
                    """
                    # vis_c = utils.load_conversion_file(vis_conversion)
                    # vis_results = utils.query_coefficients(vis_c, vis_wl)
                    # nir_c = utils.load_conversion_file(nir_conversion)
                    # nir_results = utils.query_coefficients(nir_c, nir_wl)
                    # combined_results = {**vis_results, **nir_results}

                    # reflectances = []
                    # for i, wl in enumerate(all_wavelengths):
                    #     print(f'wl: {i}: {wl}')
                    #     coefficient = combined_results[wl]
                    #     # print(f'coef: {coefficient}')
                    #     dn = spectra[i]
                    #     # print(f'dn: {dn}')
                    #     if i > 10:
                    #         exposure = 0.02
                    #     else: 
                    #         exposure = 0.01
                    #     dn_coef = dn / coefficient
                    #     reflectance = dn_coef / exposure
                    #     reflectances.append(reflectance)

                    # plt.figure(figsize=(10, 5))
                    # plt.plot(all_wavelengths, reflectance, 'ro-', label="Original Spectra")
                    # plt.xlabel("Wavelength (nm)")
                    # plt.ylabel("reflectance")
                    # plt.title("Spectra from (250, 250)")
                    # plt.legend()
                    # plt.show()

                if visualise:
                    plt.figure()
                    plt.suptitle('Frame 0', fontsize=16)
                    plt.imshow(data[0], cmap='gray')
                    plt.title(f'Slice 0')
                    plt.axis('off')

                    plt.tight_layout()
                    plt.show()

                    plt.figure()
                    plt.suptitle('Frame 0', fontsize=16)
                    plt.imshow(data[11], cmap='gray')
                    plt.title(f'Slice 0')
                    plt.axis('off')

                    plt.tight_layout()
                    plt.show()

                # Display all 2D images
                    # plt.figure(figsize=(cols * 4, rows * 4))
                    # plt.suptitle('2D Slices from Data Cube', fontsize=16)
                    # for i in range(naxis3):
                    #     plt.subplot(rows, cols, i + 1)
                    #     plt.imshow(data[i, :, :], cmap='gray')
                    #     plt.title(f'Slice {i + 1}')
                    #     plt.axis('off')

                    # plt.tight_layout()
                    # plt.show()

                    visN = cv2.normalize(data[1], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    nirN = cv2.normalize(data[14], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    legend_elements = [
                        Patch(facecolor='yellow', edgecolor='black', label='Aligned regions'),
                        Patch(facecolor='red', edgecolor='black', label='Only in vis image'),
                        Patch(facecolor='green', edgecolor='black', label='Only in nir image')
                    ]
                    overlay = utilities.overlay_images(data[0], data[11])
                    plt.figure()
                    plt.suptitle('Vis and Nir frame overlay', fontsize=16)
                    plt.imshow(overlay)
                    plt.axis('off')      
                    plt.figlegend(handles=legend_elements, loc='lower center', ncol=3, frameon=True, fontsize='medium')
                    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust to make room for legend and title

                    plt.show()

def readBinfile(filePath, channel):
    if channel == "VIS": 
        height = 1024
        width = 1024
    elif channel == "NIR":
        height = 518
        width = 648
    # print(f"height: {height} \nwidht: {width}")

    bytes_per_pixel, bit_depth = utilities.estimate_bit_depth(filePath, width, height)
    print(f'{channel} bytes per pixel: {bytes_per_pixel}')
    print(f'{channel} bit depth: {bit_depth}')
    effective_max = 2**bit_depth - 1
    print(f'effective max: {effective_max}')
    

    try:
        with open(filePath, 'rb') as file:
            binaryData = file.read()
            print(f"Read {len(binaryData)} bytes")

            imageArray = np.frombuffer(binaryData, dtype='>u2')
            imageArray = imageArray.reshape((height, width))
            print(f'(0,0): {imageArray[0][0]}')
            print(f'(3,150): {imageArray[3][150]}')
            print(f'(8,1): {imageArray[8][1]}')
            print(f'(346,647): {imageArray[346][647]}')
            print(f'(4,100) - (4, 105): {imageArray[4][100:105]}')
            print(f'(100,0) - (100, 4): {imageArray[100][:4]}')
            # print(f'array type: {imageArray.dtype}')
            # print(f'values [500][500] - [510][510]')
            # print(imageArray[500:510, 500:510])
            # plt.figure(figsize=(8,5))
            # plt.imshow(imageArray, cmap='gray')
            # plt.title(f'channel {channel}')
            # plt.axis('off')
            # plt.tight_layout()
            # plt.show()


    except FileNotFoundError:
        print(f"File {filePath} not found")
    except Exception as e:
        print(f"An error occured: {e}")

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

def update_fits_exposure(path, new_exposure, save_as=None):
    with fits.open(path, mode='update' if save_as is None else 'readonly') as hdul:
        for hdu in hdul:
            if 'EXPOSURE' in hdu.header:
                print(f"Old EXPOSURE: {hdu.header['EXPOSURE']}")
                hdu.header['EXPOSURE'] = new_exposure
                print(f"New EXPOSURE: {hdu.header['EXPOSURE']}")

        if save_as:
            hdul.writeto(save_as, overwrite=True)
            print(f"Saved updated file to {save_as}")
        else:
            print(f"Updated in place: {path}")

def update_fits_wl(path, save_as=None):
    wl_map = {
        'VIS' : '675,690,705,720,735,750,765,780,795,810,825',
        'NIR1': '875,904,933,963,992,1021,1050,1079,1108,1138,1167,1196,1225',
        'NIR2': '1225,1254,1283,1313,1342,1371,1400,1429,1458,1488,1517,1546,1575',
        'SWIR': '1675,1711,1748,1784,1820,1857,1893,1930,1966,2002,2075,2111,2148,2184,2220,2257,2293,2330,2366,2402,2439,2475'
    }
    with fits.open(path, mode='update' if save_as is None else 'readonly') as hdul:
        for hdu in hdul:
            if 'WAVELEN' in hdu.header:
                channel = hdu.header['CHANNEL']
                print(f"Old WL: {hdu.header['WAVELEN']}")
                hdu.header['WAVELEN'] = wl_map[channel]
                print(f"New WL: {hdu.header['WAVELEN']}")

        if save_as:
            hdul.writeto(save_as, overwrite=True)
            print(f"Saved updated file to {save_as}")
        else:
            print(f"Updated in place: {path}")

def test_decoding(input:str, output: str, compare: str):

    print(f'decoding: {input}')
    decoded = utilities.decompress_jp2(input, output)
    print(f'decoded file: {decoded}')
    print(f'comparing to: {compare}')
    decoded = Path(decoded)
    with decoded.open('rb') as f:
        raw = f.read(20)
    data_be = np.frombuffer(raw, dtype='>u2')  # big-endian
    data_le = np.frombuffer(raw, dtype='<u2')  # little-endian

    print("Big-endian:", data_be)
    print("Little-endian:", data_le)
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
    
def compare_bin_images(
    file_a: str | Path,
    file_b: str | Path,
    shape: tuple[int, int],
    dtype: str = "<u2",          # little-endian uint16 by default
    visualize: bool = True
) -> bool:
    file_a = Path(file_a)
    file_b = Path(file_b)

    arr_a = np.fromfile(file_a, dtype=dtype)
    arr_b = np.fromfile(file_b, dtype=dtype)

    if arr_a.size != arr_b.size:
        raise ValueError(f'Size missmatch: {file_a.name} has {arr_a.size} pixels' 
                         f'but {file_b.name} has {arr_b.size}')
    
    arr_a = arr_a.reshape(shape)
    arr_b = arr_b.reshape(shape)

    identical = np.array_equal(arr_a, arr_b)
    print(f"{'✔️  IDENTICAL' if identical else '❌  DIFFERENT'}"
          f'f- {file_a.name} vd {file_b.name}')
    
    if identical and not visualize:
        return True
    
    diff = arr_a.astype(np.int64) - arr_b.astype(np.int64)
    abs_diff = np.abs(diff)

    print(f'Max Δ: {abs_diff.max()} | Mean Δ: {abs_diff.mean():.3f}')

    if visualize:
        legend_elements = [
            Patch(facecolor='yellow', edgecolor='black', label='Aligned regions'),
            Patch(facecolor='red', edgecolor='black', label='Only in vis image'),
            Patch(facecolor='green', edgecolor='black', label='Only in nir image')
        ]
        overlay = utilities.overlay_images(arr_a, arr_b)
        plt.figure()
        plt.suptitle(f'{file_a.name} and {file_b.name} comparison', fontsize=16)
        plt.imshow(overlay)
        plt.axis('off')      
        plt.figlegend(handles=legend_elements, loc='lower center', ncol=3, frameon=True, fontsize='medium')
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust to make room for legend and title

        plt.show()

def test_diff_decoding(input: str, output:str, diff:str, dtype: str = "<u2",):
    print()
    channel_map = {
        0 : 'VIS',
        1 : 'NIR1',
        2 : 'NIR2'
    }
    shape_map = {
        0 : (1024, 1024),
        1 : (518, 648),
        2 : (518, 648),
    }

    input = Path(input)
    diff = Path(diff)

    with open(diff, 'r', encoding='utf-8') as f:
        diff_data = json.load(f)
    diff_offsets = {}

    for ch_id_str, value_dict in diff_data.items():
        ch_id = int(ch_id_str)
        ch_name = channel_map.get(ch_id)
        
        offsets = [value_dict[k] for k in sorted(value_dict, key=int)]
        diff_offsets[ch_name] = offsets
    
    pprint(diff_offsets)

    # pattern = re.compile(r"dc_(\d)_exp_(\d{3})_diffEnc\.bin$")
    pattern = re.compile(r"^dc_(\d)_exp_(\d{3})")

    files_by_channel: dict[int, list[tuple[int, Path]]] = defaultdict(list)
    for file in input.iterdir():
        if not file.suffix == '.bin':
            continue
        m = pattern.match(file.name)
        if not m:
            print(f'Skipping unmatched file:', file.name)
        channel_id = int(m.group(1))
        frame_id = int(m.group(2))
        files_by_channel[channel_id].append((frame_id, file))

    print('Forming cubes')
    cubes: dict[str, list[np.ndarray]] = {}
    for ch_id, paths in files_by_channel.items():
        paths.sort(key=lambda t: t[0])
        arrays = [
            np.fromfile(path, dtype=dtype).reshape(shape_map[ch_id])
            for _, path in paths
        ]
        channel_name = channel_map.get(ch_id)
        cubes[channel_name] = np.stack(arrays, axis=0)

    # stacked_cubes = {key: np.stack(array_list, axis=0) for key, array_list in cubes.items()}

    for key, cube in cubes.items():
        print(f'Decoding {key}')
        print(f'shape of the cube: {cube.shape}')
        decoded_cube = utilities.diff_decode(cube, diff_offsets.get(key), output, key)
        print(f'decoded cube shape: {decoded_cube.shape}')

    print(f'all files decoded')

def try_read_cds():
    readBinfile(decompressed , 'NIR')

    with fits.open(aspect_fly_fits_nir1) as hdul:
        bintable = hdul[2].data
        print()
        print(utilities.read_cds(bintable['NIR1_0'][0], 0, 0 ,1))
        print(utilities.read_cds(bintable['NIR1_0'][0], 3, 150 ,1))
        print(utilities.read_cds(bintable['NIR1_0'][0], 8, 1 ,1))
        print(utilities.read_cds(bintable['NIR1_0'][0], 346, 7 ,1))
        print(utilities.read_cds(bintable['NIR1_0'][0], 4, 100 ,5))
        print(utilities.read_cds(bintable['NIR1_0'][0], 100, 0 ,4))

 
"""
Function calls after this
"""
# test_acqseq()
# test_channel_frames_names()
# test_primary_metadata()

# test_convert_to_fits(input=autoseq_505, output=fits_output_dir)


# test_spice_metadata()

# read_fits_file(simulated_output_ASP , True)
# visualise_fits(simulated_output_ASP, visualise=True, spect=True)
# update_fits_exposure(simulated_output_vis, 0.01)
# update_fits_wl(simulated_output_vis)
# readBinfile(decompressed , 'NIR')
# read_bin_dir(decoded_binaries)
# print(test_decoding(autoseq_encoded_vis0, autoseq_decoding_ouput, autoseq_decoded_vis0))

# test_diff_decoding(autoseq_505_in,autoseq_505_out,autoseq_505_offsets)
# utilities.rename_bin_files(simulated_dir)

# compare_bin_images(os.path.join(autoseq_dir, 'diff_decoded/503/dc_0_exp_000.bin'), os.path.join(autoseq_dir, 'acqseq_503/acq_000_decompressed/dc_0_exp_000_diffEnc.bin'),(1024, 1024),visualize=False)
# try_read_cds()

# Python3 ASPECT_calibration_pipeline/levels_012/test_level_012.py







file_a = Path(os.path.join(os.getcwd(), 'test_data/ASPECT_Autoseq_20240809/diff_decoded/505/dc_0_decoded.dat02.img'))
file_b = Path(os.path.join(os.getcwd(), 'test_data/levels_012_test/test_output/ASPECT_DIFF/505/AS0_XXXXXX_240813T145718_0A.fits'))

arr_a = np.fromfile(file_a, dtype='<u2')

with fits.open(file_b) as hdul:
    data = hdul[1].data

arr_b = data[1]

if arr_a.size != arr_b.size:
    raise ValueError(f'Size missmatch: {file_a.name} has {arr_a.size} pixels' 
                        f'but {file_b.name} has {arr_b.size}')

arr_a = arr_a.reshape(1024, 1024)

print(f'values from files')
print(f'(0,0); a: {arr_a[0][0]} b: {arr_b[0][0]} ')
print(f'(500,500); a: {arr_a[500][500]} b: {arr_b[500][500]} ')

identical = np.array_equal(arr_a, arr_b)
print(f"{'✔️  IDENTICAL' if identical else '❌  DIFFERENT'}"
        f'f- {file_a.name} vd {file_b.name}')