import cv2
import os
import levels_012.modules.utilities as utilities
import levels_012.modules.convertToFits as convertToFits
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import re
import math
from matplotlib.patches import Patch
from pprint import pprint
from collections import defaultdict
from typing import List, Union
import level_3.mgm as mgm
from level_3.test_utilities import show_mgm_figures
import level_3.level_3_utilities as level_3_utilities



# ASPECT DIFF encoded images
autoseq_dir = os.path.join(os.getcwd(), 'test_data/ASPECT_Autoseq_20240809') # FOLDER
diff_decoded = os.path.join(autoseq_dir, 'diff_decoded') # Diff_decoded folder

diff_output_dir = os.path.join(os.getcwd(), 'test_data/test_output/ASPECT_DIFF')

# SIMULATED ASPECT images
simulated_dir = os.path.join(os.getcwd(), 'test_data/ASPECT_simulated_images/2027-03-23_06_00_00-McEwen/acq_000')
simulated_vis = os.path.join(simulated_dir, 'dc_0_exp_000.bin')
simulated_nir1 = os.path.join(simulated_dir, 'dc_1_exp_000.bin')

simulated_output_dir = os.path.join(os.getcwd(), 'pipeline_results/ASPECT_simulated_20270323_McEwen')
simulated_output_vis = os.path.join(simulated_output_dir, 'AS0_000000_270323T060000_1B.fits')
simulated_output_nir1 = os.path.join(simulated_output_dir, 'AS1_000000_270323T060000_1B.fits')
simulated_output_nir2 = os.path.join(simulated_output_dir, 'AS2_000000_270323T060000_1B.fits')
simulated_output_ASP = os.path.join(simulated_output_dir, 'ASP_000000_270323T060000_2B.fits')

# simulated_cube = os.path(os.getcwd())

mgm_test = os.path.join(os.getcwd(), 'test_data/mgm_test_spectra/DataSet1_nm.txt')
mgm_didymos = os.path.join(os.getcwd(), 'test_data/mgm_test_spectra/didymos_spectra.txt')

invalid_test = os.path.join(os.getcwd(), 'test_data/test_output/ASPECT_simulated/2027-03-23_06_00_00/example-3/AS0_XXXXXX_270323T060000_1B.fits')

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

def read_fits_file(path, visualise = False):
    with fits.open(path) as hdul:
        print(f'Fitsfile from path:')
        print({path})
        print(f'lenght of hdul: {len(hdul)}')

        for i, hdu in enumerate(hdul):

            h = hdu.header
            print(f'Header for HDU {i}')
            print(repr(h))

            # if isinstance(hdu, fits.ImageHDU):
                # print(f"Min: {hdu.data[11].min()}")
                # print(f"Max: {hdu.data[11].max()}")
                # print(f'data[4][250][250]: {hdu.data[4][250][250]}')
                

            if visualise:
                if isinstance(hdu, fits.ImageHDU):
                    print("→ This is an ImageHDU")
                    print(f'data type: {type(hdu.data)}')
                    for i, frame in enumerate(hdu.data):
                        print(f'frame: {i}')
                        print(f"Min: {frame.min()}")
                        print(f"Max: {frame.max()}")
                        print(f'points from frame {i}')
                        print(f'(250, 250): {frame[250][250]}')
                        print(f'(10, 150): {frame[10][150]}')
                        print(f'(300, 300): {frame[300][300]}')
                        if i % 5 == 0:
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

def visualise_fits(fitsPath, visualise:bool = True, spect:bool = True):
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

                    vis_wl = [int(w.strip()) for w in header['0_WL'].split(',')]
                    nir1_wl = [int(w.strip()) for w in header['1_WL'].split(',')]
                    nir2_wl = [int(w.strip()) for w in header['2_WL'].split(',')]
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

                    plt.figure(figsize=(10, 5))
                    plt.plot(all_wavelengths, spectra, 'ro-', label="Spectra")
                    plt.xlabel("Wavelength (nm)")
                    plt.ylabel("values")
                    plt.title(f"Spectra from ({x}, {y})")
                    plt.legend()
                    plt.show()

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
                    plt.show()


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
                    plt.figure(figsize=(cols * 4, rows * 4))
                    plt.suptitle('2D Slices from Data Cube', fontsize=16)
                    for i in range(naxis3):
                        plt.subplot(rows, cols, i + 1)
                        plt.imshow(data[i, :, :], cmap='gray')
                        plt.title(f'Slice {i + 1}')
                        plt.axis('off')

                    plt.tight_layout()
                    plt.show()

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
    else:
        with open(filePath, 'rb') as file:
            binaryData = file.read()
            print(f"Read {len(binaryData)} bytes")
            imageArray = np.frombuffer(binaryData, dtype='>u2')
            print(imageArray)
            return

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

            imageArray = np.frombuffer(binaryData, dtype='<u2')
            imageArray = imageArray.reshape((height, width))
            print(f"Min: {imageArray.min()}")
            print(f"Max: {imageArray.max()}")

            print(f'(254,255): {imageArray[255][254]}')
            print(f'(3,150): {imageArray[3][150]}')
            print(f'(8,1): {imageArray[8][1]}')
            print(f'(346,647): {imageArray[346][647]}')
            # print(f'(4,100) - (4, 105): {imageArray[4][100:105]}')
            # print(f'(100,0) - (100, 4): {imageArray[100][:4]}')
            print(f'array type: {imageArray.dtype}')
            plt.figure(figsize=(8,5))
            plt.imshow(imageArray, cmap='gray')
            plt.title(f'channel {channel} little_endian')
            plt.axis('off')
            plt.tight_layout()
            plt.show()


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
    is_fits: bool,
    index: int,
    shape: tuple[int, int],
    dtype: str = "<u2",          # little-endian uint16 by default
    visualize: bool = True
) -> bool:
    """
    Compare two files, file_a is a bin file and file_b can be bin or fits
    """
    file_a = Path(file_a)
    file_b = Path(file_b)

    arr_a = np.fromfile(file_a, dtype=dtype)

    if is_fits:
        with fits.open(file_b) as hdul:
            data = hdul[1].data
            arr_b = data[index]
    else:
        arr_b = np.fromfile(file_b, dtype=dtype)
        arr_b = arr_b.reshape(shape)

    arr_a = arr_a.reshape(shape)
    if arr_a.size != arr_b.size:
        raise ValueError(f'Size missmatch: {file_a.name} has {arr_a.size} pixels' 
                         f'but {file_b.name} has {arr_b.size}')
    

    identical = np.array_equal(arr_a, arr_b)
    print(f"{'✔️  IDENTICAL' if identical else '❌  DIFFERENT'}"
          f'f- {file_a.name} vs {file_b.name}')
    print(f'{arr_a[10][100]} == {arr_b[10][100]}')
    print(f'{type(arr_a[10][100])} == {type(arr_b[10][100])}')
    
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

def try_read_cds(bin, fits):
    readBinfile(bin , 'NIR')

    with fits.open(fits) as hdul:
        bintable = hdul[2].data
        print()
        print(utilities.read_cds(bintable['NIR1_0'][0], 0, 0 ,1))
        print(utilities.read_cds(bintable['NIR1_0'][0], 3, 150 ,1))
        print(utilities.read_cds(bintable['NIR1_0'][0], 8, 1 ,1))
        print(utilities.read_cds(bintable['NIR1_0'][0], 346, 7 ,1))
        print(utilities.read_cds(bintable['NIR1_0'][0], 4, 100 ,5))
        print(utilities.read_cds(bintable['NIR1_0'][0], 100, 0 ,4))

def replace_header_values_with_unk(fits_path: Union[str, os.PathLike], keys_to_replace: List[str]) -> None:
    """
    Replaces specified header values with 'UNK' in all HDUs of a FITS file,
    while preserving the original comments. Modifies the file in-place.

    Parameters:
    - fits_path: Path to the input FITS file.
    - keys_to_replace: List of header keys to update with value 'UNK'.
    """
    print(Path(fits_path).name)
    with fits.open(fits_path, mode='update') as hdul:
        for hdu in hdul:
            header = hdu.header
            for key in keys_to_replace:
                if key in header:
                    comment = header.comments[key]
                    header[key] = ('N/A', comment)
        hdul.flush()  # Write changes to disk

def replace_header_value_with_custom(fits_path: Union[str, os.PathLike], key_to_replace: str, value, comment: Union[None, str]) -> None:
    with fits.open(fits_path, mode='update') as hdul:
        for hdu in hdul:
            header = hdu.header
            if key_to_replace in header:
                if comment == None:
                    comment = header.comments[key_to_replace]
                header[key_to_replace] = (value, comment)
        hdul.flush()

def inspect_pipeline_results(asp: str, as0: str, as1: str, vis_bin: str, nir_bin: str):
    """
    Function to inspect the results of the pipeline.
    
    Parameters: 
        asp: path to a fits file of level 2. 
        as0: path to a VIS channel fits file of level 0 or 1.
        as1: path to a NIR channel fits file of level 0 or 1.
        vis_bin: path to a VIS channel binary file (decompressed and decoded)
        nir_bin: path to a NIR channel binary file (decompressed and decoded)
    """
    
    asp = Path(asp)
    as0 = Path(as0)
    as1 = Path(as1)
    vis_bin = Path(vis_bin)
    nir_bin = Path(nir_bin)

    print('Inspecting the results of ASPECT pipeline with files')
    print(f'Raw VIS binary: {vis_bin.name}')
    print(f'Raw NIR binary: {nir_bin.name}')
    print(f'FITS VIS: {as0.name}')
    print(f'FITS NIR: {as1.name}')
    print(f'Combined FITS: {asp.name}')

    try:
        with open(vis_bin, 'rb') as vis_bin_file, open(nir_bin, 'rb') as nir_bin_file:
            vis_bin_data = vis_bin_file.read()
            nir_bin_data = nir_bin_file.read()
            vis_bin_array = np.frombuffer(vis_bin_data, dtype='<u2')
            nir_bin_array = np.frombuffer(nir_bin_data, dtype='<u2')
            vis_bin_img = vis_bin_array.reshape(1024, 1024)
            nir_bin_img = nir_bin_array.reshape(512, 640)
        with fits.open(as0) as as0_hdul, fits.open(as1) as as1_hdul, fits.open(asp) as asp_hdul:
            as0_data = as0_hdul[1].data
            as1_data = as1_hdul[1].data
            asp_data = asp_hdul[1].data
    except Exception as e:
        print(f'Error occured reading the files: {e}')

    print(f'Comparing files')
    def compare_pixels(img1, img2, x, y):
        if img1[x][y] == img2[0][x][y]:
            print(f'pixel at [{x}][{y}]: {img1[x][y]} == {img2[0][x][y]} ✅')
        else:
            print(f'pixel at [{x}][{y}]: {img1[x][y]} != {img2[0][x][y]} ❌')

    print('VIS')
    compare_pixels(vis_bin_img, as0_data, 0, 0)
    compare_pixels(vis_bin_img, as0_data, 500, 500)
    compare_pixels(vis_bin_img, as0_data, 350, 750)

    print(f'NIR')
    compare_pixels(nir_bin_img, as1_data, 0, 0)
    compare_pixels(nir_bin_img, as1_data, 250, 250)
    compare_pixels(nir_bin_img, as1_data, 350, 150)

    print(f'Visualising alignment')
    vis = as0_data[0]
    nir = as1_data[0]

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(vis, cmap='gray')
    plt.title(f'VIS Frame_0')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(nir, cmap='gray')
    plt.title(f'NIR1 Frame_0')
    plt.axis('off')
    plt.suptitle('Comparison of VIS and NIR Channels') 
    plt.show()

    # Step 1: Edge detection
    edges1 = utilities.laplacian(vis)
    edges2 = utilities.laplacian(nir)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(edges1, cmap='gray')
    plt.title(f'VIS')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(edges2, cmap='gray')
    plt.title(f'NIR1')
    plt.axis('off')
    plt.suptitle('Laplacian edges') 
    plt.show()

    # Step 2: Feature detection using ORB
    orb = cv2.ORB_create(nfeatures=2000) # create ORB feature detector
    # keypoints and binary descriptions
    keypoints1, descriptors1 = orb.detectAndCompute(edges1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(edges2, None)
    # Draw keypoints on each image
    image1_with_kp = cv2.drawKeypoints(edges1, keypoints1, None, color=(0, 255, 0), flags=0)
    image2_with_kp = cv2.drawKeypoints(edges2, keypoints2, None, color=(0, 255, 0), flags=0)

    # Display using matplotlib
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image1_with_kp, cmap='gray')
    plt.title(f'ORB Keypoints ({len(keypoints1)}) - VIS')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image2_with_kp, cmap='gray')
    plt.title(f'ORB Keypoints ({len(keypoints2)}) - NIR')
    plt.axis('off')
    plt.suptitle('ORB Feature Keypoints')
    plt.show()

    index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                    table_number=30,  # Number of hash tables
                    key_size=20,     # Size of the key
                    multi_probe_level=2)  # Number of probes
        
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params) # Initialize the FLANN
    flann_matches = flann.knnMatch(descriptors1, descriptors2, k=2) # Match features
    print(f'FLANN matches before filtering: {len(flann_matches)}')
    matches = utilities.filter_by_distance(flann_matches)
    print(f'FLANN matches after filtering: {len(matches)}')
    N = 500
    matches_to_draw = matches[:N]
    # Draw matches on combined image
    matched_img = cv2.drawMatches(
        edges1, keypoints1,
        edges2, keypoints2,
        matches_to_draw,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    # Show the match visualization
    plt.figure(figsize=(15, 8))
    plt.imshow(matched_img)
    plt.title(f'{N} FLANN feature matches')
    plt.axis('off')
    plt.show()

    print('Visualising the results of alignment')
    legend_elements = [
        Patch(facecolor='yellow', edgecolor='black', label='Aligned regions'),
        Patch(facecolor='red', edgecolor='black', label='Only in vis image'),
        Patch(facecolor='green', edgecolor='black', label='Only in nir image')
    ]
    overlay = utilities.overlay_images(asp_data[0], asp_data[13])
    plt.figure()
    plt.suptitle('Vis and Nir frame overlay', fontsize=16)
    plt.imshow(overlay)
    plt.axis('off')      
    plt.figlegend(handles=legend_elements, loc='lower center', ncol=3, frameon=True, fontsize='medium')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust to make room for legend and title
    plt.show()

def insert_header_entry(fits_path: Union[str, os.PathLike], key: str, value: Union[str, float, int], comment: str = "", index: int = None, hdu_index: int = 0) -> None:
    """
    Insert a new header entry into a FITS file at a specified position.

    Parameters:
        fits_path: Path to the FITS file (modified in-place)
        key: Header keyword to add
        value: Value to associate with the keyword
        comment: Optional comment to describe the keyword
        index: Optional index to insert at (0-based). Appends if None or out of bounds.
        hdu_index: HDU to modify (default is 0 = primary HDU)
    """

    with fits.open(fits_path, mode='update') as hdul:
        header = hdul[hdu_index].header
        card = (key, value, comment)

        if index is None or index >= len(header):
            header[key] = value, comment
        else:
            header.insert(index, card)
        hdul.flush()

def remove_header_entries(fits_path: Union[str, os.PathLike], keys: List[str], hdu_index: int = 0 ) -> None:
    """
    Remove a specific header key from a FITS file.

    Parameters: 
        fits_path: Path to the FITS file
        keys: List of keys to be removed
        hdu_index: The HDU index where the header entry is located
    """

    with fits.open(fits_path, mode='update') as hdul:
        header = hdul[hdu_index].header
        for key in keys:
            if key in header:
                del header[key]
        hdul.flush()
        
def test_mgm(data):
    # strength, center, std
    # initGuess = [[0.3, 0.94, 0.11], [0.5, 1.12, 0.11], [0.3, 1.32, 0.09], [0.2, 2.14, 0.19]]
    initGuess = [[0.3, 940, 110], [0.5, 1120, 110], [0.3, 1320, 90], [0.2, 2140, 190]]
    # initGuess = [[0.67, 950, 100],[0.11, 1210, 80], [0.57, 1950, 270]]
    with open(data, 'r') as f:
        dat = np.loadtxt(f)
    # RMS, band parameters, continuum parameters, continuum parameters P-values
    result = mgm.fit(dat, initGuess, contLinDeg=0, eps=0.1)
    print(f'mgm results')
    print(result)
    figure = mgm.plot(dat, result)
    show_mgm_figures(figure)

def create_2d_fits(fits_path, index, output_dir, name):

    with fits.open(fits_path) as hdul:
        primary_header = hdul[0].header
        image_header = hdul[1].header
        frame = hdul[1].data[index]

    primary_hdu = fits.PrimaryHDU(None,primary_header)
    image_hdu = fits.ImageHDU(frame, image_header)

    new_hdul = fits.HDUList([primary_hdu, image_hdu])
    file_name = os.path.join(output_dir, name)
    new_hdul.writeto(file_name, overwrite=True)

"""
Function calls after this
"""
# test_acqseq()
# test_channel_frames_names()
# test_primary_metadata()

# test_convert_to_fits(input=autoseq_505, output=fits_output_dir)
# test_mgm(mgm_didymos102

# test_spice_metadata()

# replace_header_values_with_unk(os.path.join(os.getcwd(), 'pipeline_results/ASPECT_simulated_20270323_McEwen/AS0_000000_270323T060000_1B.fits'), ['0_SP1', '0_SP2', '0_SP3', '1_SP1', '1_SP2', '1_SP3', '2_SP1', '2_SP2', '2_SP3', '0_ORDER', '1_ORDER', '2_ORDER'])
# replace_header_value_with_custom(os.path.join(os.getcwd(), 'pipeline_results/ASPECT_simulated_20270323_McEwen/AS0_000000_270323T060000_1B.fits'), '0_CCDTMP', 'N/A', "Detector temp [K] ('N/A' [C])")
# replace_header_value_with_custom(os.path.join(os.getcwd(), 'pipeline_results/ASPECT_simulated_20270323_McEwen/AS0_000000_270323T060000_1B.fits'), '0_FPI1', 'N/A', "FPI1 temp [K] ('N/A' [C])")
# replace_header_value_with_custom(os.path.join(os.getcwd(), 'pipeline_results/ASPECT_simulated_20270323_McEwen/AS0_000000_270323T060000_1B.fits'), '0_FPI2', 'N/A', "FPI2 temp [K] ('N/A' [C])")

# replace_header_value_with_custom(os.path.join(os.getcwd(), 'pipeline_results/ASPECT_simulated_20270323_McEwen/AS0_000000_270323T060000_0A.fits'), '0_CCDTMP', 'N/A', "Detector temp [DNs] ")
# replace_header_value_with_custom(os.path.join(os.getcwd(), 'pipeline_results/ASPECT_simulated_20270323_McEwen/AS0_000000_270323T060000_0A.fits'), '0_FPI1', 'N/A', "FPI temperature 1")
# replace_header_value_with_custom(os.path.join(os.getcwd(), 'pipeline_results/ASPECT_simulated_20270323_McEwen/AS0_000000_270323T060000_0A.fits'), '0_FPI2', 'N/A', "FPI temperature 2")

# replace_header_value_with_custom(os.path.join(os.getcwd(), 'pipeline_results/ASPECT_simulated_20270323_McEwen/AS0_000000_270323T060000_1B.fits'), 'OBSERVPH', 'simulated', None)

# remove_header_entries(os.path.join(os.getcwd(), 'pipeline_results/ASPECT_simulated_20270323_McEwen/ASP_000000_270323T060000_2B_705nm.fits'),['1_SP1', '1_SP2', '1_SP3', '2_SP1', '2_SP2', '2_SP3', '1_ORDER', '2_ORDER', '1_EXPOS', '2_EXPOS', '1_FPI1', '1_FPI2', '2_FPI1', '2_FPI2', '1_WL', '2_WL', '1_CCDTMP', '2_CCDTMP', '1_FRAMES', '2_FRAMES'], 1)


# read_fits_file(os.path.join(os.getcwd(), 'pipeline_results/ASPECT_in-flight-dark_250225/104/ASP_000000_200101T014800_2B.fits'), False)

# read_fits_file(os.path.join(os.getcwd(), 'test_data/AFC/AF1_0EGDH4_250401T103140_1B.fits'), False)
visualise_fits(simulated_output_ASP, visualise=True, spect=True)
# update_fits_exposure(simulated_output_vis, 0.01)
# update_fits_wl(simulated_output_vis)

# readBinfile(os.path.join(os.getcwd(), 'test_data/ASPECT_in-flight-dark_250225/acqseq_104/acq_000_decompressed/dc_2_exp_000.bin'), 'NIR')

# read_bin_dir(decoded_binaries)
# print(test_decoding(autoseq_encoded_vis0, autoseq_decoding_ouput, autoseq_decoded_vis0))

# test_diff_decoding(autoseq_505_in,autoseq_505_out,autoseq_505_offsets)
# utilities.rename_bin_files(simulated_dir)

# file_a = os.path.join(os.getcwd(), 'test_data/ASPECT_autoseq_20240809/2027-03-23_06_00_00-McEwen/acq_000/dc_1_exp_000.bin')
# file_a = os.path.join(autoseq_dir, 'acqseq_501/acq_000_diff_decoded/VIS_decoded_003.bin')
file_a = os.path.join(os.getcwd(), 'test_data/ASPECT_in-flight-dark_250225/acqseq_102/acq_000_decompressed/dc_0_exp_001.bin')
file_b = os.path.join(os.path.join(os.getcwd(), 'pipeline_results/ASPECT_in-flight-dark_250225/102/AS0_000000_200101T014553_1B.fits'))
# compare_bin_images(file_a, file_b, True, 1, (1024, 1024), visualize=True)

# create_2d_fits(os.path.join(os.getcwd(), 'pipeline_results/ASPECT_simulated_20270323_McEwen/ASP_000000_270323T060000_2B.fits'), 2, os.path.join(os.getcwd(), 'pipeline_results/ASPECT_simulated_20270323_McEwen'),'ASP_000000_270323T060000_2B_705nm.fits')

# try_read_cds(os.path.join(os.getcwd(), 'test_data/ASPECT_in-flight-dark_250225/acqseq_104/acq_000_decompressed/dc_1_exp_000.bin'), os.path.join(os.getcwd(), 'pipeline_results/ASPECT_in-flight-dark_250225/104/ASP_000000_200101T014800_2B.fits'))
# inspect_pipeline_results(simulated_output_ASP, simulated_output_vis, simulated_output_nir1, simulated_vis, simulated_nir1)


# test_mgm(mgm_test)

""" 
Python3 ASPECT_calibration_pipeline/test_level_012.py
"""



