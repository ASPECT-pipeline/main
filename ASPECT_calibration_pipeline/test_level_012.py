import cv2
import os
import io
import levels_012.modules.utilities as utilities
import levels_012.modules.convertToFits as convertToFits
import levels_012.modules.badPixels as badpixels
import levels_012.modules.darkSubtraction as darksubtraction
import levels_012.modules.flatField as flatfield
import levels_012.modules.extractCDS as extractCDS
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
import xarray as xr
import cftime 
from datetime import datetime
from functools import lru_cache
from levels_012.modules.reflectance import reflectance_calibration
import pandas as pd
from scipy.io import loadmat
from level_3.modules.BAR_BC_method import calc_band_parameters
import sys
from config import initGuess, reverse_channel_map

def read_fits_file(path, visualise = False):
    with fits.open(path) as hdul:
        print(f'Fitsfile from path:')
        print({path})
        print(f'lenght of hdul: {len(hdul)}')

        for i, hdu in enumerate(hdul):

            h = hdu.header
            print(repr(h))

            print(hdu.shape)

            print(f'data shape: {hdu.data.shape}')

            if visualise:
                data = hdu.data

                if data.ndim == 3:
                    # Iterate over frames in a data cube
                    for frame_idx, img in enumerate(data):
                        plt.figure()
                        plt.imshow(img, cmap='gray')
                        plt.title(f'HDU {i} - Slice {frame_idx}')
                        plt.axis('off')
                        plt.tight_layout()
                        plt.show()
                        ans = input("Press Enter to see next, or 'q' to quit: ").strip().lower()
                        if ans == 'q':
                            break
                elif data.ndim == 2:
                    # Display single 2D image
                    plt.figure()
                    plt.imshow(data, cmap='gray')
                    plt.title(f'2D Image')
                    plt.axis('off')
                    plt.tight_layout()
                    plt.show()                

def readBinfile(filePath, channel):
    if channel == "Vis": 
        height = 1024
        width = 1024
    elif channel == "NIR":
        height = 518
        width = 648
    elif channel == 'SIMULATED':
        height = 512
        width = 640
    else:
        with open(filePath, 'rb') as file:
            binaryData = file.read()
            print(f"Read {len(binaryData)} bytes")
            imageArray = np.frombuffer(binaryData, dtype='>u2')
            print(imageArray)
            return

    # print(f"height: {height} \nwidht: {width}")

    # bytes_per_pixel, bit_depth = utilities.estimate_bit_depth(filePath, width, height)
    # print(f'{channel} bytes per pixel: {bytes_per_pixel}')
    # print(f'{channel} bit depth: {bit_depth}')
    # effective_max = 2**bit_depth - 1
    # print(f'effective max: {effective_max}')
    

    try:
        with open(filePath, 'rb') as file:
            binaryData = file.read()
            print(f"Read {len(binaryData)} bytes")

            imageArray = np.frombuffer(binaryData, dtype='<u2')
            imageArray = imageArray.reshape((height, width))
            # print(f"Min: {imageArray.min()}")
            # print(f"Max: {imageArray.max()}")
            print(imageArray[0][:5])
            #Visualise
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

def update_fits_exposure(path, new_exposure, save_as=None):
    with fits.open(path, mode='update' if save_as is None else 'readonly') as hdul:
        header = hdul[0].header
        channel = header.get('CHANNEL')
        channel_id = reverse_channel_map.get(channel)
        exposure_str = f'{channel_id}_EXPOS'
        for hdu in hdul:
            if exposure_str in hdu.header:
                print(f"Old EXPOSURE: {hdu.header[exposure_str]}")
                hdu.header[exposure_str] = new_exposure
                print(f"New EXPOSURE: {hdu.header[exposure_str]}")

        if save_as:
            hdul.writeto(save_as, overwrite=True)
            print(f"Saved updated file to {save_as}")
        else:
            print(f"Updated in place: {path}")

def update_fits_wl(path, save_as=None):
    wl_map = {
        '0_WL' : '675,690,705,720,735,750,765,780,795,810,825',
        '1_WL': '875,904,933,963,992,1021,1050,1079,1108,1138,1167,1196,1225',
        '2_WL': '1225,1254,1283,1313,1342,1371,1400,1429,1458,1488,1517,1546,1575',
        '3_WL': '1675,1711,1748,1784,1820,1857,1893,1930,1966,2002,2075,2111,2148,2184,2220,2257,2293,2330,2366,2402,2439,2475'
    }
    with fits.open(path, mode='update' if save_as is None else 'readonly') as hdul:
        hdu = hdul[0]
        for key in list(wl_map.keys()):
            if key in hdu.header:
                print(f"Old WL: {hdu.header[key]}")
                hdu.header[key] = wl_map[key]
                print(f"New WL: {hdu.header[key]}")

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
            data = hdul[0].data
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

def try_read_cds(bin, fits_file):
    readBinfile(bin , 'NIR')

    with fits.open(fits_file) as hdul:
        bintable = hdul[1].data
        print()
        print(utilities.read_cds(bintable['1_000'][0], 0, 0 ,1))
        print(utilities.read_cds(bintable['1_000'][0], 3, 150 ,1))
        print(utilities.read_cds(bintable['1_000'][0], 8, 1 ,1))
        print(utilities.read_cds(bintable['1_000'][0], 346, 7 ,1))
        print(utilities.read_cds(bintable['1_000'][0], 4, 100 ,5))
        print(utilities.read_cds(bintable['1_000'][0], 100, 0 ,4))

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
            as0_data = as0_hdul[0].data
            as1_data = as1_hdul[0].data
            asp_data = asp_hdul[0].data
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

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(asp_data[0], cmap='gray')
    plt.title(f'Aligned VIS')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(asp_data[12], cmap='gray')
    plt.title(f' NIR')
    plt.axis('off')
    plt.suptitle('Images after Alignment')
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

def inspect_npz(path, *, show_stats=True, max_list_items=5):
    """
    Print a readable summary of an .npz archive.
    - Unwraps 0-D object arrays (common when dicts are pickled into .npz).
    - Recurses into dicts/lists/tuples and shows array shapes/dtypes.
    - For likely transmission bundles (e.g., keys like 'wavelengths', 'transmissions'),
      prints quick stats (length, min/max).
    """
    path = Path(path)

    def is_array(x): return isinstance(x, np.ndarray)
    def is_scalar_object_array(x): return is_array(x) and x.dtype == object and x.shape == ()
    def safe_item(x):
        try:
            return x.item()
        except Exception:
            return x

    def arr_stats(a):
        if not show_stats or not np.issubdtype(a.dtype, np.number) or a.size == 0:
            return ""
        try:
            amin = np.nanmin(a)
            amax = np.nanmax(a)
            return f" (min={amin:g}, max={amax:g})"
        except Exception:
            return ""

    def print_kv(k, v, indent=2):
        pad = " " * indent

        # Unwrap 0-D object arrays (often a dict)
        if is_scalar_object_array(v):
            v = safe_item(v)

        # Numpy array
        if is_array(v):
            print(f"{pad}- {k}: ndarray shape={v.shape}, dtype={v.dtype}{arr_stats(v)}")
            return

        # Dict-like
        if isinstance(v, dict):
            print(f"{pad}- {k}: dict with {len(v)} key(s)")
            # Special: look for common spectrum keys
            for sk in sorted(v.keys()):
                sv = v[sk]
                if is_scalar_object_array(sv):
                    sv = safe_item(sv)
                if is_array(sv):
                    extra = arr_stats(sv)
                    print(f"{pad}    • {sk}: ndarray shape={sv.shape}, dtype={sv.dtype}{extra}")
                else:
                    # brief preview for scalars/strings/other
                    preview = sv
                    if isinstance(sv, (list, tuple)):
                        preview = f"{type(sv).__name__}[{len(sv)}]"
                    elif isinstance(sv, (str, bytes)):
                        preview = f"{type(sv).__name__}(len={len(sv)})"
                    print(f"{pad}    • {sk}: {preview}")
            return

        # List/tuple: show a few items
        if isinstance(v, (list, tuple)):
            print(f"{pad}- {k}: {type(v).__name__} with {len(v)} item(s)")
            for i, item in enumerate(v[:max_list_items]):
                it = safe_item(item) if is_scalar_object_array(item) else item
                if is_array(it):
                    print(f"{pad}    [{i}]: ndarray shape={it.shape}, dtype={it.dtype}{arr_stats(it)}")
                elif isinstance(it, dict):
                    print(f"{pad}    [{i}]: dict with {len(it)} key(s)")
                else:
                    t = type(it).__name__
                    s = f"{t}(len={len(it)})" if isinstance(it, (str, bytes)) else t
                    print(f"{pad}    [{i}]: {s}")
            if len(v) > max_list_items:
                print(f"{pad}    … ({len(v)-max_list_items} more)")
            return

        # Fallback: plain scalar / other object
        t = type(v).__name__
        if isinstance(v, (str, bytes)):
            print(f"{pad}- {k}: {t}(len={len(v)})")
        else:
            print(f"{pad}- {k}: {t}")

    with np.load(path, allow_pickle=True) as z:
        print(f"{path} — {len(z.files)} item(s)")
        # top-level keys
        for k in z.files:
            v = z[k]
            # show top-level summary and recurse one level
            if is_scalar_object_array(v):
                v = safe_item(v)

            if is_array(v):
                print(f"  {k}: ndarray shape={v.shape}, dtype={v.dtype}{arr_stats(v)}")
            elif isinstance(v, dict):
                print(f"  {k}: dict with {len(v)} key(s)")
                # print inner keys
                for sk in sorted(v.keys()):
                    print_kv(sk, v[sk], indent=4)
            elif isinstance(v, (list, tuple)):
                print(f"  {k}: {type(v).__name__} with {len(v)} item(s)")
                for i, item in enumerate(v[:max_list_items]):
                    print_kv(f"[{i}]", item, indent=4)
                if len(v) > max_list_items:
                    print(f"      … ({len(v)-max_list_items} more)")
            else:
                t = type(v).__name__
                if isinstance(v, (str,bytes)):
                    print(f"  {k}: {t}(len={len(v)})")
                else:
                    print(f"  {k}: {t}")

def try_asteroid_mask(path):
    with fits.open(path) as hdul:
        data = hdul[0].data
    combined = level_3_utilities.extract_asteroid(data)

# Rotation 
def as_native_f32(img: np.ndarray) -> np.ndarray:
    if img.dtype.byteorder == '>' or (img.dtype.byteorder == '=' and sys.byteorder == 'big'):
        img = img.byteswap().newbyteorder()
    if img.dtype != np.float32:
        img = img.astype(np.float32, copy=False)
    return np.ascontiguousarray(img)

def estimate_disk(img: np.ndarray):
    """
    Returns (cx, cy, r) in pixels.
    Uses an adaptive threshold + largest contour + minEnclosingCircle.
    """
    a = as_native_f32(img)
    # normalize to 8-bit just for robust thresholding
    u8 = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # gentle blur + Otsu to segment disk from space
    u8b = cv2.GaussianBlur(u8, (0,0), 2.0)
    _, m = cv2.threshold(u8b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # keep largest blob
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        # fallback: whole image center, half-min dimension
        h, w = a.shape
        return (w/2, h/2, min(w, h)/2 * 0.9), (m>0)
    c = max(cnts, key=cv2.contourArea)
    (cx, cy), r = cv2.minEnclosingCircle(c)
    return (cx, cy, r), (m>0)

def normalized_coords(shape, cx, cy, r):
    """
    Returns x,y,r2,mask where x,y in [-1,1] over a square crop 2r x 2r
    centered on (cx,cy). mask marks the unit disk x^2+y^2<=1.
    """
    H, W = shape
    R = int(np.ceil(r))
    # crop bounds (clamped to image)
    x0 = max(0, int(cx - R)); x1 = min(W, int(cx + R))
    y0 = max(0, int(cy - R)); y1 = min(H, int(cy + R))
    hh, ww = y1 - y0, x1 - x0

    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    x = (xx + x0 - cx) / r
    y = (yy + y0 - cy) / r
    r2 = x*x + y*y
    mask = r2 <= 1.0
    return (slice(y0, y1), slice(x0, x1)), x, y, r2, mask

def overlay_limb(img, cx, cy, r, color=(0,255,0)):
    a = cv2.normalize(as_native_f32(img), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    bgr = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    cv2.circle(bgr, (int(round(cx)), int(round(cy))), int(round(r)), color, 1, cv2.LINE_AA)
    return bgr

def yaw_rotate_front_hemisphere(img: np.ndarray, cx: float, cy: float, r: float,
                                yaw_deg: float, limb_darkening: float = 0.4):
    """
    Treat 'img' (float32) as an orthographic photo of the FRONT hemisphere.
    Rotate by yaw_deg (CCW looking from north), and re-project with self-occlusion.
    Returns (crop_original, crop_rotated), both float32, shape ≈ (2R, 2R).
    """
    a = img.astype(np.float32, copy=False)
    H, W = a.shape
    R = int(np.ceil(r))

    # Build a tight crop and normalized coords (x,y) in [-1,1]
    y0, y1 = max(0, int(cy - R)), min(H, int(cy + R))
    x0, x1 = max(0, int(cx - R)), min(W, int(cx + R))
    crop = a[y0:y1, x0:x1]

    hh, ww = crop.shape
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    x = (xx + x0 - cx) / r
    y = (yy + y0 - cy) / r
    r2 = x*x + y*y
    unit = r2 <= 1.0

    # Lift to sphere: front hemisphere z>=0
    z = np.zeros_like(x, dtype=np.float32)
    z[unit] = np.sqrt(np.clip(1.0 - r2[unit], 0.0, 1.0))

    # Output view grid is the same (x,y,z) we just built; compute its lat'/lon'
    latp = np.zeros_like(x); lonp = np.zeros_like(x)
    latp[unit] = np.arcsin(np.clip(y[unit], -1.0, 1.0))
    lonp[unit] = np.arctan2(x[unit], z[unit])  # atan2(X', Z')

    # Undo yaw to find source longitude on the original front hemisphere
    yaw = np.deg2rad(yaw_deg)
    lon = lonp - yaw
    lat = latp

    # Source point on original front hemisphere (X0,Y0,Z0)
    X0 = np.zeros_like(x); Y0 = np.zeros_like(y); Z0 = np.zeros_like(z)
    X0[unit] = np.cos(lat[unit]) * np.sin(lon[unit])
    Y0[unit] = np.sin(lat[unit])
    Z0[unit] = np.cos(lat[unit]) * np.cos(lon[unit])

    # Only sample where the original view actually had front hemisphere
    visible0 = (Z0 >= 0) & unit

    # Map back to source pixel coords in the original image
    u = (X0 * r + cx).astype(np.float32)
    v = (Y0 * r + cy).astype(np.float32)

    # Sample with bilinear interpolation; anything outside becomes 0
    sampled = cv2.remap(a, u, v, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

    # Optional simple limb darkening using current view Z' (which equals z)
    if limb_darkening > 0:
        w = (1 - limb_darkening) + limb_darkening * np.clip(z, 0, 1)
        sampled = sampled * w

    out = np.zeros_like(crop, dtype=np.float32)
    out[visible0] = sampled[visible0]
    # Outside unit disk is 0 by construction; back hemisphere also 0
    return crop, out

def rotate_asteroid(img, angle_deg):
    img = img.astype('<f4', copy=False)
    h, w = img.shape
    center = (w/2, h/2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rotated

# Solar irradiace functions 
def _extract_years(time_da: xr.DataArray) -> np.ndarray:
    """Return an int array of years from an xarray time coordinate, robust to datetime64, cftime, or numeric."""
    # Case A: datetime64 -> use .dt.year
    if np.issubdtype(time_da.dtype, np.datetime64):
        return time_da.dt.year.values.astype(int)

    vals = time_da.values

    # Case B: object array of cftime datetimes -> read .year per element
    if vals.dtype == object and len(vals) and isinstance(vals.flat[0], cftime.datetime):
        return np.array([t.year for t in vals], dtype=int)

    # Case C: numeric times -> decode via units/calendar, then read .year
    units = time_da.attrs.get("units", None)
    cal = time_da.attrs.get("calendar", "standard")
    if units is None:
        # Fall back: just treat as year numbers already
        return np.array(vals, dtype=int)
    decoded = cftime.num2date(vals, units=units, calendar=cal)
    return np.array([t.year for t in decoded], dtype=int)

def solar_irradiance_1au_at_wavelength(
    nc_path: str,
    wl_nm: float,
    year: int | None = None,
    return_uncertainty: bool = True,
):
    """
    Get F_sun(λ) at 1 AU from a NetCDF file, at a specific wavelength (nm).
    Works even when 'time' uses cftime or remains numeric.
    """
    ds = xr.open_dataset(nc_path)

    # Variables
    ssi = ds["SSI"]                  # (time, wavelength), W m^-2 nm^-1
    wl  = ds["wavelength"].astype(float)

    # Mask missing values (-99) if present
    mv = ssi.attrs.get("missing_value", None)
    if mv is not None:
        ssi = ssi.where(ssi != mv)

    # Choose time slice
    if year is None:
        spec = ssi.mean(dim="time", skipna=True)
        idx = None  # not used
    else:
        years_all = _extract_years(ds["time"])
        # Clamp requested year to available range
        y_min, y_max = int(years_all.min()), int(years_all.max())
        y_req = int(np.clip(year, y_min, y_max))
        # Nearest year index
        idx = int(np.argmin(np.abs(years_all - y_req)))
        spec = ssi.isel(time=idx)

    # Interpolate SSI to requested wavelength
    F = spec.interp(wavelength=wl_nm, kwargs={"fill_value": np.nan}).item()

    if not return_uncertainty:
        return F

    # Uncertainty if available
    if "SSI_UNC" in ds:
        unc = ds["SSI_UNC"]
        mv_unc = unc.attrs.get("missing_value", None)
        if mv_unc is not None:
            unc = unc.where(unc != mv_unc)
        unc_spec = unc.mean(dim="time", skipna=True) if year is None else unc.isel(time=idx)
        F_unc = unc_spec.interp(wavelength=wl_nm, kwargs={"fill_value": np.nan}).item()
    else:
        F_unc = np.nan

    return F, F_unc

def _years_from_time(time_da: xr.DataArray) -> np.ndarray:
    """Return integer years robustly (datetime64, cftime, or numeric)."""
    if np.issubdtype(time_da.dtype, np.datetime64):
        return time_da.dt.year.values.astype(int)
    vals = time_da.values
    if vals.dtype == object and vals.size and isinstance(vals.flat[0], cftime.datetime):
        return np.array([t.year for t in vals], dtype=int)
    units = time_da.attrs.get("units", None)
    cal = time_da.attrs.get("calendar", "standard")
    if units is None:
        return np.array(vals, dtype=int)
    decoded = cftime.num2date(vals, units=units, calendar=cal)
    return np.array([t.year for t in decoded], dtype=int)

def load_spectrum(nc_path: str, mode: str = "mean", year: int | None = None):
    """
    Returns (wavelength_nm, Fsun_1AU) where Fsun is in W m^-2 nm^-1.
    mode: "mean" (time-mean) or "nearest_year" (pick nearest to 'year').
    """
    ds = xr.open_dataset(nc_path)
    ssi = ds["SSI"]                     # (time, wavelength), W m^-2 nm^-1
    wl  = ds["wavelength"].astype(float)

    # Mask missing values (-99) if defined
    mv = ssi.attrs.get("missing_value", None)
    if mv is not None:
        ssi = ssi.where(ssi != mv)

    if mode == "mean" or year is None:
        spec = ssi.mean(dim="time", skipna=True)
        label = "Time-mean SSI"
    elif mode == "nearest_year":
        years = _years_from_time(ds["time"])
        idx = int(np.argmin(np.abs(years - int(year))))
        spec = ssi.isel(time=idx)
        label = f"SSI (nearest year: {years[idx]})"
    else:
        raise ValueError("mode must be 'mean' or 'nearest_year'.")

    return wl.values, spec.values, label, ds  # return ds to reuse uncertainty if you want

def plot_ssi(nc_path: str, mode: str = "mean", year: int | None = None,
             wl_range: tuple[float, float] | None = None,
             show_uncertainty: bool = True):
    wl, F, label, ds = load_spectrum(nc_path, mode=mode, year=year)

    # Optional: plot ±1σ band if available
    F_lo = F_hi = None
    if show_uncertainty and "SSI_UNC" in ds:
        unc = ds["SSI_UNC"]
        mv_unc = unc.attrs.get("missing_value", None)
        if mv_unc is not None:
            unc = unc.where(unc != mv_unc)
        if mode == "mean" or year is None:
            unc_spec = unc.mean(dim="time", skipna=True).values
        else:
            years = _years_from_time(ds["time"])
            idx = int(np.argmin(np.abs(years - int(year))))
            unc_spec = unc.isel(time=idx).values
        F_lo = F - unc_spec
        F_hi = F + unc_spec

    # Make the plot
    plt.figure()
    plt.plot(wl, F, linewidth=1)
    if F_lo is not None and F_hi is not None:
        plt.fill_between(wl, F_lo, F_hi, alpha=0.2)

    if wl_range is not None:
        plt.xlim(*wl_range)

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Solar spectral irradiance at 1 AU (W m$^{-2}$ nm$^{-1}$)")
    plt.title(label)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def _open_nc(nc_path: str):
    """Try common xarray engines; raise if none work."""
    last_err = None
    for eng in ("netcdf4", "h5netcdf", "scipy"):
        try:
            return xr.open_dataset(nc_path, engine=eng, decode_times=False)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not open {nc_path} with netcdf4/h5netcdf/scipy: {last_err}")

def _years_from_time(time_da: xr.DataArray) -> np.ndarray:
    """Robustly extract integer years from an xarray time coordinate."""
    # If times are numeric "days since ...", decode with cftime:
    vals = np.asarray(time_da.values)
    units = time_da.attrs.get("units", None)
    cal   = time_da.attrs.get("calendar", "standard")
    if units is not None:
        decoded = cftime.num2date(vals, units=units, calendar=cal)
        return np.array([t.year for t in decoded], dtype=int)
    # Fallback: treat as already-years
    return vals.astype(int)

def make_dense_ssi_table(
    nc_path: str,
    out_csv: str,
    *,
    wl_min: float = 300.0,
    wl_max: float = 2000.0,
    step_nm: float = 1.0,
    mode: str = "mean",          # "mean" or "nearest_year"
    year: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a dense table of SSI at 1 AU, sampled every 'step_nm' from wl_min..wl_max.
    Writes a CSV with columns: wavelength_nm,Fsun_1AU_Wm2_per_nm
    Returns (wl_grid, F_grid).
    """
    ds  = _open_nc(nc_path)
    ssi = ds["SSI"]  # (time, wavelength) in W m^-2 nm^-1

    # Mask file's missing values (-99 or _FillValue)
    mv = ssi.attrs.get("missing_value", ssi.attrs.get("_FillValue", None))
    if mv is not None:
        ssi = ssi.where(ssi != mv)

    # Choose spectrum
    if mode == "mean" or year is None:
        spec = ssi.mean(dim="time", skipna=True)
        picked_year = None
    else:
        years = _years_from_time(ds["time"])
        idx   = int(np.argmin(np.abs(years - int(year))))
        spec  = ssi.isel(time=idx)
        picked_year = int(years[idx])

    wl_src = ds["wavelength"].astype(float).values
    F_src  = spec.values

    # Build dense grid and interpolate
    wl_grid = np.arange(wl_min, wl_max + 1e-12, step_nm, dtype=float)
    F_grid  = np.interp(wl_grid, wl_src, F_src, left=np.nan, right=np.nan)

    # Write CSV with a few metadata lines
    header = [
        f"# source_nc={Path(nc_path).name}",
        f"# mode={mode}",
        f"# year={picked_year if picked_year is not None else 'time-mean'}",
        "# units=Fsun_1AU in W m^-2 nm^-1; wavelength in nm",
        f"# generated_on={datetime.utcnow().isoformat(timespec='seconds')}Z",
        "wavelength_nm,Fsun_1AU_Wm2_per_nm",
    ]
    rows = [f"{w:.6f},{f:.8e}" for w, f in zip(wl_grid, F_grid)]
    Path(out_csv).write_text("\n".join(header + rows))
    return wl_grid, F_grid

def _load_ssi_csv(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load dense SSI table created earlier.
    Ignores lines starting with '#'.
    Returns (wavelength_nm, Fsun_1AU_Wm2_per_nm).
    """
    data = np.loadtxt(csv_path, delimiter=",", comments="#", dtype=float)
    if data.ndim == 1:  # single row case
        data = data[None, :]
    wl = data[:, 0]
    F  = data[:, 1]
    return wl, F

def print_ssi_value(csv_path: str, wl_nm: float, mode: str = "nearest") -> float:
    """
    Read and print SSI(λ) at 1 AU from the CSV for a given wavelength.
    
    Parameters
    ----------
    csv_path : str
        Path to the dense SSI CSV (columns: wavelength_nm,Fsun_1AU_Wm2_per_nm).
    wl_nm : float
        Target wavelength in nm.
    mode : {"nearest","interp","exact"}
        Lookup mode:
          - "nearest": pick closest wavelength in the file (default)
          - "interp" : linear interpolation
          - "exact"  : require exact match (within 1e-9 nm)
    
    Returns
    -------
    float
        SSI at 1 AU in W m^-2 nm^-1 (NaN if not found/invalid).
    """
    wl, F = _load_ssi_csv(csv_path)
    wl_nm = float(wl_nm)

    if mode == "exact":
        idx = np.where(np.isclose(wl, wl_nm, rtol=0, atol=1e-9))[0]
        if idx.size == 0:
            val = np.nan
            print(f"No exact wavelength {wl_nm:g} nm in file.")
        else:
            val = float(F[idx[0]])
            print(f"SSI(1 AU) at {wl_nm:g} nm = {val:.8e} W m^-2 nm^-1  [exact match]")

    elif mode == "interp":
        if wl_nm < wl.min() or wl_nm > wl.max():
            val = np.nan
            print(f"{wl_nm:g} nm is outside the table range [{wl.min():.3f}, {wl.max():.3f}] nm.")
        else:
            val = float(np.interp(wl_nm, wl, F))
            print(f"SSI(1 AU) at {wl_nm:g} nm = {val:.8e} W m^-2 nm^-1  [linear interp]")

    else:  # "nearest"
        idx = int(np.argmin(np.abs(wl - wl_nm)))
        val = float(F[idx])
        print(f"SSI(1 AU) near {wl_nm:g} nm -> {wl[idx]:.6f} nm = {val:.8e} W m^-2 nm^-1  [nearest]")

    return val

def read_pds3_solar_spectrum(path):
    REC_BYTES = 512

    # Pointers (records start at 1)
    rec_idnum = 11
    rec_idname = 12
    rec_ref = 13
    rec_data = 14

    # Layout
    idname_len = 32
    ref_count = 8
    n = 901
    wl_start = 200.0
    wl_step = 1.0

    with open(path, "rb") as f:
        # ID number (int32, little-endian)
        f.seek((rec_idnum - 1) * REC_BYTES)
        idnum = np.fromfile(f, dtype="<i4", count=1)[0]

        # Name (32 bytes, ASCII)
        f.seek((rec_idname - 1) * REC_BYTES)
        name_bytes = np.fromfile(f, dtype="u1", count=idname_len).tobytes()
        idname = name_bytes.decode("ascii", "ignore").rstrip("\x00").strip()

        # Reference array (8 float64)
        f.seek((rec_ref - 1) * REC_BYTES)
        ref = np.fromfile(f, dtype="<f8", count=ref_count)

        # Spectrum data (901 float64)
        f.seek((rec_data - 1) * REC_BYTES)
        y = np.fromfile(f, dtype="<f8", count=n)

    # Wavelength grid in nm (from label)
    wl_nm = wl_start + wl_step * np.arange(n)

    for wl, spec in zip(wl_nm, y):
        print(f"{wl:.1f} : {spec:.6e}")

    return {
        "idnum": int(idnum),
        "idname": idname,
        "ref": ref,          # np.ndarray shape (8,)
        "wavelength_nm": wl_nm,  # shape (901,)
        "irradiance": y      # shape (901,), units not specified in label
    }

def resample_txt_to_1nm_and_print(
    path,
    columns="MCebKur,MChKur",
    wl_start=200.0,
    wl_stop=3000.0,
    method="flux"
):
    """
    Read a whitespace-separated solar spectrum .txt (header on first non-comment line),
    resample to integer-nm grid, print as two columns, and return (bins_nm, spectrum).

    Parameters
    ----------
    path : str
        Input .txt file. Header must include wavelength and requested irradiance columns.
        Wavelength can be in 'nm' or 'CM-1' (wavenumber). If values look like micrometers (<20),
        they are converted to nm by ×1000.
    columns : str
        Comma-separated list of column names to average row-wise (e.g. 'MCebKur,MChKur').
    wl_start, wl_stop : float
        Desired nm range for output grid.
    method : {'center','flux'}
        'center' → linear interpolation at bin centers (integer nm).
        'flux'   → flux-conserving bin average over [c-0.5, c+0.5].

    Returns
    -------
    bins : np.ndarray
        Integer-nm centers actually produced (clipped to data coverage).
    y : np.ndarray
        Resampled irradiance at those centers.
    """
    import math
    import numpy as np

    # ---------- 1) Read header ----------
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        # find header (first non-empty, non-comment line)
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                header = s.split()
                break
        else:
            raise ValueError("No header line found in file.")

        # normalize header tokens
        norm = lambda t: t.strip().replace("-", "_")
        name_to_idx = {norm(h): i for i, h in enumerate(header)}

        # wavelength columns
        nm_idx = name_to_idx.get("nm")
        wn_idx = name_to_idx.get("CM_1", name_to_idx.get("CM-1", None))

        # which irradiance columns to use
        want_cols = [norm(c) for c in columns.split(",") if c.strip()]
        col_indices = []
        missing = []
        for cname in want_cols:
            ix = name_to_idx.get(cname)
            if ix is None:
                missing.append(cname)
            else:
                col_indices.append(ix)
        if not col_indices:
            raise ValueError(f"None of the requested columns were found: {missing}. "
                             f"Available: {list(name_to_idx.keys())}")

        # ---------- 2) Stream rows; collect wavelength+row-average flux ----------
        wl_list, f_list = [], []
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()

            # wavelength → nm
            wl = None
            try:
                if nm_idx is not None and nm_idx < len(parts):
                    wl = float(parts[nm_idx])  # might be nm or µm magnitude
                elif wn_idx is not None and wn_idx < len(parts):
                    wn = float(parts[wn_idx])  # cm^-1
                    wl = 1e7 / wn             # nm
            except ValueError:
                wl = None
            if wl is None:
                continue

            # if it looks like micrometers, convert to nm
            if wl < 20.0:
                wl *= 1000.0

            # row-wise average across requested columns (skip missing/bad cells)
            vals = []
            for ix in col_indices:
                if ix < len(parts):
                    try:
                        vals.append(float(parts[ix]))
                    except ValueError:
                        pass
            if not vals:
                continue

            wl_list.append(wl)
            f_list.append(sum(vals) / len(vals))

    if not wl_list:
        raise ValueError("No valid data rows parsed (check column names and file format).")

    wl = np.asarray(wl_list, dtype=float)
    fx = np.asarray(f_list, dtype=float)

    # ---------- 3) Clean, sort, dedupe ----------
    ok = np.isfinite(wl) & np.isfinite(fx)
    wl, fx = wl[ok], fx[ok]
    order = np.argsort(wl)
    wl, fx = wl[order], fx[order]

    # collapse identical wavelengths by averaging
    if wl.size == 0:
        raise ValueError("No valid finite data after cleaning.")
    uniq, idx = np.unique(wl, return_index=True)
    counts = np.r_[idx[1:], wl.size] - idx
    fx = np.add.reduceat(fx, idx) / counts
    wl = uniq

    # ---------- 4) Build output grid ----------
    if method == "flux":
        # require full coverage of each bin [c-0.5, c+0.5]
        lo = math.ceil(max(wl_start, wl.min() + 0.5))
        hi = math.floor(min(wl_stop,  wl.max() - 0.5))
        if hi < lo:
            raise ValueError(f"Data coverage {wl.min():.3f}..{wl.max():.3f} nm "
                             f"does not support flux bins {wl_start}..{wl_stop} nm.")
        bins = np.arange(float(lo), float(hi) + 1.0, 1.0)

        # ---------- 5a) Flux-conserving bin average ----------
        edges = np.concatenate(([bins[0] - 0.5],
                                0.5 * (bins[:-1] + bins[1:]),
                                [bins[-1] + 0.5]))
        # insert edge samples and integrate piecewise linear curve
        f_edges = np.interp(edges, wl, fx)
        wl_aug  = np.concatenate([wl, edges])
        fx_aug  = np.concatenate([fx, f_edges])
        o = np.argsort(wl_aug)
        wl_aug, fx_aug = wl_aug[o], fx_aug[o]

        dλ   = wl_aug[1:] - wl_aug[:-1]
        trap = 0.5 * (fx_aug[1:] + fx_aug[:-1]) * dλ
        Icum = np.concatenate([[0.0], np.cumsum(trap)])

        I_at_edges = np.interp(edges, wl_aug, Icum)
        rebinned = (I_at_edges[1:] - I_at_edges[:-1]) / 1.0  # per nm

    elif method == "center":
        # only require the center to lie within data range
        lo = math.ceil(max(wl_start, wl.min()))
        hi = math.floor(min(wl_stop,  wl.max()))
        if hi < lo:
            raise ValueError(f"Data coverage {wl.min():.3f}..{wl.max():.3f} nm "
                             f"does not include centers {wl_start}..{wl_stop} nm.")
        bins = np.arange(float(lo), float(hi) + 1.0, 1.0)

        # ---------- 5b) Center sampling (linear interpolation at c) ----------
        rebinned = np.interp(bins, wl, fx)

    else:
        raise ValueError("method must be 'center' or 'flux'.")

    # ---------- 6) Print and return ----------
    # for w, val in zip(bins[:25], rebinned[:25]):
    #     print(f"{w:.1f} : {val:.6e}")

    return bins, rebinned

def compare_resampled_to_pds3(pds3_source, bins_nm, y_resampled):
    """
    Compare your resampled spectrum against the PDS3 product and print line-by-line differences.

    Parameters
    ----------
    pds3_source : dict | (wl_nm, irr) tuple | str
        - dict from your reader with keys "wavelength_nm" and "irradiance", OR
        - tuple (wl_nm, irr) arrays, OR
        - path string to a PDS3 file (requires read_pds3_solar_spectrum(...) in scope).
    bins_nm : array-like
        Wavelength centers (nm) of your resampled spectrum.
    y_resampled : array-like
        Your resampled irradiance values at bins_nm.

    Prints
    ------
    wl : pds3 : ours : diff (where diff = pds3 - ours)
    Summary stats:
        count, total_abs_difference, mean_abs_difference (MAE), bias (mean diff),
        rmse, min_difference, max_difference,
        mean_abs_percent_diff, median_abs_percent_diff

    Returns
    -------
    dict with keys:
        wavelength_nm, pds3, ours, diff,
        total_abs, mae, bias, rmse, min_diff, max_diff,
        mean_abs_percent_diff, median_abs_percent_diff
    """
    import numpy as np

    # --- Load PDS3 arrays ---
    if isinstance(pds3_source, dict):
        p_wl = np.asarray(pds3_source["wavelength_nm"], float)
        p_y  = np.asarray(pds3_source["irradiance"], float)
    elif isinstance(pds3_source, tuple) and len(pds3_source) == 2:
        p_wl = np.asarray(pds3_source[0], float)
        p_y  = np.asarray(pds3_source[1], float)
    elif isinstance(pds3_source, str):
        if "read_pds3_solar_spectrum" not in globals():
            raise ValueError("If pds3_source is a path, read_pds3_solar_spectrum(...) must exist in scope.")
        d = read_pds3_solar_spectrum(pds3_source)
        p_wl = np.asarray(d["wavelength_nm"], float)
        p_y  = np.asarray(d["irradiance"], float)
    else:
        raise TypeError("pds3_source must be a dict, (wl,irr) tuple, or path string.")

    bins_nm     = np.asarray(bins_nm, float)
    y_resampled = np.asarray(y_resampled, float)

    # --- Align on common wavelengths (robust to tiny float noise) ---
    p_wl_r = np.round(p_wl, 6)
    b_wl_r = np.round(bins_nm, 6)

    p_map    = {w: v for w, v in zip(p_wl_r, p_y)}
    ours_map = {w: v for w, v in zip(b_wl_r, y_resampled)}

    common = sorted(set(p_map.keys()) & set(ours_map.keys()))
    if not common:
        raise ValueError("No overlapping wavelengths between PDS3 and your resampled grid.")

    p_vals    = np.array([p_map[w]    for w in common], dtype=float)
    ours_vals = np.array([ours_map[w] for w in common], dtype=float)

    # diff defined as (PDS3 - ours)
    diffs = p_vals - ours_vals

    # --- Print per-line ---
    for w, pv, ov, dv in zip(common, p_vals, ours_vals, diffs):
        print(f"{w:.1f} : {pv:.6e} : {ov:.6e} : {dv:.6e}")

    # --- Summary stats ---
    total_abs = float(np.sum(np.abs(diffs)))
    mae       = float(np.mean(np.abs(diffs)))             # average deviation
    bias      = float(np.mean(diffs))                     # signed average difference
    rmse      = float(np.sqrt(np.mean(diffs**2)))
    min_diff  = float(np.min(diffs))
    max_diff  = float(np.max(diffs))

    # Relative differences (%), ignore zeros in PDS3 to avoid div-by-zero
    nz = np.abs(p_vals) > 0
    if np.any(nz):
        abs_pct = np.abs(diffs[nz]) / np.abs(p_vals[nz]) * 100.0
        mean_abs_pct = float(np.mean(abs_pct))
        median_abs_pct = float(np.median(abs_pct))
    else:
        mean_abs_pct = float('nan')
        median_abs_pct = float('nan')

    # Print summary
    print(f"\ncount: {len(common)}")
    print(f"total_abs_difference: {total_abs:.6e}")
    print(f"mean_abs_difference (MAE): {mae:.6e}")
    print(f"bias (mean difference): {bias:.6e}")
    print(f"rmse: {rmse:.6e}")
    print(f"min_difference: {min_diff:.6e}")
    print(f"max_difference: {max_diff:.6e}")
    print(f"mean_abs_percent_diff: {mean_abs_pct:.3f}%")
    print(f"median_abs_percent_diff: {median_abs_pct:.3f}%")

    return {
        "wavelength_nm": np.array(common, dtype=float),
        "pds3": p_vals,
        "ours": ours_vals,
        "diff": diffs,
        "total_abs": total_abs,
        "mae": mae,
        "bias": bias,
        "rmse": rmse,
        "min_diff": min_diff,
        "max_diff": max_diff,
        "mean_abs_percent_diff": mean_abs_pct,
        "median_abs_percent_diff": median_abs_pct,
    }

"""
calibration
"""
def read_flats():

    root = Path(__file__).parent.resolve()
    flat_folder = root / 'calibration_data/FLATS'

    flat = flat_folder / 'AS2_FLAT_HIGH.fts'
    
    with fits.open(flat) as hdul:

        primary_hdu = hdul[0]
        primary_header = primary_hdu.header
        primary_data = primary_hdu.data
        print(repr(primary_header))

        plt.imshow(primary_data)
        plt.show()

def try_flat_cal(path):
    with fits.open(path) as hdul:
        original = utilities.convert_to_float64(hdul)
        original = extractCDS.extract_cds_pixels(original)
        original = hdul[0].data
        flat_corrected_hdul = flatfield.flat_field_calibration(hdul)
        flat_corrected = flat_corrected_hdul[0].data

    
    for i in range(len(original)):
        diff = np.mean(np.abs(original[i] - flat_corrected[i]))
        print(f'frame {i} mean diff: {diff}')
    
    reprod = original[1] / flat_corrected[1]
    plt.figure()

    plt.subplot(1,3,1)
    plt.imshow(original[1], cmap='gray')
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(flat_corrected[1], cmap='gray')
    plt.title('Flat field corrected')
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(reprod, cmap='viridis')
    plt.title('original / corrected')
    plt.axis('off')

    plt.show()
"""
mat files
"""
def read_mat_files(path):
    mat_data = loadmat(path)
    print(list(mat_data.keys()))  # List all variable names
    print(f"exposure: {mat_data['exposure_ms']}")
    print(f"wavelengths: {mat_data['wavelengths']}")
    print(f"shape of cube: {mat_data['cube'].shape}")

def dump_mat_cube_frames(channel, mat_path, out_dir):
    """
    Read a MATLAB .mat hyperspectral cube and write per-frame raw .bin files.

    Parameters
    ----------
    channel  : str   'Vis' or 'NIR' (case-insensitive)
    mat_path : str   path to .mat file that contains 'cube' (H, W, N) and 'wavelengths'
    out_dir  : str   folder to write .bin files into (created if missing)

    Returns
    -------
    list of str : file paths written
    """
    import os
    import numpy as np
    from scipy.io import loadmat

    ch = channel.strip().lower()
    if ch not in ("vis", "nir"):
        raise ValueError("channel must be 'Vis' or 'NIR'")

    os.makedirs(out_dir, exist_ok=True)

    # Load the .mat (MAT v5 style)
    data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if "cube" not in data:
        raise KeyError("The .mat file does not contain a 'cube' variable.")
    cube = data["cube"]

    if cube.ndim != 3:
        raise ValueError(f"'cube' must be 3D (H, W, N). Got shape {cube.shape}")

    H, W, N = cube.shape

    written = []

    if ch == "vis":
        # Expect 10 frames; write all as dc_0_exp_###
        for i in range(N):
            fname = f"dc_0_exp_{i:03d}.bin"
            path = os.path.join(out_dir, fname)
            # preserve dtype; write row-major contiguous
            cube[..., i].ravel(order="C").tofile(path)
            written.append(path)

    else:  # NIR
        # Expect 20 frames; split first half to dc_1, second half to dc_2
        half = N // 2  # with N=20 -> 10
        # first half
        for i in range(half):
            fname = f"dc_1_exp_{i:03d}.bin"
            path = os.path.join(out_dir, fname)
            cube[..., i].ravel(order="C").tofile(path)
            written.append(path)
        # second half
        for i in range(half, N):
            fname = f"dc_2_exp_{(i - half):03d}.bin"
            path = os.path.join(out_dir, fname)
            cube[..., i].ravel(order="C").tofile(path)
            written.append(path)

    print(f"Wrote {len(written)} files to: {out_dir}")
    print(f"cube dtype={cube.dtype}, shape=(H={H}, W={W}, N={N})")
    return written

"""
spectra
"""

def plot_spectrum(csv_path, spectrum_col: int, *, delimiter=None, skiprows=0,
                  xlabel="Wavelength (nm)", ylabel="Intensity", title=None):
    """
    Plot a spectrum from a file where col0 = wavelength and cols 1..N = spectra.

    Parameters
    ----------
    csv_path : str or Path
        Path to the data file (csv/tsv/space-separated).
    spectrum_col : int
        1-based index of the spectrum column to plot (1 maps to file column #1).
    delimiter : str or None, optional
        If None, autodetect (',' -> CSV, else whitespace).
    skiprows : int, optional
        Number of header rows to skip.
    xlabel, ylabel, title : str, optional
        Axis labels and plot title.
    """
    csv_path = Path(csv_path)

    # Auto-detect delimiter if not given
    if delimiter is None:
        with open(csv_path, "r", encoding="utf-8") as f:
            first = f.readline()
        delimiter = "," if "," in first else None  # None => whitespace split

    # Load data
    arr = np.loadtxt(csv_path, delimiter=delimiter, skiprows=skiprows)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("Expected at least two columns: wavelength + one spectrum.")

    # Validate requested spectrum column
    if spectrum_col < 1 or spectrum_col >= arr.shape[1]:
        raise IndexError(
            f"spectrum_col must be in [1, {arr.shape[1]-1}] "
            f"(you gave {spectrum_col})."
        )

    wl = arr[:, 0]
    y  = arr[:, spectrum_col]  # 1-based among spectra == file column index

    # Plot
    plt.figure()
    plt.plot(wl, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title or f"Spectrum column {spectrum_col}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return wl, y


""" 
Python3 ASPECT_calibration_pipeline/test_level_012.py
"""

"""
Function calls after this
"""

asp_sim = os.path.join(os.getcwd(), 'pipeline_results/ASPECT_simulated_20270323_McEwen/ASP_000000_270323T060000_3C_MGM.fits')
file_a = os.path.join(os.getcwd(),'test_data/ASPECT_noise_project/acqseq_100/acq_000/dc_0_exp_000.bin')
file_b = os.path.join(os.getcwd(), 'pipeline_results/ASPECT_noise_project/D1/AS0_000000_270101T060000_1B.fits')

# Calibration tests

test_dir = (Path(__file__).parent.parent / 'pipeline_results').resolve()
aps_250225 = test_dir / 'ASPECT_in-flight-dark_250225'
as0_250225_1A = aps_250225 / '100' / 'AS0_000000_200101T014231_1A.fits'
as1_250225_1A = aps_250225 / '104' / 'AS1_000000_200101T014800_1A.fits'
as2_250225_1A = aps_250225 / '104' / 'AS2_000000_200101T014800_1A.fits'

try_flat_cal(as0_250225_1A)
# read_flats()

"""
Alignment Demo
"""

asp = os.path.join(os.getcwd(), 'pipeline_results/ASPECT_simulated_20270323_McEwen/ASP_000000_270323T060000_2B.fits')
as0 = os.path.join(os.getcwd(), 'pipeline_results/ASPECT_simulated_20270323_McEwen/AS0_000000_270323T060000_1B.fits')
as1 = os.path.join(os.getcwd(), 'pipeline_results/ASPECT_simulated_20270323_McEwen/AS1_000000_270323T060000_1B.fits')
vis_bin = os.path.join(os.getcwd(), 'test_data/ASPECT_simulated_images/2027-03-23_06_00_00-McEwen/acq_000/dc_0_exp_000.bin')
nir_bin = os.path.join(os.getcwd(), 'test_data/ASPECT_simulated_images/2027-03-23_06_00_00-McEwen/acq_000/dc_1_exp_000.bin')

# inspect_pipeline_results(asp=asp,as0=as0, as1=as1, vis_bin=vis_bin,nir_bin=nir_bin)