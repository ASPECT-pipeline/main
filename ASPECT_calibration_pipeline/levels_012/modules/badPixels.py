import numpy as np
from astropy.io import fits
from pathlib import Path
from astropy.io.fits import HDUList
from config import reverse_channel_map, calibration_directory
from typing import List

import matplotlib.pyplot as plt

def get_valid_neighbors(arr: np.ndarray, i: int, j: int) -> List:
    h, w = arr.shape[:2]
    offsets = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    out = []
    for di, dj in offsets:
        ni, nj = i + di, j + dj
        if 0 <= ni < h and 0 <= nj < w:
            if arr[ni, nj] == 0:
                out.append((ni, nj))
                d = 0
    return out


def replace_bad_pixels(hdul: HDUList) -> HDUList:
    """
    Function for removing bad pixels from each 2D image inside a cube given a mask.

    Description:
        - Iterate over each 2D slice of the cube
            - iterate over each pixel and identify bad pixels with the mask
            - change the value of bad pixels into mean of neigbours
            - if no 'good' neighbours put the pixel on hold
        - Iterate over the pictures on hold and determine valeus for them
        - Create a new FITS file and return the path to it

    Parmeters:
        hdul (HDUList): The HDU list of the FITS file that is modified
    
    Returns:
        The modified HDU list of the FITS file
    """

    # Data from fits file
    hdu = hdul[0]
    header = hdu.header
    data = hdu.data

    channel = header.get('CHANNELS') # Channel (Vis, NIR1, NIR2, SWIR)
    channel_id = reverse_channel_map.get(channel)
    if channel == 'SWIR':
            return hdul
    # Read the bad pixel mask for this channel
    try: 
        cal_dir = Path(calibration_directory)
        cal_file = Path(cal_dir / f'{channel_id}_bad-pixel_mask.bin')
        arr = np.fromfile(cal_file, dtype=np.uint16)
        if channel == 'Vis':
            w = h = 1024
            bad_pixel_mask =  arr.reshape((w, h)).astype(np.uint16)
        elif channel in ('NIR1', 'NIR2'):
            w = 512
            h = 640
            bad_pixel_mask =  arr.reshape((w, h)).astype(np.uint16)
        else:
            print(f"[WARNING] incorrect channel '{channel}'")
            return hdul
    except Exception as e:
        print(f'[WARNING] Caught Exception while reading bad pixel mask: {e}')
        return hdul
    # bad-pixel correction
    try:
        # To store the bad pixels that did not have 'good' neighbours
        on_hold = []

        # To store the calibrated datacube
        new_data_cube = data.astype(np.float64, copy=True)

        # Iterate over all 2D images inside the cube
        for num, frame in enumerate(new_data_cube):
            # Update the mask to remove the already corrected values
            mask = bad_pixel_mask.copy()
            for i in range(frame.shape[0]):
                for j in range(frame.shape[1]):
                    if mask[i, j] == 1:
                        valid_neighbors = get_valid_neighbors(mask, i, j)
                        if not valid_neighbors:
                            on_hold.append((i,j))
                            continue # No valid neihbours
                        sum = 0
                        d = len(valid_neighbors)
                        for ni, nj in valid_neighbors:
                            sum += frame[ni, nj]
                        new_val = sum / d
                        frame[i, j] = new_val
                        mask[i, j] = 0
            # If some pixels did not have 'good' neigbors
            if on_hold:
                on_hold.reverse()
                for i, j in on_hold:
                    valid_neighbors = get_valid_neighbors(mask, i, j)
                    sum = 0
                    d = len(valid_neighbors)
                    for ni, nj in valid_neighbors:
                        sum += frame[ni, nj]
                    new_val = sum / d
                    frame[i, j] = new_val
                    mask[i, j] = 0
        hdul[0].data = new_data_cube
        print(f'Bad pixels replaced')
        return hdul
    except Exception as e:
        print(f'[WARNING] bad pixel removal failed: {e}')
        return hdul