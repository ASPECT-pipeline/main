import numpy as np
from astropy.io import fits
from pathlib import Path
from astropy.io.fits import HDUList
from dataclasses import dataclass
from config import reverse_channel_map, _path_bad_pixels, _path_sim_bad_pixels
from typing import List

import matplotlib.pyplot as plt

@dataclass
class BadRegion:
    """
    One bad-pixel region specification from the BADPIXELS TXT file

    type: str
        Region type code:
        'P' = single pixel,
        'H' = horizontal cluster,
        'V' = vertical cluster,
        'R' = rectangular region
    col: int | None
        Column (x) start coordinate in pixels. None if file contains '-'
    row: int | None
        Row (y) start coordinate in pixels. None if file contains '-'
    size_x: int
        width of the region in pixels
    size_y: int
        height of the region in pixels
    """
    type: str
    col: int | None
    row: int | None
    size_x: int
    size_y: int

def parse_bad_pixel_list(txt_path: str | Path) -> list[BadRegion]:
    """
    Parses the tab separated BADPIXELS TXT file into structured region objects. 

    The file foramt should have 5 tab-separated fields per data line:
        Type, Col, Row, SizeX, SizeY
    
    txt_path: str | Path 
        Path to txt file containing bad pixel information

    return: A list of BadRedion elements created based on the txt file
    """

    regions: list[BadRegion] = []

    with open(txt_path, "r", encoding='utf-8', errors='replace') as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            if line.lower().startswith("type"): # Header
                continue
        
            parts = line.split("\t")
            parts = [p.strip() for p in parts if p.strip() != ""]
            if len(parts) != 5:
                print(f"[WARNING] Error in parsing bad pixels on line {line_no}: Expexted 5 tab fields, got {len(parts)}")
                continue
        
            t, col_s, row_s, sx_s, sy_s = parts
            col = None if col_s == "-" else int(col_s)
            row = None if row_s == "-" else int(row_s)

            regions.append(BadRegion(
                type=t,
                col=col,
                row=row,
                size_x=int(sx_s),
                size_y=int(sy_s)
            ))
        return regions

def slice_region(r: BadRegion) -> tuple[slice, slice]:
    """
    Converts a BadRegion into Numpy index slices (yslice, xslice) for a 2D frame.

    Interpretation rules:
    - P: selects one pixel at (row, col)
    - H: selects rows [row : row+size_y] and cols [x0 : x0+size_x]
    - V: selects rows [y0 : y0+size_y] and cols [col : col+size_x]
    - R: selects rows (row : row+size_y) and cols [col : col+size_X]

    r: BadRegion
        Region definition to convert into slices
    
    return: (yslice, xslice) selecting the pixels from the region
    """
    t = r.type.upper()

    if t == "P":
        if r.row is None or r.col is None:
            raise ValueError("P type requires both row and col")
        return (slice(r.row, r.row + 1), slice(r.col, r.col + 1))
    
    if t == "H":
        if r.row is None:
            raise ValueError("H type requires row")
        x0 = 0 if r.col is None else r.col
        return (slice(r.row, r.row + r.size_y) , slice(x0, x0 + r.size_x))
    
    if t == "V":
        if r.col is None:
            raise ValueError("V type requires a col")
        y0 = 0 if r.row is None else r.row
        return (slice(y0, y0 + r.size_y), slice(r.col, r.col + r.size_x,))
    
    if t == "R":
        if r.row is None or r.col is None:
            raise ValueError("R type requires both row and col")
        return (slice(r.row, r.row + r.size_y), slice(r.col, r.col + r.size_x))
    
    raise(f"Unknown type: {r.type}")

def slices_to_mask(shape: tuple[int, int], slice_list: list[tuple[slice, slice]]) -> np.ndarray:
    """
    Builds a boolean mask from the sliced regions
    
    shape: tuple[int, int]
        Shape of the frame
    slice_list: list[tuple[slice, slice]]
        List of slice tuples defining all bad pixel regions
    
    returns: Boolean array where True indicates a bad pixel
    """
    mask = np.zeros(shape, dtype=bool)
    for ys, xs in slice_list:
        mask[ys, xs] = True
    return mask

def repace_nan_8neighor(frame: np.ndarray, max_iter: int = 50) -> np.ndarray:
    """
    Fill NaN pixels in a frame by interatively averaging finite 8-neighbours
    
    In each iteration:
        - Identify NaN pixels
        - For every pixel, compute:
            s[y, x] = sum of finite neihbour values in the 8-connected neighbourhood
            c[y,x] = count of finite neighbours
        - update only NaN pixels with c>0 using s/c
        - Repeat up to 'max_iter' times or when all NaN pixels are replaced.

    frame: np.ndarray
        2D floating array. Bad pixels must be already set to np.nan
    max_iter: int
        Backup to stop the iteration if it is not done already
    
    returns: A new 2D array with NaNs filled gy the average of neihgbouring pixels.
    """

    a = frame.copy()
    for i in range(max_iter):
        nan_mask = np.isnan(a)
        if not nan_mask.any():
            break

        finite = np.isfinite(a)
        s = np.zeros_like(a, dtype=a.dtype)
        c = np.zeros_like(a, dtype=a.dtype)

        for dy, dx in [(-1, -1), (-1, 0), (-1, 1),
                       ( 0, -1),          ( 0, 1),
                       ( 1, -1), ( 1, 0), ( 1, 1)]:
            
            v = np.roll(a, shift=(dy, dx), axis=(0, 1))
            ok = np.roll(finite, shift=(dy, dx), axis=(0, 1))

            if dy == -1: ok[-1, :] = False
            if dy ==  1: ok[0, :] = False
            if dx == -1: ok[:, -1] = False
            if dx ==  1: ok[:, 0] = False

            s += np.where(ok, v, 0.0)
            c += ok.astype(np.float64)

        fill = nan_mask & (c > 0)
        if not fill.any():
            break
        
        a[fill] = s[fill] / c[fill]



    return a

def replace_bad_pixels(hdul: HDUList) -> HDUList:
    """
        Replace bad pixels in a FITS data cube using an external bad-pixel map.

        This function:
        1) Read the data cube from FITS primary HDU
        2) Determines the correct bad pixel list file based on FITS header
        3) Parses the bad-pixel list into regions, converts the regions to slices and constructs the boolean bad-pixel maks.
        4) For each frame:
            - set bad pixels to NaN using the mask
            - fill the NaNs with the neighbouring average and iterative process
        5) Writes the corrected cube back to the HDUList

        hdul: HDUList
            Open FITS HDUList with data stored in the primary HDU
        returns: HDUList with bad-pixel corrected data
    """
    # Data from fits file
    hdu = hdul[0]
    header = hdu.header
    data = hdu.data

    channel = header.get('ASP_CHANNELS') # Channel (Vis, NIR1, NIR2, SWIR)
    missphase = header.get('MISSPHAS')

    channel_id = reverse_channel_map.get(channel)
    if channel == 'SWIR':
            return hdul

    try: 
        if missphase == 'SIMULATED': # For simulated data
            bp_dir = Path(_path_sim_bad_pixels)
            bp_file = bp_dir / f'AS{channel_id}_BADPIXELS_v1.TXT'
        else:
            order = header.get(f'AS{channel_id}_ORDER')
            if order not in ('LOW', 'HIGH'):
                print(f'[WARNING] channel {channel} order is {order}. Bad pixel calibration failed.')
                return hdul
            bp_dir = Path(_path_sim_bad_pixels)
            bp_file = bp_dir / f'AS{channel_id}_BADPIXELS_{order}.TXT'
        
        regions = parse_bad_pixel_list(bp_file)
        slice_list = [slice_region(r) for r in regions]
        bad_mask = slices_to_mask(data[0].shape, slice_list)

        new_data_cube = data.astype(np.float64, copy=True)

        for i, frame in enumerate(data):
            bad_frame = frame.astype(np.float64, copy=True)
            bad_frame[bad_mask] = np.nan

            new_data_cube[i] = (repace_nan_8neighor(bad_frame))
        

        hdul[0].data = new_data_cube
        print(f'Bad pixels replaced')
        return hdul

    except Exception as e:
        print(f'[WARNING] Caught Exception while reading bad pixels: {e}')
        return hdul
