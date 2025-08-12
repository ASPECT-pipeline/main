import numpy as np
from astropy.io.fits import HDUList
from pathlib import Path
from config import reverse_channel_map, calibration_directory


def dark_subtraction(hdul: HDUList) -> HDUList:

    """
    Function for subtracting a dark frame from each 2D image.

    Parmeters:
        hdul (HDUList): The HDU list of the FITS file that is modified
    
    Returns:
        The modified HDU list of the FITS file
    """

    # Data from fits file
    hdu = hdul[0]
    header = hdu.header
    data = hdu.data
    channel = header.get('CHANNELS') # Channel (VIS, NIR1, NIR2, SWIR)
    channel_id = reverse_channel_map.get(channel)
    # Read Dark frame for this channel
    if channel == 'SWIR':
            return hdul
    try: 
        cal_dir = Path(calibration_directory)
        cal_file = Path(cal_dir / f'{channel_id}_dark_frame.bin', dtype=np.uint16)
        arr = np.fromfile(cal_file)
        if channel == 'VIS':
            w = h = 1024
            dark_frame =  arr.reshape((h,w), dtype=np.uint16)
        elif channel in ('NIR1', 'NIR2'):
            w = 512
            h = 640
            dark_frame = arr.reshape((h,w), dtype=np.uint16)
        else:
            print(f"[WARNING] incorrect channel '{channel}'")
            return hdul
    except Exception as e:
        print(f'[WARNING] Caught Exception while reading dark frame: {e}')
        return hdul
    # Do dark fram correction
    try:
        # To store the calibrated datacube
        new_data_cube = hdu.data.astype(np.float64, copy=True)

        # Loop over the 2D images inside the extension
        for i, image in enumerate(data):
            # Subtract the dark frame from image
            new_data_cube[i] = image - dark_frame

        # Add the modified image_HDU to the new HDU list
        data = new_data_cube
        return hdul
    except Exception as e:
        print(f'[WARNING] Dark subtraction failed: {e}')
        return hdul