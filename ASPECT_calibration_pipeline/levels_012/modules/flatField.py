import numpy as np
from astropy.io import fits
from astropy.io.fits import HDUList
from pathlib import Path
from config import reverse_channel_map, calibration_directory



def flat_field_calibration(hdul: HDUList) -> HDUList:
    """
    Applies flatfield correction to each 2D frame on the channel
    
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
    if channel == 'SWIR':
            return hdul
    # Read the Flat field for this channel
    try: 
        cal_dir = Path(calibration_directory)
        cal_file = Path(cal_dir / f'{channel_id}_flat_field.bin', dtype=np.uint16)
        arr = np.fromfile(cal_file)
        if channel == 'VIS':
            w = h = 1024
            flat_field =  arr.reshape((h,w), dtype=np.uint16)
        elif channel in ('NIR1', 'NIR2'):
            w = 512
            h = 640
            flat_field =  arr.reshape((h,w), dtype=np.uint16)
        else:
            print(f"[WARNING] incorrect channel '{channel}'")
            return hdul
    except Exception as e:
        print(f'[WARNING] Caught Exception while reading flat field: {e}')
        return hdul
    # Flat field correction
    try:

        # To store the calibrated datacube
        new_data_cube = hdu.data.astype(np.float64, copy=True)

        # loop over the 2D images inside the extension
        for i, image in enumerate(data):
            # Divide the image with the flatfield 
            new_data_cube[i] = image / flat_field
        
        data = new_data_cube
        return hdul
    except Exception as e:
        print(f'[WARNING] Flat field failed: {e}')
        return hdul