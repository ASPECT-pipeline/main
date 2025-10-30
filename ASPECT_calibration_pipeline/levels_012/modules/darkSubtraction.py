import numpy as np
from astropy.io.fits import HDUList
from astropy.io import fits
from pathlib import Path
from config import reverse_channel_map, _path_dark, _path_sim_dark


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
    channel = header.get('ASP_CHANNELS') # Channel (Vis, NIR1, NIR2, SWIR)
    missphase = header.get('MISSPHAS')

    channel_id = reverse_channel_map.get(channel)
    # Read Dark frame for this channel
    if channel == 'SWIR':
            return hdul
    
    order = header.get(f'AS{channel_id}_ORDER')

    if missphase == 'SIMULATED': # For simulated data
        try: 
            dark_dir = Path(_path_sim_dark)
            dark_file = dark_dir / f'AS{channel_id}_DARK.fits'
            with fits.open(dark_file) as dark_hdul:
                dark_frame = dark_hdul[0].data
        except Exception as e:
            print(f'[WARNING] Caught Exception while reading dark frame: {e}')
            return hdul
    else:
        if order not in ('LOW', 'HIGH'):
            print(f'[WARNING] channel {channel} order is {order}. Dark subtraction failed.')
            return hdul
        try: 
            dark_dir = Path(_path_dark)
            dark_file = dark_dir / f'AS{channel_id}_DARK_{order}.fits'
            with fits.open(dark_file) as dark_hdul:
                dark_frame = dark_hdul[0].data
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
            corrected = image - dark_frame
            corrected = np.clip(corrected, 0, None)
            new_data_cube[i] = corrected


        hdul[0].data = new_data_cube
        print(f'Dark frame subtracted')
        return hdul
    except Exception as e:
        print(f'[WARNING] Dark subtraction failed: {e}')
        return hdul