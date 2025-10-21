import numpy as np
from astropy.io.fits import HDUList
from astropy.io import fits
from pathlib import Path
from config import reverse_channel_map, _path_dark


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
    channel_id = reverse_channel_map.get(channel)
    # Read Dark frame for this channel
    if channel == 'SWIR':
            return hdul
    
    order = header.get(f'AS{channel_id}_ORDER')
    if order not in ('LOW', 'HIGH'):
        print(f'[WARNING] channel {channel} order is {order}. Dark subtraction failed.')
        return hdul
    
    try: 
        dark_dir = Path(_path_dark)
        flat_file = dark_dir / f'AS{channel_id}_DARK_{order}.fts'
        with fits.open(flat_file) as flat_hdul:
            flat_field = flat_hdul[0].data
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


        hdul[0].data = new_data_cube
        print(f'Dark frame subtracted')
        return hdul
    except Exception as e:
        print(f'[WARNING] Dark subtraction failed: {e}')
        return hdul