import numpy as np
from astropy.io import fits
from astropy.io.fits import HDUList
from pathlib import Path
from config import reverse_channel_map, _path_flat



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
    missphas = header.get('MISSPHAS')
    channel = header.get('ASP_CHANNELS') # Channel (Vis, NIR1, NIR2, SWIR)
    channel_id = reverse_channel_map.get(channel)
    if channel == 'SWIR' or missphas == 'SIMULATED':
            return hdul
    
    order = header.get(f'AS{channel_id}_ORDER')
    if order not in ('LOW', 'HIGH'):
        print(f'[WARNING] channel {channel} order is {order}. Flat field calibration failed.')
        return hdul
    
    # Read the Flat field for this channel and order
    try: 
        flat_dir = Path(_path_flat)
        flat_file = flat_dir / f'AS{channel_id}_FLAT_{order}.fts'
        with fits.open(flat_file) as flat_hdul:
            flat_field = flat_hdul[0].data
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
        
        hdul[0].data = new_data_cube
        print(f'Flat field calibrated')
        return hdul
    except Exception as e:
        print(f'[WARNING] Flat field failed: {e}')
        return hdul