import numpy as np
from astropy.io import fits
from astropy.io.fits import HDUList
from config import reverse_channel_map


def reflectance_calibration(hdul: HDUList) -> HDUList:
    """
    The image is calibrated by pixel linear transformations. The pixels represent I/f reflectance units.

    Parameters: 
        hdul (HDUList) : hdu list containing the Primary HDU data (image) to be converted to I/f 
    
    Returns: 
        Reflectance unit converted data inside the same hdul 
    """

    primary_hdu = hdul[0]
    primary_header = primary_hdu.header
    data = primary_hdu.data

    channels = primary_header.get('CHANNELS')
    wl = {}

    for channel in channels: 
        channel_id = reverse_channel_map.get(channel)
        wl[channel] = primary_header.get(f'{channel_id}_WL').split(',')

    
