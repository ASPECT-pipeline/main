import numpy as np
from astropy.io import fits
from astropy.io.fits import HDUList


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

    try:
        if channel == 'SWIR':
            return hdul
        else:

            width = header.get('NAXIS1')
            height = header.get('NAXIS2')

            # Place holder for the darkframe
            darkFrame = np.zeros((height, width), dtype=hdul[0].data.dtype)

            # To store the calibrated datacube
            new_data_cube = hdu.data.astype(np.float64, copy=True)

            # Loop over the 2D images inside the extension
            for i, image in enumerate(data):
                # Subtract the dark frame from image
                new_data_cube[i] = image - darkFrame

            # Add the modified image_HDU to the new HDU list
            data = new_data_cube
            return hdul
    except Exception as e:
        print(f'[WARNING] Dark subtraction failed: {e}')
        return hdul