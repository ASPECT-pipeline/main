import numpy as np
from astropy.io import fits
from astropy.io.fits import HDUList

"""
Function for converting the pixel values into scientific units.

    Description:
        - Iterated over all 2D images inside the data cube multiplying it with a coefficient.
        - Creates a new FITS file with the calibrated data
"""


def radiometric_calibration(hdul: HDUList) -> HDUList:
    """
    Function for converting the pixel values into scientific units.

     Parmeters:
        hdul (HDUList): The HDU list of the FITS file that is modified
    
    Returns:
        The modified HDU list of the FITS file
    """

    # Data from fits file
    hdu = hdul[0]
    data = hdu.data
    coefficient = 1 # Temporary coefficient for radiometric calibration

    try:
        new_data_cube = data.astype(np.float64, copy=True)
        #loop over the 2D images inside the extension
        for i, image in enumerate(data):
            new_data_cube[i] = (image * coefficient).astype(np.float64) # multiply the image with the coefficient 
        data = new_data_cube
        hdul[0].data = data
        print(f'Radiometric calibrated')
        return hdul
    except Exception as e: 
        print(f'[WARNING] Radiometric calibration failed: {e}')
        return hdul