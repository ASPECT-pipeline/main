import numpy as np
from astropy.io.fits import HDUList
from config import _path_sim_coef
from pathlib import Path

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
    header = hdu.header
    data = hdu.data

    missphas = header.get('MISSPHAS')
    channel = header.get('ASP_CHANNELS')

    distanceToSun = 1.0 # au

    if missphas == 'SIMULATED':
        match channel:
            case 'Vis':
                coef_file = Path(_path_sim_coef) / 'calibration-coefficients-VIS.dat'
                integration_time = 10
            case 'NIR1':
                coef_file = Path(_path_sim_coef) / 'calibration-coefficients-NIR1.dat'
                integration_time = 20
            case 'NIR2':
                coef_file = Path(_path_sim_coef) / 'calibration-coefficients-NIR2.dat'
                integration_time = 20
            case _:
                coef_file = ''
                return hdul
        coefs = np.loadtxt(coef_file)
        try:
            new_data_cube = data.astype(np.float64, copy=True)
            #loop over the 2D images inside the extension
            for i, image in enumerate(data):
                cal_image = (image * coefs[i,1] / (integration_time * distanceToSun**2)).astype(np.float64) # multiply the image with the coefficient 
                new_data_cube[i] = cal_image
            data = new_data_cube
            hdul[0].data = data
            print(f'Radiometric calibrated')
            return hdul
        except Exception as e: 
            print(f'[WARNING] Radiometric calibration failed: {e}')
            return hdul
    else:
        print(f'[WARNING] Radiometric claibration not yet implemented for real data.')
        return hdul