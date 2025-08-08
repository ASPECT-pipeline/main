import numpy as np
from astropy.io import fits
from astropy.io.fits import HDUList



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
    
    # This step is not done to SWIR images
    try:
        if channel == 'SWIR':
            return hdul
        else:
            width = header.get('NAXIS1')
            height = header.get('NAXIS2')

            # Place holder flatfield array
            # This should be replaced wiht the correct flatfield used for the imager.
            flatField = np.ones((height, width), dtype=hdul[0].data.dtype)

            # To store the calibrated datacube
            new_data_cube = hdu.data.astype(np.float64, copy=True)

            # loop over the 2D images inside the extension
            for i, image in enumerate(data):
                # Divide the image with the flatfield 
                new_data_cube[i] = image / flatField
            
            data = new_data_cube
            return hdul
    except Exception as e:
        print(f'[WARNING] Flat field failed: {e}')
        return hdul