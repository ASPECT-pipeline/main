import os
import numpy as np
from astropy.io import fits
from pathlib import Path



def flat_field_calibration(fits_path: str, output_dir: str) -> str:
    """
    Applies flatfield correction to each 2D frame on the channel
    
    Parmeters:
        fits_path: Path to the FITS file.
        output: Path to the folder where the new fits file will be stored.

    Returns:
        path to the created fits file.
    """
    fits_path = Path(fits_path)
    output_dir = Path(output_dir)

    # Open the fits file
    with fits.open(fits_path) as hdul:

        # Data from fits file
        primary_hdu = hdul[0]
        primary_header = primary_hdu.header
        img_HDU = hdul[1] # Contains the image cube (or swir readings)
        img_header = img_HDU.header # Image HDU header
        channel = img_header.get('CHANNEL') # Channel (VIS, NIR1, NIR2, SWIR)
        
        # This step is not done to SWIR images
        if channel == 'SWIR':
            return fits_path
        else:
            width = img_header.get('NAXIS1')
            height = img_header.get('NAXIS2')

            # Create new list of HDU's and append the primary HDU, new image HDU and other extensions
            HDUs = []
            HDUs.insert(0, hdul[0])

            # Place holder flatfield array
            # This should be replaced wiht the correct flatfield used for the imager.
            flatField = np.ones((height, width), dtype=hdul[1].data.dtype)

            # To store the calibrated datacube
            new_data_cube = img_HDU.data.astype(np.float64, copy=True)
            flatField = flatField.astype(np.float64)

            # loop over the 2D images inside the extension
            for i, image in enumerate(new_data_cube):
                # Divide the image with the flatfield 
                new_data_cube[i] = image / flatField
            
        
            ImageHDU = fits.ImageHDU(data=new_data_cube, header=img_header)
            HDUs.append(ImageHDU)
            # Add all other extensions except for the original Image HDU
            for i in range(1, len(hdul)):
                if not isinstance(hdul[i], fits.ImageHDU):  # Skip the original Image HDU
                    HDUs.append(hdul[i])
        

        # create the new fits file with dark-subtracted images
        hdu_list = fits.HDUList(HDUs)
        stem = fits_path.stem
        suffix = fits_path.suffix
        new_calibration_level = '1B'
        file_name = stem[:25] + new_calibration_level + suffix
        primary_header['FILENAME'] = file_name
        primary_header['PROCLEVL'] = new_calibration_level
        # Create the new fits file with dark-subtracted images
        fits_file = os.path.join(output_dir, file_name)
        hdu_list.writeto(fits_file, overwrite=True)


    return(fits_file)