import os
import numpy as np
from astropy.io import fits
from pathlib import Path



def dark_subtraction(fits_path: str | Path, output_dir: str | Path) -> str:

    """
    Function for subtracting a dark frame from each 2D image.

    Parmeters:
        fits_path: Path to the FITS file.
        output_dir: Path to the folder where the new fits file will be stored.

    Returns:
        path to the created fits file.
    """

    fits_path = Path(fits_path)
    output_dir = Path(output_dir)

    #Read the FITS file
    with fits.open(fits_path) as hdul:

        # Data from fits file
        primary_hdu = hdul[0]
        primary_header = primary_hdu.header
        img_HDU = hdul[1] # Contains the image cube (or swir readings)
        img_header = img_HDU.header # Image HDU header
        channel = img_header.get('CHANNEL') # Channel (VIS, NIR1, NIR2, SWIR)
       
        if channel == 'SWIR':
            return(fits_path)
        else:


            width = img_header.get('NAXIS1')
            height = img_header.get('NAXIS2')

            # Create new list of HDU's and append the cube to it
            HDUs = []
            HDUs.insert(0, hdul[0])

            # Place holder for the darkframe
            darkFrame = np.zeros((height, width), dtype=hdul[1].data.dtype)

            # To store the calibrated datacube
            new_data_cube = img_HDU.data.copy()

            # Loop over the 2D images inside the extension
            for i, image in enumerate(new_data_cube):
                
                # Subtract the dark frame from image
                new_data_cube[i] = image - darkFrame

            # Add the modified image_HDU to the new HDU list
            ImageHDU = fits.ImageHDU(data=new_data_cube, header=img_header)
            HDUs.append(ImageHDU)
        # Add all other extensions except for the original Image HDU
        for i in range(1, len(hdul)):
            if not isinstance(hdul[i], fits.ImageHDU):  # Skip the original Image HDU
                HDUs.append(hdul[i])

        hdu_list = fits.HDUList(HDUs)
        # File name for new fits
        stem = fits_path.stem
        suffix = fits_path.suffix
        new_calibration_level = '1B'
        file_name = stem[:25] + new_calibration_level + suffix
        primary_header = hdu_list[0].header
        primary_header['FILENAME'] = file_name
        primary_header['PROCLEVL'] = new_calibration_level
        # Create the new fits file with dark-subtracted images
        fits_file = os.path.join(output_dir, file_name)
        hdu_list.writeto(fits_file, overwrite=True)

    return(fits_file)