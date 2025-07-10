import os
import numpy as np
from astropy.io import fits
import modules.utilities as utilities
from pathlib import Path

"""
    This file extracts correlated double sampling (CDS) pixels surrounding the NIR images, storing them into a separate BinaryTableHDU.
    The function also converts the image data into double precission (float64) values. 

    Binary table architecture
        - BinaryTableHDU can be accessed with hdul[2] (third HDU after primary, and image)
            - Contains one column for each 2D image, named as, Channel_i, where channel is 'NIR1' or 'NIR2' i is the frame number
            - Each column will have 1 row with all cds pixels flattened total of 7984 values
            - utilities offer function read_cds() to retrieve the desired cds pixels from the created file

"""

def extract_cds_pixels(fits_path: str | Path, output_dir: str | Path) -> str:
    """
    Extracts the CDS pixels from NIR images and strores them in a separate BinaryTable.
    Converts the image data into double precision floating point values.

    Parmeters:
        fits_path: Path to the FITS file.
        output: Path to the folder where the new fits file will be stored.
    
    Returns:
        Path to the new fits file.
    """

    fits_path = Path(fits_path)
    output_dir = Path(output_dir)

    with fits.open(fits_path) as hdul:

        # Data from fits file
        primary_hdu = hdul[0]
        primary_header = primary_hdu.header
        img_HDU = hdul[1] # Contains the image cube (or swir readings)
        img_data = img_HDU.data # Image data
        img_header = img_HDU.header # Image HDU header
        channel = img_header.get('CHANNEL') # Channel (VIS, NIR1, NIR2, SWIR)
        missphas = primary_header.get('MISSPHAS')
        HDUs = []  # Create new list of HDU's and append the primary HDU, new image HDU and other extensions
        HDUs.insert(0, hdul[0])
        if missphas == 'TEST':
            double_precission = fits.ImageHDU(data=img_data.astype(np.float64), header=img_header)
            HDUs.append(double_precission)
        else:
            if channel in ('NIR1', 'NIR2'):
                #Get image dimensions
                width = img_header.get('NAXIS1')
                height = img_header.get('NAXIS2')
                slices = img_header.get('NAXIS3')


                # Empty array for cleaned data
                cleanedData = np.zeros((slices, height - 6, width - 8), dtype=np.float64)
                
                # Prepare a list to hold the columns for CDS binary table
                cds_list = []
                # Step 4: Iterate over each slice of the cube
                for i, image in enumerate(img_data):
                    # Extract the cds pixels from the slice
                    cleanedImage, cds = utilities.extract_cds(image)
                    cds_list.append(cds)               
                    # Append the cleanedImage to the new image HDU
                    cleanedData[i, :, :] = cleanedImage.astype(np.float64)

                # Create Image HDU with keywords
                cleaned_cube = fits.ImageHDU(data=cleanedData, header=img_header) 
                # Create columns for the cds pixels
                columns = []
                for i, col in enumerate(cds_list):
                    column = fits.Column(
                        name= f'{channel}_{i}',
                        format=f'{len(col)}J', # unsigned data
                        array=[col]
                    )
                    columns.append(column)

                # Create a binary table HDU for the cds pixels
                cds_table = fits.BinTableHDU.from_columns(columns)
                HDUs.append(cleaned_cube)
                HDUs.append(cds_table)
            elif channel == 'VIS':
                double_precission = fits.ImageHDU(data=img_data.astype(np.float64), header=img_header)
                HDUs.append(double_precission)
            else:
                cols = []
                for col in img_data.columns:
                    name = col.name
                    value = img_data[name].astype(np.float64)
                    new_col = fits.Column(name=name, format='D', array=value)
                    cols.append(new_col)
                new_table = fits.BinTableHDU.from_columns(cols, header=img_header) 
                HDUs.append(new_table)

        hdu_list = fits.HDUList(HDUs)
        stem = fits_path.stem
        suffix = fits_path.suffix
        new_calibration_level = '1B'
        file_name = stem[:25] + new_calibration_level + suffix
        primary_header = hdu_list[0].header
        primary_header['FILENAME'] = file_name
        primary_header['PROCLEVL'] = new_calibration_level
        # create the new fits file with dark-subtracted images
        fits_file = os.path.join(output_dir, file_name)
        hdu_list.writeto(fits_file, overwrite=True)

    return(fits_file)