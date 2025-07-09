import os
import numpy as np
from astropy.io import fits
import modules.utilities as utilities
from pathlib import Path

"""
    This file extracts correlated double sampling (CDS) pixels surrounding the NIR images, storing them into a separate BinaryTableHDU.
    The function also converts the image data into double precission (float64) values. 

    Binary table architecture
        - Variable Array Table (VLA) can be accessed with hdul[2] (third HDU after primary, and image)
            - VLA contains one column for each 2D image, named as, Image_i, where i is the frame number
            - Each column will have 518 rows, one for each row in the image
            - Each cell of a row will contain a list of CDS extacted from the corresponding row and image.
            - Each row will have either 648 diasnostic pixels (rows 1-5, and 518) or 8 diasnostic pixels,
                where first 4 is form the start of the row and last 4 from the end of the row (rows 6 - 517)
            - For instance the diagnostic pixels can be accessed:
                vla_table = hdul[2].data
                # Iterate over each column
                for col_name in vla_table.columns.names:
                    print(f"Column: {col_name}") # The 2D frame of the data cube
                    # Iterate over each row in the column (image)
                    for i, row in enumerate(vla_table[col_name]):
                        # Print each cell in the row
                        print(f"  Row {i+1}: {row}") #row is a list of diagnostic pixels on the corresponding row of the image
        - this functions uses extract_diagnostics function from utilities.py to extract the pixels.

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
        print()
        print(f'data before cds extraction')
        print(f'cube[0] (250, 250): {img_data[0][250][250]}')
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
                
                # Prepare a list to hold the columns for the VLA table
                vla_columns = []
                
                # Step 4: Iterate over each slice of the cube
                for i, image in enumerate(img_data):
                    
                    # Extract the diagnostic data from the slice
                    cleanedImage, diagnostics = utilities.extract_diagnostics(image)
                    
                    # Create a FITS column for the current slice, with the name indicating the slice
                    col_name = f'Image_{i+1}'
                    vla_columns.append(fits.Column(name=col_name, format='PI()', array=diagnostics))
                    
                    # Append the cleanedImage to the new image HDU
                    cleanedData[i, :, :] = cleanedImage.astype(np.float64)

                
                # Create Image HDU with keywords
                cleaned_cube = fits.ImageHDU(data=cleanedData, header=img_header) # Also converts the images to double precission
                print()
                print(f'cleaned cube[0] (250, 250): {cleaned_cube.data[0][250][250]}')
                # Create a binary table HDU for the diagnostic pixels
                vla_hdu = fits.BinTableHDU.from_columns(vla_columns)

                HDUs.append(cleaned_cube)
                HDUs.append(vla_hdu)
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