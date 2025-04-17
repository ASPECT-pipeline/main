import os
import numpy as np
from astropy.io import fits
import utilities

"""
    This file contains 2 functions used to extract diagnostic pixels from NIR channel images.
        
    extractDiagnosticPixels(fitsFile, outputPath)
        - takes a FITS file and path to output folder as parameter
        - returns a new FITS file where diagnostic pixels are extracted from the data cube and stored to a Varible Lenght Array Table extention
        - Variable Array Table (VLA) can be accessed with hdul[2]
            - VLA contains one column for each 2D image, named as, Image_i, where i is the number of the image
            - Each column will have 518 rows for each row of the image
            - each cell of one row will contain a list of diasnostic pixells extacted from the corresponding row and image.
            - For instance the diagnostic pixels can be accessed:
                vla_table = hdul[2].data
                # Iterate over each column
                for col_name in vla_table.columns.names:
                    print(f"Column: {col_name}") # The 2D image of the data cube
                    # Iterate over each row in the column (image)
                    for i, row in enumerate(vla_table[col_name]):
                        # Print each cell in the row
                        print(f"  Row {i+1}: {row}") #row is a list of diagnostic pixels on the corresponding row of the image
            - each row will have either 648 diasnostic pixels (rows 1-5, and 518) or 8 diasnostic pixels,
                where first 4 is form the start of the row and last 4 from the end of the row (rows 6 - 517)

"""

def extract_diagnostic_pixels(fits_path: str, output: str):

    with fits.open(fits_path) as hdul:

        # Data from fits file
        primary_HDU = hdul[0]
        img_HDU = hdul[1] # Contains the image cube (or swir readings)
        img_data = img_HDU.data # Image data
        img_header = img_HDU.header # Image HDU header
        channel = img_header.get('CHANNEL') # Channel (VIS, NIR1, NIR2, SWIR)

        # This step is not done to VIS or SWIR images
        if channel in ('VIS', 'SWIR'):
            print(f'Returning same file in ramove diasnostics for channle: {channel}')
            return fits_path
        
        #Get image dimensions
        width = img_header.get('NAXIS1')
        height = img_header.get('NAXIS2')
        slices = img_header.get('NAXIS3')

        # Create new list of HDU's and append the primary HDU, new image HDU and other extensions
        HDUs = []
        HDUs.insert(0, hdul[0])

        # Empty array for cleaned data
        cleanedData = np.zeros((slices, height - 6, width - 8), dtype=img_data.dtype)
        
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
            cleanedData[i, :, :] = cleanedImage
        
        #Create Image HDU with keywords
        cleaned_cube = fits.ImageHDU(data=cleanedData, header=img_header)

        #Create a binary table HDU for the diagnostic pixels
        vla_hdu = fits.BinTableHDU.from_columns(vla_columns)

        HDUs.append(cleaned_cube)
        HDUs.append(vla_hdu)

        #File name for new fits
        file_name = f'{channel}_1A_Rd.fits'

        # create the new fits file with dark-subtracted images
        hdu_list = fits.HDUList(HDUs)
        fits_file = os.path.join(output, file_name)
        hdu_list = fits.HDUList(HDUs)
        hdu_list.writeto(fits_file, overwrite=True)

    return(fits_file)