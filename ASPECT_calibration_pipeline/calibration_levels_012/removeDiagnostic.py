import os
import numpy as np
from astropy.io import fits

"""
    This file contains 2 functions used to extract diagnostic pixels from NIR channel images.

    FUNCTIONS:
        
        extractDiagnosticPixels(image)
            - takes a 2-Dimensional image as a parameter
            - returns a new image where diagnostic pixels are cropped out from the sides and
              a list of rows, where each row contains the diagnostic pixels of the corresponding row in the image

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

def extract_diagnostics(image):
    # Define diagnostic pixel regions
    top = 5  # Five lines at the top
    bottom = 1  # One line at the bottom
    left = 4  # Four columns on the left
    right = 4  # Four columns on the right
    # To store the extracted pixels
    diagnosticPixels = []

    # Step 1: Extract the first 5 rows
    for row in image[:top]:
        diagnosticPixels.append(row.tolist())
    
    # Step 2: For the remaining rows (except the last one), extract the first 4 and last 4 values
    for row in image[top:-bottom]:
        left_values = row[:left]
        right_values = row[-right:]
        combined_row = np.concatenate((left_values, right_values)).tolist()
        diagnosticPixels.append(combined_row)
    
    # Step 3: Extract the last row as a separate list
    diagnosticPixels.append(image[-1].tolist())

    # Remove diagnostic pixels to create the cleaned image
    cleanedImage = image[
        top:-bottom,  # Remove top and bottom rows
        left:-right  # Remove left and right columns
    ]

    return (cleanedImage, diagnosticPixels)

def extract_diagnostic_pixels(fits_path, output_folder):

    with fits.open(fits_path) as hdul:

        # Data from fits file
        primary_HDU = hdul[0]
        img_HDU = hdul[1] # Contains the image cube (or swir readings)
        img_data = img_HDU.data # Image data
        img_header = img_HDU.header # Image HDU header
        channel = img_header.get('CHANNEL') # Channel (VIS, NIR1, NIR2, SWIR)

        # This step is not done to VIS or SWIR images
        if (channel == 'VIS' or 'SWIR'):
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
            cleanedImage, diagnostics = extract_diagnostics(image)
            
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
        fits_file = os.path.join(output_folder, file_name)

        # create the new fits file with dark-subtracted images
        hdulist = fits.HDUList(HDUs)
        hdulist.writeto(fits_file, overwrite=True)

        return(fits_file)