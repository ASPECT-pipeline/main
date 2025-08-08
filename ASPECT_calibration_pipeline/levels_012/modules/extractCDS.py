import numpy as np
from astropy.io import fits
from astropy.io.fits import HDUList
import levels_012.modules.utilities as utilities
from config import reverse_channel_map

"""
    This file extracts correlated double sampling (CDS) pixels surrounding the NIR images, storing them into a separate BinaryTableHDU.
    The function also converts the image data into double precission (float64) values. 

    Binary table architecture
        - BinaryTableHDU can be accessed with hdul[2] (third HDU after primary, and image)
            - Contains one column for each 2D image, named as, Channel_i, where channel is 'NIR1' or 'NIR2' i is the frame number
            - Each column will have 1 row with all cds pixels flattened total of 7984 values
            - utilities offer function read_cds() to retrieve the desired cds pixels from the created file

"""

def extract_cds_pixels(hdul: HDUList) -> HDUList:
    """
    Extracts the CDS pixels from NIR images and strores them in a separate BinaryTable.

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
    missphas = header.get('MISSPHAS')

    if channel in ('VIS', 'SWIR') or missphas == 'SIMULATE':
        return hdul
    elif channel in ('NIR1', 'NIR2'): 
        try:
            # Get image dimensions
            width = header.get('NAXIS1')
            height = header.get('NAXIS2')
            slices = header.get('NAXIS3')

            # Empty array for cleaned data
            cleaned_data = np.zeros((slices, height - 6, width - 8), dtype=np.float64)
            
            # Prepare a list to hold the columns for CDS binary table
            cds_list = []
            # Step 4: Iterate over each slice of the cube
            for i, image in enumerate(data):
                # Extract the cds pixels from the slice
                cleaned_image, cds = utilities.extract_cds(image)
                cds_list.append(cds)               
                # Append the cleanedImage to the new image HDU
                cleaned_data[i, :, :] = cleaned_image.astype(np.float64)

            # Create Image HDU with old header
            data = cleaned_data

            # Create columns for the cds pixels
            channel_index = reverse_channel_map[channel]
            frames = header.get(f'{channel_index}_FRAMES').split(',')
            columns = []
            for i, col in enumerate(cds_list):
                column = fits.Column(
                    name= f'{channel_index}_{frames[i]}',
                    format=f'{len(col)}J', # unsigned data
                    array=[col]
                )
                columns.append(column)

            # Create a binary table HDU for the cds pixels
            cds_table = fits.BinTableHDU.from_columns(columns)
            hdul.append(cds_table)
            return hdul
        except Exception as e:
            print(f'[WARNING] Exctracting diagnostics failed: {e}')
            return hdul
    else:
        raise ValueError(f'Channel missmatch in extract CDS: {channel}, Should be in (VIS, NIR1, NIR2, SWIR)')
