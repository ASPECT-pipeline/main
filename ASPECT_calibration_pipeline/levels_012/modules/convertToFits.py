import os
import numpy as np
import json
from astropy.io import fits
import modules.utilities as utilities

import matplotlib.pyplot as plt # for testing

"""

This file is for converting a folder with binary files and metadata into one FITS file that has 
all acquisitions of a single channel in a 3D-cube.

The function:
convertToFits(dirPath, outputFolder) takes as parameters the path to the directory containing the data. 
The folder at dirPath is expected to have the following structure:

dirPath/             
├── acq_000/                
│   ├── dc_1_exp_000.bin         
│   ├── dc_1_exp_001.bin         
│   └── ...                
│
└── meta/                   
    ├── calibration.json   
    ├── config.json           
    └── telemetry.json     


"""

def convert_to_fits(dir_path: str, target:str, output_dir:str) -> str:
    """
    Parmeters:
        dirPath: Path to a folder containing data of an acquisition from a single sensor.
        output: Path to the folder where the fits files will be stored.
    """
    meta_folder = os.path.join(dir_path, 'meta')
    telemetry_path = os.path.join(meta_folder, 'telemetry.json')
    config_path = os.path.join(meta_folder, 'config.json')

    acq_folder = os.path.join(dir_path, 'acq_000')

    channel_acq = utilities.collect_channel_acq_info(dir_path)

    channel_info = channel_acq['channel_info']
    ACQ_ID = channel_acq['ACQ_ID']
    ACQ_SEQ_ID = channel_acq['ACQ_SEQ_ID']

    channel_names = list(channel_info.keys()) # List of all channels in acquisition folder

    metadata = utilities.get_static_metadata() # Meta data template


    # For channel create a fits primary HDU and Image HDU / Binary HDU (SWIR)
    # Add metadata and image data and write files to output fodler
    created_fits_files = []
    for channel in channel_names:
        print(f'creating fits files for channel: {channel}, ACQ_ID: {ACQ_ID}, ACQ_SEQ_ID: {ACQ_SEQ_ID}')

        # List of HDU blocks
        HDUs = []

        # Primary HDU
        primary_hdu = fits.PrimaryHDU()
        primary_header = primary_hdu.header

        primary_header['EXTEND'] = (True, "There are extensions following this primary HDU")
        for key, (value, comment) in metadata.items():
            primary_header[key] = (value, comment)
        
        comment_insertions = [
            ('OBSTARGT', ' - - - - - - - - Instrument data - - - - - - - - '),
            ('ERRORFLG', ' - - - - - - - - Instrument specific data - - - - - - - - '),
            ('HIERARCH TEMP_TELE_1', ' - - - - - - - - SPICE data - - - - - - - - '),
            ('SOL_ELNG', ' - - - - - - - - Calibration specific data - - - - - - - - ')
        ]

        for after_key, comment_text in comment_insertions:
            if after_key in primary_header:
                primary_header.insert(after_key, ("COMMENT", comment_text), after=True)
        
        """ High level data """
        channel_original_name = channel_info[channel][0]
        # FILENAME
        # SWCREATE
        primary_header['ORIGFILE'] = channel_original_name # Original file name
        primary_header['PROCLEVL'] = '0' # Calibration level
        # MISSPHASE
        # OBSERVPH
        primary_header['OBSTARGT'] = target # observation target


        """ Instrument data """
        primary_header['OBJECT'] = target # observed object
        # AMBTEMP
        # SC_CLK

        primary_metadata = utilities.collect_primary_metadata(meta_folder=meta_folder, channel=channel)
        for key, value in primary_metadata.items():
            card = primary_header.cards[key]
            comment = card.comment
            primary_header[key] = (value, comment)


        # Spice kernel data
        spice_data = utilities.collect_spice_metadata(telemetry=telemetry_path, mk='ops',channel=channel)
        for key, value in spice_data.items():
            card = primary_header.cards[key]
            comment = card.comment
            primary_header[key] = (value, comment)

        print(repr(primary_header))
        # Generate FITS file name 
        utc_time = primary_header['DATE-OB']
        sc_clk = primary_header['SC_CLK']
        fits_name = utilities.form_fits_name(channel, sc_clk, utc_time, '0A')
        
        primary_header['FILENAME'] = fits_name
        """
        Image data
        VIS, NIR1, NIR2 -> ImageHDU extension
        SWIR -> Binary table extension
        """
        channel_files = channel_info[channel][1]
        if channel == 'SWIR':
            #Binary table
            swir_data = []
            for i, bin_file in enumerate(channel_files):
                file_path = os.path.join(acq_folder, bin_file)
                with open(file_path, 'rb') as file:
                    bin_data = file.read()
                    values = np.frombuffer(bin_data, dtype=np.uint16)
                    swir_data.append(values)
            
            # Convert the swir data into NumPy Array
            swir_array = np.array(swir_data, dtype=np.uint16)

            #Create FITS columns - one column for each file
            cols = []
            for i in range(swir_array.shape[1]):
                col_data = swir_array[:, i]
                col = fits.Column(name=f"SWIR_Frame_{i+1}", format="I", array=col_data) # I for 16-bit integers
                cols.append(col)

            #Create Binary Table HDU from the columns
            col_defs = fits.ColDefs(cols)
            hdu = fits.BinTableHDU.from_columns(col_defs)
        else:
            if channel == 'VIS':
                height = 1024
                width = 1024
            elif channel == 'NIR1' or channel == 'NIR2':
                height = 518
                width = 648
            else: 
                raise ValueError(f"Unknown channle: {channel}. Expected one of: 'VIS', 'NIR1', 'NIR', 'SWIR'." )
            
            image_data = []
            for i, bin_file in enumerate(channel_files):
                file_path = os.path.join(acq_folder, bin_file)
                decompressed_output_dir = os.path.join(dir_path, 'acq_000_decompressed')
                decompressed_output = utilities.decompress_jp2(file_path, decompressed_output_dir)
                array = np.fromfile(decompressed_output, dtype=np.uint16).reshape((height, width))
                image_data.append(array)

            data_cube = np.array(image_data) # Stack the images into a cube
            hdu = fits.ImageHDU(data_cube)
        
        # ADD metadata to fits extension header
        data_header = hdu.header
        config_data = utilities.read_config(config_path, channel)
        for key, value in config_data.items():
            data_header[key] = (value)
        
        HDUs.insert(0, primary_hdu) # insert the primary HDU to first
        HDUs.append(hdu) # append the data HDUx
        # Create a FITS file 
        file_name = fits_name
        fits_file = os.path.join(output_dir, file_name)
        hdu_list = fits.HDUList(HDUs)
        hdu_list.writeto(fits_file, overwrite=True)

        created_fits_files.append(fits_file)
    return created_fits_files