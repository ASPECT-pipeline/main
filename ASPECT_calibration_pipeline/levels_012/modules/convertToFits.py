import os
import numpy as np
from astropy.io import fits
from pathlib import Path
from typing import List, Tuple
import modules.utilities as utilities
from modules._constants import spice_mk, channel_map, software, missphase, observph, target, object
import json
import re

import matplotlib.pyplot as plt # for testing

"""

This file converts an acquisition folder into channel specific FITS files containing 
all frames from the respective channel and header metadata about the acquisition.

The acquisition directory is expected to have to follow the structure and naming below.
The acquisition directory can contain images from multiple channels.

dir_path/             
├── acq_000/                
│   ├── dc_0_exp_000.bin.jp2         
│   ├── dc_0_exp_001.bin.jp2        
│   └── ...                
│
└── meta/                   
    ├── calib.json   
    ├── config.json           
    └── telemetry.json     


"""

def convert_to_fits(
        dir_path: Path,
        output_dir: Path,
        channel: str,
        channel_info: Tuple[str, List[str]],
        diff: str | Path | None = None, 
    ) -> List[Path]:

    """
    Parmeters:
        dir_path (Path):    Directory containing the acquisition files.
        output_dir (Path):  Directory where the Fits file(s) are stored.
        channel (str):      Instrument channel name
        channel_info:       Tuple[channel name, List[file names of that channel]]
        diff:               Either string, Path or None. Used if the fiels are differnetial encoded
    
    Return:
        Path: Path to the directory containing the Fits file(s) (output_dir)
    """
    dir_path = Path(dir_path)
    output_dir = Path(output_dir)

    acq_dir, meta_dir, telemetry_path, config_path = utilities.verify_acquisition_directory(dir_path)

    orig_file_name, files = channel_info

    # List of HDU blocks
    HDUs = []

    # Primary HDU
    primary_hdu = fits.PrimaryHDU()
    primary_header = primary_hdu.header

    primary_header['EXTEND'] = (True, "There are extensions following this primary HDU")

    # Append high level primary metadata
    primary_metadata = utilities.collect_primary_metadata(swcreate=software,orig_file=orig_file_name, missphas=missphase, observph=observph, obstargt=target)
    for key, (value, comment) in primary_metadata.items():
        primary_header.append((key, value, comment))
    
    # Append instrument metadata
    instrument_metadata = utilities.collect_instrument_metadata(telemetry_path=telemetry_path, config_path=config_path, channel=channel, object=object)
    for key, (value, comment) in instrument_metadata.items():
        primary_header.append((key, value, comment))

    # Append instrument specific metadata
    instrument_specific_metadata = utilities.collect_instrument_specific_metadata(telemetry_path=telemetry_path, config_path=config_path, channel=channel)
    for key, (value, comment) in instrument_specific_metadata.items():
        primary_header.append((key, value, comment))

    # Append spice kernel metadata
    spice_metadata = utilities.collect_spice_metadata(telemetry_path=telemetry_path, mk=spice_mk, target=target)
    for key, (value, comment) in spice_metadata.items():
        primary_header.append((key, value, comment))

    # Append calibration metadata
    calibration_metadata = utilities.collect_calibration_metadata()
    for key, (value, comment) in calibration_metadata.items():
        primary_header.append((key, value, comment))

    # Add comments to help the readability
    primary_header.insert('OBSTARGT',('COMMENT', ' - - - - - - - - Instrument data - - - - - - - - '), after=True)
    primary_header.insert('ERRORFLG',('COMMENT', ' - - - - - - - - Instrument specific data - - - - - - - - '), after=True)
    primary_header.insert('SPICE_MK',('COMMENT', ' - - - - - - - - SPICE data - - - - - - - - '), after=False)
    primary_header.insert('SOL_ELNG',('COMMENT', ' - - - - - - - - Calibration specific data - - - - - - - - '), after=True)
    

    # Generate FITS file name 
    utc_time = primary_header['DATE-OB']
    sc_clk = primary_header['SC_CLK']
    fits_name = utilities.form_fits_name(channel, sc_clk, utc_time, '0A')
    
    primary_header['FILENAME'] = fits_name
    # print(repr(primary_header))

    """
    Image data
    VIS, NIR1, NIR2 -> ImageHDU extension
    SWIR -> Binary table extension
    """
    if channel == 'SWIR':
        # BinaryTable HDU
        cols = []
        frame_numbers = []
        for i, bin_file in enumerate(files):
            match = re.search(r'exp_(\d{3})', bin_file)
            if match:
                frame_numbers.append(match.group(1))
            else:
                raise ValueError(f'Unknown frame number. The filename is expectected to contain the frame number after exp_.')
            file_path = Path(acq_dir) / bin_file
            try:
                with file_path.open('rb') as f:
                    bin_data = f.read()
                if len(bin_data) != 4: 
                    raise ValueError(f'SWIR file {file_path} does not contain excatly 4 bytes.')
                value = int.from_bytes(bin_data, 'big' , signed=False)
                value = np.int32(value)
            except Exception as e:
                raise IOError(f"Error reading binary file {file_path}: {e}") from e
            col = fits.Column(name=f'SWIR_{i}', format='J', array=[value]) # J for 32-bit, I for 16-bit integers
            cols.append(col)

        # Create binary table
        hdu = fits.BinTableHDU.from_columns(cols)
    else:
        # Image HDU
        if channel == 'VIS':
            height = 1024
            width = 1024
        elif channel == 'NIR1' or channel == 'NIR2':
            height = 518
            width = 648
            if missphase == 'TEST':
                height = 512
                width = 640
        else: 
            raise ValueError(f"Unknown channel: {channel}. Expected one of: 'VIS', 'NIR1', 'NIR', 'SWIR'." )
        
        image_data = []
        frame_numbers = []
        for i, bin_file in enumerate(files):
            match = re.search(r'exp_(\d{3})', bin_file)
            if match:
                frame_numbers.append(match.group(1))
            else:
                raise ValueError(f'Unknown frame number. The filename is expectected to contain the frame number after exp_.')
            file_path = Path(acq_dir) / bin_file
            if bin_file.endswith(".jp2"):
                decompressed_output_dir = Path(dir_path) / 'acq_000_decompressed'
                decompressed_output = utilities.decompress_jp2(file_path, decompressed_output_dir)
                array = np.fromfile(decompressed_output, dtype='<u2').reshape((height, width)) # big-endian 16-bit unsigned
            else:
                array = np.fromfile(file_path, dtype='<u2').reshape((height, width)) # big-endian 16-bit unsigned

            image_data.append(array)

        data_cube = np.array(image_data) # Stack the images into a cube
        # Differential decoding
        if diff != None:
            print(f'Diff decofing files')
            differential = Path(diff)
            diff_decoded_output_dir = Path(dir_path) / 'acq_000_diff_decoded'
            with open(diff, 'r', encoding='utf-8') as f:
                diff_data = json.load(f)
            
            diff_offsets = {} # Offsets as a dictionary
            for ch_id_str, value_dict in diff_data.items():
                ch_id = int(ch_id_str)
                ch_name = channel_map.get(ch_id)
                offsets = [value_dict[k] for k in sorted(value_dict, key=int)]
                diff_offsets[ch_name] = offsets

            data_cube = utilities.diff_decode(data_cube, diff_offsets.get(channel), diff_decoded_output_dir, channel, frame_numbers)

        hdu = fits.ImageHDU(data_cube)
    # Add metadata to fits extension header

    data_header = hdu.header
    image_data = utilities.collect_image_metadata(config_path, channel)
    for key, (value, comment) in image_data.items():
        data_header.append((key, value, comment))
    
    frame_number_list = ','.join(frame_numbers)
    data_header.append(('FRAMES', frame_number_list, 'Acquisition frames'))
    HDUs.insert(0, primary_hdu) # insert the primary HDU to first
    HDUs.append(hdu) # append the data HDUx
    # Create a FITS file 
    file_name = fits_name
    fits_file = os.path.join(output_dir, file_name)
    hdu_list = fits.HDUList(HDUs)
    hdu_list.writeto(fits_file, overwrite=True)


    return fits_file