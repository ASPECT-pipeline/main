import os
import numpy as np
from astropy.io import fits
from pathlib import Path
from typing import List, Tuple
import levels_012.modules.utilities as utilities
from config import spice_mk, channel_map, reverse_channel_map, INSTRUME, ORIGIN, SWCREATE, MISSPHAS, OSERV_ID, OBJECT, SC_CLK, TARGET
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
        diff:               Either string, Path or None. Used if the files are differnetial encoded
    
    Return:
        Path: Path to the directory containing the Fits file(s) (output_dir)
    """
    dir_path = Path(dir_path)
    output_dir = Path(output_dir)

    acq_dir, telemetry_path, config_path = utilities.get_acq_tel_con(dir_path)

    orig_file_name, files = channel_info

    
    # Primary HDU
    primary_hdu = fits.PrimaryHDU()
    primary_header = primary_hdu.header

    # Get metadata template
    metadata_temp = utilities.get_header_template()
    for key, (value, comment) in metadata_temp.items():
        primary_header.append((key, value, comment))

    date = utilities.get_current_utc_time_str()

    # Add manual input data
    comment = primary_header.comments['INSTRUME'] 
    primary_header['INSTRUME'] = (INSTRUME, comment)
    comment = primary_header.comments['ORIGIN'] 
    primary_header['ORIGIN'] = (ORIGIN, comment)
    comment = primary_header.comments['MISSPHAS'] 
    primary_header['MISSPHAS'] = (MISSPHAS, comment)
    comment = primary_header.comments['OSERV_ID'] 
    primary_header['OSERV_ID'] = (OSERV_ID, comment)
    comment = primary_header.comments['ORIGFILE'] 
    primary_header['ORIGFILE'] = (orig_file_name, comment)
    comment = primary_header.comments['SWCREATE'] 
    primary_header['SWCREATE'] = (SWCREATE, comment)
    comment = primary_header.comments['DATE'] 
    primary_header['DATE'] = (date, comment)
    comment = primary_header.comments['PROCLEVL'] 
    primary_header['PROCLEVL'] = ('0A', comment)
    comment = primary_header.comments['SC_CLK'] 
    primary_header['SC_CLK'] = (SC_CLK, comment)
    comment = primary_header.comments['OBJECT'] 
    primary_header['OBJECT'] = (OBJECT, comment)
    comment = primary_header.comments['TARGET'] 
    primary_header['TARGET'] = (TARGET, comment)

    # For simulated use test time
    if MISSPHAS == 'SIMULATED':
        test = True
    else:
        test = False

    # Append spice kernel metadata
    spice_metadata = utilities.collect_spice_metadata(telemetry_path=telemetry_path, mk=spice_mk, target=TARGET, test=test)
    for key, (value, comment) in spice_metadata.items():
        primary_header[key] = (value, comment)
    
    # Add comments to help the readability
    primary_header.insert('PROCLEVL',('COMMENT', ' - - - - - - - - Instrument data - - - - - - - - '), after=True)
    primary_header.insert('SPICE_MK',('COMMENT', ' - - - - - - - - SPICE data - - - - - - - - '), after=False)
    primary_header.insert('SOL_ELNG',('COMMENT', ' - - - - - - - - Calibration specific data - - - - - - - - '), after=True)
    
    """
    Image data
    Vis, NIR1, NIR2 -> ImageHDU extension
    SWIR -> Binary table extension
    """
    if channel == 'SWIR':
        # BinaryTable HDU
        values = []
        frame_numbers = []
        for i, bin_file in enumerate(files):
            match = re.search(r'exp_(\d{3})', bin_file)
            if match:
                frame = match.group(1) # 000, 001, ...
                frame_numbers.append(frame)
            else:
                raise ValueError(f'Unknown frame number. The filename is expectected to contain the frame number after exp_.')
            file_path = Path(acq_dir) / bin_file
            try:
                with file_path.open('rb') as f:
                    bin_data = f.read()
                if len(bin_data) != 4: 
                    raise ValueError(f'SWIR file {file_path} does not contain excatly 4 bytes.')
                value = int.from_bytes(bin_data, 'big' , signed=False)
                values.append(value)
            except Exception as e:
                raise IOError(f"Error reading binary file {file_path}: {e}") from e
        array = np.array(values, dtype=np.uint32)
        primary_hdu.data = array
    else:
        # Image HDU
        if channel == 'Vis':
            height = 1024
            width = 1024
        elif channel == 'NIR1' or channel == 'NIR2':
            height = 518
            width = 648
            if MISSPHAS == 'SIMULATED':
                height = 512
                width = 640
        else: 
            raise ValueError(f"Unknown channel: {channel}. Expected one of: 'Vis', 'NIR1', 'NIR', 'SWIR'." )
        
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
                array = np.fromfile(decompressed_output, dtype='<u2').reshape((height, width)) # little-endian 16-bit unsigned
            else:
                array = np.fromfile(file_path, dtype='<u2').reshape((height, width)) # little-endian 16-bit unsigned
            image_data.append(array)


        data_cube = np.array(image_data) # Stack the images into a cube
        # Differential decoding
        if diff != None:
            print(f'Diff decoding files')
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

        primary_hdu.data = data_cube

    # Append instrument metadata
    instrument_data = utilities.collect_instrument_metadata(telemetry_path=telemetry_path, channel=channel, missphas=MISSPHAS)

    for i, (key, value) in enumerate(instrument_data.items()):
        if key in primary_header:
            comment = primary_header.comments[key]
            card_length = len(key) + len(value) + len(comment) + 4
            if card_length <= 80:
                primary_header[key] = (value, comment)
            else:
                primary_header[key] = (value, '')
        else:
            print(f"[WARNING] key '{key}' not found in header")
    primary_header.insert('ASP_ACQDATE',('COMMENT', ' - - - - - - - - Instrument specific data - - - - - - - - '), after=False)

    # Add instrument specific metadata this only for one channel at this point.
    channel_index = reverse_channel_map[channel]
    image_specific_data = utilities.collect_instrument_specific_metadata(config_path=config_path, channel=channel, frame_numbers=frame_numbers, missphas=MISSPHAS)

    inset_index = primary_header.index(f'AS{channel_index}_FPI_TEMP2')
    for i, (key, (value, comment)) in enumerate(image_specific_data.items()):
        if key in primary_header:
                primary_header[key] = (value, comment)
        else:
            card_length = len(key) + len(value) + len(comment) + 13
            if card_length <= 80:
                primary_header.insert(inset_index + i, (f'HIERARCH {key}', value, comment), after=True)
            else:
                primary_header.insert(inset_index + i, (f'HIERARCH {key}', value, ''), after=True)
    # Generate FITS file name 
    utc_time = primary_header.get('DATE-OBS')
    sc_clock = primary_header.get('SC_CLK')
    if sc_clock in (None, 'UNK'):
        sc_clock = 0
    image_number = utilities.sc_clock_to_base32(sc_seconds=sc_clock)
    fits_name = utilities.form_fits_name(channel, image_number, utc_time, '0A')
    primary_header['FILENAME'] = fits_name

    # Create a FITS file 
    file_name = fits_name
    fits_file = os.path.join(output_dir, file_name)
    primary_hdu.writeto(fits_file, overwrite=True)

    return fits_file