import cv2
import numpy as np
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import json
from astropy.io import fits
from astropy.io.fits import Header, PrimaryHDU, ImageHDU, BinTableHDU, HDUList
import os
import sys
import re
import levels_012.modules.hera_spice as hera_spice
from datetime import datetime, timezone
import subprocess
from pathlib import Path
from config import kelvin, channel_map, reverse_channel_map
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

def validate_pipeline_steps(steps: list[str]) -> bool:
    allowed_steps = {'1', '2', '3'}
    steps = [s.strip() for s in steps if s.strip()]
    if not steps:
        raise ValueError(f"Pipeline step list is empty. See config.py file 'pipeline' allowed steps: {allowed_steps}")
    invalid = [step for step in steps if step not in allowed_steps]
    if invalid:
        raise ValueError(f"Invalid pipeline steps found: {invalid}. See config.py file 'pipeline' allowed steps: {allowed_steps}")

def verify_directory_path(p: str | Path) -> Path:
    """
    Convert input to a Path object by pathlib python library and verify it is an ecisting directory.

    Parameters:
        p (str | Path): The path to check
    
    Returns:
        Path: the verified directory path.
    
    Raises:
        ValueError: If the path does not exist or is not a directory.
    """
    path = Path(p)

    if not path.exists():
        raise ValueError(f'Path does not exist: {path}')
    if not path.is_dir():
        raise ValueError(f'Path is not a directory: {path}')

    return path

def verify_acquisition_directory(p: Path) -> Path:
    """
    Verifies that the given acquisition directory contains:
     - a 'meta' subdirectory with 'telemetry.json' and 'config.json'
     - a subdirectory which name starts with 'acq_'

    Parameters: 
        p (Path): Path to the acquisition directory.

    Returns:
        Path to acq_dir

    Raises:
        ValueError: If any expected subdirectory or file is missing
    """
    
    if not p.exists() or not p.is_dir():
        raise ValueError(f"Provided path does not exist or is not a directory: {p}")
    
    acq_dirs = sorted(
        [d for d in p.iterdir() if d.is_dir() and d.name.startswith('acq_')]
    )
    if not acq_dirs:
        raise ValueError(f"No subsirectory starting with 'acq_' found in {p}")
    if len(acq_dirs) > 1:
            print(f"[WARNING] Multiple 'acq_' directories found in {p}. Using the first one: {acq_dirs[0].name}")
    acq_dir = acq_dirs[0]

    meta_dir = p / 'meta'
    if not meta_dir.is_dir():
        raise ValueError(f"Missing 'meta' directory in {p}")

    telemetry_file = meta_dir / 'telemetry.json'
    config_file = meta_dir / 'config.json'

    if not telemetry_file.is_file():
        raise ValueError(f"Missing 'telemetry.json' in: {meta_dir}")
    if not config_file.is_file():
        raise ValueError(f"Missing 'config.json' in: {meta_dir}")
    
    return acq_dir

def get_acq_tel_con(p: Path) -> Tuple[Path, Path, Path]:
    """
    Gets the acquistion directory, telemetry JSON and config JSON

    Parameters: 
        p (Path): Path to the acquisition directory.

    Returns:
        Tuple[Path, Path, Path]: Paths to (acq_dir, telemetery_file, and config_file)

    Raises:
        ValueError: If any expected subdirectory or file is missing
    """
    try:
        acq_dirs = sorted(
            [d for d in p.iterdir() if d.is_dir() and d.name.startswith('acq_')]
        )
        acq_dir = acq_dirs[0]
        meta_dir = p / 'meta'

        telemetry_file = meta_dir / 'telemetry.json'
        config_file = meta_dir / 'config.json'
        return(acq_dir, telemetry_file, config_file)
    except Exception as e:
        raise ValueError(f'Error while retrieving the acquisition directory: {e}')

def channel_files(acq_dir: Path) -> Dict[str, Tuple[str, List[str]]]:
    """
    Separates the acquisition folder content into separate channels based on their name.
    Generates the original filename for each channel and a list of files belonnign to that channel. 

    Parameters: 
        acq_dir (Path): Acquisition directory
    
    Returns:
        Dict[channel, Tuple[example_filename, [all_channel_filenames]]]
    """
    acq_dir = Path(acq_dir)
    # Regular expressions for channel and frame
    pattern = re.compile(r'^dc_(\d)_')
    frame_pattern = re.compile(r'exp_(\d{3})')

    # Extracts frame number for sorting. 
    def get_frame_num(fname: str) -> int:
        match = frame_pattern.search(fname)
        if match:
            return int(match.group(1))  # '003' -> 3
        raise ValueError(f"Filename does not match expected pattern: {fname}")

    channel_files: Dict[str, List[str]] = defaultdict(list)

    for file in acq_dir.iterdir():
        if not file.is_file():
            continue
        match = pattern.match(file.name)
        if match:
            index = int(match.group(1))
            if index in channel_map:
                channel_name = channel_map[index]
                channel_files[channel_name].append(file.name)


    channel_info: Dict[str, Tuple[str, List[str]]] = {}

    for channel, files in channel_files.items():
        sorted_files = sorted(files, key=get_frame_num)
        if len(sorted_files) > 1:
            orig_name = frame_pattern.sub('exp_XXX', sorted_files[0])
        else: 
            orig_name = sorted_files[0]
        channel_info[channel] = (orig_name, sorted_files)

    return channel_info

def sc_clock_to_base32(sc_seconds: int, offset: int = 0) -> str:
    """
    Convert a spacecraft time (in seconds) to an unique 6 character image number, increasing by the acquisition time.
    It generated as the base 32 coding of the image capture SC clock seconds. In case of clock counter restart, 
    the time of the restart event will be added to result a continuously increasing number.

    Parameters: 
        sc_seconds (int): The spacecraft clock time in seconds.
        offset (int): Offset to add to clock time (e.g. after clock reset). Default is 0.

    Returns:
        str: A 6-character base-32 string representing the unique image number
    """

    alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUV'
    base = 32
    value = sc_seconds + offset
    if value < 0:
        raise ValueError("Clock time + offset must be non-negative")
    result = ''
    for _ in range(6):
        result = alphabet[value % base] + result
        value //= base
    
    return result

def get_current_utc_time_str():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')

def form_fits_name(channel: str, image_number: str, utc_time: str, calib_lvl: str) -> str:
    asp_id = reverse_channel_map[channel]
    try:
        utc_format = datetime.strptime(utc_time, "%Y-%m-%dT%H:%M:%S.%f").strftime("%y%m%dT%H%M%S")
    except Exception as e:
        print(f"[WARNING] Failed to parse UTC time '{utc_time}': {e}")
        utc_format = 'XXXXXXXXXXXXX'
    if image_number == '':
        image_number = 'XXXXXX'
    file_name = f'AS{asp_id}_{image_number}_{utc_format}_{calib_lvl}.fits'
    return file_name

def get_header_template() -> Dict[str, Tuple[str, str]]: 
    metadata = {
        'INSTRUME'  : ('UNK', 'Camera ID'),
        'ORIGIN'    : ('UNK', 'HERA imaging instruments'),
        'MISSPHAS'  : ('UNK', 'HERA Mission Phase ID'),
        'OSERV_ID'  : ('UNK', 'HERA Observation ID'),
        'FILENAME'  : ('UNK', 'Name of the fits file'),
        'ORIGFILE'  : ('UNK', 'Original file name.'),
        'SWCREATE'  : ('UNK', 'Software identification'),
        'DATE'      : ('UNK', 'file creation time UTC'),
        'PROCLEVL'  : ('UNK', 'Calibration level'),
        'DATE-OBS'  : ('UNK', 'Observation time UTC'),
        'SC_CLK'    : ('UNK', 'SC clock Hera instrument format'),
        'OBJECT'    : ('UNK', 'Observation target'),
        'SPICE_MK'  : ('UNK', 'SPICE metakernel'),
        'SPICEVER'  : ('UNK', 'SPICE dataset version'),
        'SPICECLK'  : ('UNK', 'SC clock SPICE format'),
        'SUN_POSX'  : ('UNK', 'Sun position vector X [km]'),
        'SUN_POSY'  : ('UNK', 'Sun position vector Y [km]'),
        'SUN_POSZ'  : ('UNK', 'Sun position vector Z [km]'),
        'SOLAR_D'   : ('UNK', 'Solar distance'),
        'EARTPOSX'  : ('UNK', 'Earth position vector X [km]'),
        'EARTPOSY'  : ('UNK', 'Earth position vector Y [km]'),
        'EARTPOSZ'  : ('UNK', 'Earth position vector Z [km]'),
        'EARTH_D'   : ('UNK', 'Earth distance'),
        'TARGET'    : ('UNK', 'Observation target'),
        'TRG_POSX'  : ('UNK', 'Target position vector X [km]'),
        'TRG_POSY'  : ('UNK', 'Target position vector Y [km]'),
        'TRG_POSZ'  : ('UNK', 'Target position vector Z [km]'),
        'TRG_DIST'  : ('UNK', 'Target distance'),
        'SC_QUATW'  : ('UNK', 'Spacecraft quaternion (W)'),
        'SC_QUATX'  : ('UNK', 'Spacecraft quaternion (X)'),
        'SC_QUATY'  : ('UNK', 'Spacecraft quaternion (Y)'),
        'SC_QUATZ'  : ('UNK', 'Spacecraft quaternion (Z)'),
        'CAM_RA'    : ('UNK', 'Camera axis RA [deg]'),
        'CAM_DEC'   : ('UNK', 'Camera axis DEC [deg]'),
        'CAM_NAZ'   : ('UNK', 'Camera axis north azimuth [deg]'),
        'SOL_ELNG'  : ('UNK', 'Solar elongation'),
        'CALPHASE'  : ('', 'Calibration phase'),
        'HIERARCH ASP_ACQDATE'      : ('Invalid', 'Telemetry ACQ_DATE'),
        'HIERARCH ASP_CHANNELS'     : ('UNK', 'Channels in this file'),
        'HIERARCH AS0_CCDTEMP'      : ('UNK', 'Vis detector temperature [DN]'),
        'HIERARCH AS0_FPI_TEMP1'    : ('UNK', 'Vis FPI 1 temperature [DN]'),
        'HIERARCH AS0_FPI_TEMP2'    : ('UNK', 'Vis FPI 2 temperature [DN]'),
        'HIERARCH AS1_CCDTEMP'      : ('UNK', 'NIR1 detector temperature [DN]'),
        'HIERARCH AS1_FPI_TEMP1'    : ('UNK', 'NIR1 FPI 1 temperature [DN]'),
        'HIERARCH AS1_FPI_TEMP2'    : ('UNK', 'NIR1 FPI 2 temperature [DN]'),
        'HIERARCH AS2_CCDTEMP'      : ('UNK', 'NIR2 detector temperature [DN]'),
        'HIERARCH AS2_FPI_TEMP1'    : ('UNK', 'NIR2 FPI 1 temperature [DN]'),
        'HIERARCH AS2_FPI_TEMP2'    : ('UNK', 'NIR2 FPI 2 temperature [DN]'),
        'HIERARCH AS3_CCDTEMP'      : ('UNK', 'SWIR detector temperature [DN]'),
        'HIERARCH AS3_FPI_TEMP1'    : ('UNK', 'SWIR FPI 1 temperature [DN]'),
        'HIERARCH AS3_FPI_TEMP2'    : ('UNK', 'SWIR FPI 2 temperature [DN]'),
    }

    return metadata

def collect_spice_metadata(
        telemetry_path: Path, 
        mk: str | Path,  
        target: str = 'DIDYMOS', 
        test: bool = True
    )-> Dict[str, Tuple[str, str]]:
    """
    Collect specified spice kernel data for fits primary header.

    Parameters:
        telemetry (str): Path to the telemetry JSON file of the acquisition
        mk (str | Path): Path to the meta kernel file.
        target (str): Target of the observation

    Returns: 
        Dict[header_keyword, Tuple(value, comment)]
    """
    telemetry_path = Path(telemetry_path) 
    mk = str(mk)
    spice_metadata = {
        'SPICE_MK'  : ('UNK', 'SPICE metakernel'),
        'SPICEVER'  : ('UNK', 'SPICE dataset version'),
        'SPICECLK'  : ('UNK', 'SC clock SPICE format'),
        'SUN_POSX'  : ('UNK', 'Sun position vector X [km]'),
        'SUN_POSY'  : ('UNK', 'Sun position vector Y [km]'),
        'SUN_POSZ'  : ('UNK', 'Sun position vector Z [km]'),
        'SOLAR_D'   : ('UNK', 'Solar distance'),
        'EARTPOSX'  : ('UNK', 'Earth position vector X [km]'),
        'EARTPOSY'  : ('UNK', 'Earth position vector Y [km]'),
        'EARTPOSZ'  : ('UNK', 'Earth position vector Z [km]'),
        'EARTH_D'   : ('UNK', 'Earth distance'),
        'TARGET'    : (target, 'Observation target'),
        'TRG_POSX'  : ('UNK', 'Target position vector X [km]'),
        'TRG_POSY'  : ('UNK', 'Target position vector Y [km]'),
        'TRG_POSZ'  : ('UNK', 'Target position vector Z [km]'),
        'TRG_DIST'  : ('UNK', 'Target distance'),
        'SC_QUATW'  : ('UNK', 'Spacecraft quaternion (W)'),
        'SC_QUATX'  : ('UNK', 'Spacecraft quaternion (X)'),
        'SC_QUATY'  : ('UNK', 'Spacecraft quaternion (Y)'),
        'SC_QUATZ'  : ('UNK', 'Spacecraft quaternion (Z)'),
        'CAM_RA'    : ('UNK', 'Camera axis RA [deg]'),
        'CAM_DEC'   : ('UNK', 'Camera axis DEC [deg]'),
        'CAM_NAZ'   : ('UNK', 'Camera axis north azimuth [deg]'),
        'SOL_ELNG'  : ('UNK', 'Solar elongation')
    }

    try:
        telemetry_data = json.loads(telemetry_path.read_text(encoding='utf-8'))
    except Exception as e:
        print(f"[WARNING] Failed to read or parse telemetry file '{telemetry_path}': {e}")
        return spice_metadata

    utc_ob = telemetry_data['ACQ_DATE']
    if test:
        utc_ob = '2027-03-23T06:00:00.000' # Testing # Old '2025-06-15T05:40:46.6666'

    hera_spice.load_meta_kernel(mk) # Load the meta kernel

    et = hera_spice.utc_2_et(utc_ob)
    milani_frame = 'MILANI_SPACECRAFT'
    camera_frame = 'MILANI_NAVCAM'

    mk_id = hera_spice.query_mk_identifier() # Meta kernel version
    spice_metadata['SPICE_MK'] = (mk_id, 'SPICE metakernel')

    spice_version = hera_spice.query_spice_version() # SPICE dataset version
    spice_metadata['SPICEVER'] = (spice_version, 'SPICE dataset version')

    sclk = hera_spice.get_sclk(et, milani_frame) # SC clock in spice format
    spice_metadata['SPICECLK'] = (sclk, 'SC clock SPICE format')

    # Sun position vector and distnace from observer
    sun_position, sun_distance_au = hera_spice.query_position_distance(target='SUN', et=et, frame='J2000', abcorr='NONE', observer=milani_frame)
    spice_metadata['SUN_POSX'] = (sun_position[0], 'Sun position vector X [km]')
    spice_metadata['SUN_POSY'] = (sun_position[1], 'Sun position vector Y [km]')
    spice_metadata['SUN_POSZ'] = (sun_position[2], 'Sun position vector Z [km]')
    spice_metadata['SOLAR_D']  = (sun_distance_au, 'Solar distance [AU]')

    # Earth position vector and distnace from observer
    earth_position, earth_distance_au = hera_spice.query_position_distance(target='EARTH', et=et, frame='J2000', abcorr='NONE', observer=milani_frame)
    spice_metadata['EARTPOSX'] = (earth_position[0], 'Earth position vector X [km]')
    spice_metadata['EARTPOSY'] = (earth_position[1], 'Earth position vector Y [km]')
    spice_metadata['EARTPOSZ'] = (earth_position[2], 'Earth position vector Z [km]')
    spice_metadata['EARTH_D']  = (earth_distance_au, 'Earth distance [AU]')

    # IMPLEMENT THE OBSERVATION TARGET spice_metadata['TAGET'] = ...
    spice_metadata['TARGET'] = (target, 'Observation target')

    # Target position vector and distnace from observer
    target_position, target_distance_au = hera_spice.query_position_distance(target=target, et=et, frame='J2000', abcorr='NONE', observer=milani_frame)
    spice_metadata['TRG_POSX'] = (target_position[0], 'Target position vector X [km]')
    spice_metadata['TRG_POSY'] = (target_position[1], 'Target position vector Y [km]')
    spice_metadata['TRG_POSZ'] = (target_position[2], 'Target position vector Z [km]')
    spice_metadata['TRG_DIST']  = (target_distance_au, 'Target distance [AU]')

    
    # Spacecraft quaternions
    quaternions = hera_spice.query_spacecraft_quaternions(frame_name=milani_frame, et=et, tol=1, ref='J2000' ) # Quaternions are returned in format (X, Y, Z, W)
    spice_metadata['SC_QUATW'] = (quaternions[3], 'Spacecraft quaternion (W)')
    spice_metadata['SC_QUATX'] = (quaternions[0], 'Spacecraft quaternion (X)')
    spice_metadata['SC_QUATY'] = (quaternions[1], 'Spacecraft quaternion (Y)')
    spice_metadata['SC_QUATZ'] = (quaternions[2], 'Spacecraft quaternion (Z)')

    # Camera attitude
    ra_deg, dec_deg, naz_deg = hera_spice.query_camera_pointing_info(camera_frame=camera_frame, et=et, inertial_frame='J2000', target_frame='DIDYMOS_FIXED')
    spice_metadata['CAM_RA'] = (ra_deg, 'Camera axis RA [deg]')
    spice_metadata['CAM_DEC'] = (dec_deg, 'Camera axis DEC [deg]')
    spice_metadata['CAM_NAZ'] = (naz_deg, 'Camera axis north azimuth [deg]')

    # Camera solar elongation
    solar_angle = hera_spice.query_camera_solar_elongation(camera_frame=camera_frame, et=et, abcorr='NONE',observer=milani_frame)
    spice_metadata['SOL_ELNG'] = (solar_angle, 'Solar elongation')

    hera_spice.unload_all_kernels() # Unload all kernels at the end

    return spice_metadata

def check_order(sp: float, channel: str) -> str:
    """
    Checks the order of the acquisition. The order is either high or low based on the third frame's setpoint reading.

    Parameters:
        sp (float): Setpoint value
        channel (str): channel name
    
    Returns: 
        str: high / low 
    """
    match channel:
        case 'Vis' | 'NIR1':
            if sp > 19000:
                return 'HIGH'
            else:
                return 'LOW'
        case 'NIR2':
            if sp > 20000:
                return 'HIGH'
            else:
                return 'LOW'
        case 'SWIR':
            return 'N/A'

def collect_instrument_metadata(telemetry_path: Path, channel: str, missphas: str) -> Dict[str, str]:
    """
    Collect image specific metadata

    Parameters: 
        telemetry_path (Path): Path object to the telemetry JSON file
        channel (str): Instrument channel
        missphas (str): Mission Phase ID, if SIMULATED the temperatures and setpoints are set tot N/A

    Returns:
        Dict[header_keyword, Tuple(value, comment)]
    """
    meta_data = {}
    meta_data['ASP_CHANNELS'] = channel

    channels = list(reverse_channel_map.keys())
    # Load telemetry file

    try:
        telemetry_data = json.loads(telemetry_path.read_text(encoding='utf-8'))
        channel_specific_telemetries = {
            channel: telemetry_data.get(channel.upper(), {})
            for channel in channels
        }
    except Exception as e:
        print(f"[WARNING] Failed to read or parse telemetry file '{telemetry_path}': {e}")

    try:
        acq_date = telemetry_data.get('ACQ_DATE', None)
        if acq_date:
            dt = datetime.strptime(acq_date, "%a %b %d %H:%M:%S %Y")
            meta_data['DATE-OBS'] = dt.strftime("%Y-%m-%dT%H:%M:%S.000")
            meta_data['ASP_ACQDATE'] = dt.strftime("%Y-%m-%dT%H:%M:%S.000")
        else:
            print(f"[WARNING] 'ACQ_DATE' missing in telemetry file.")
    except Exception as e:
        print(f"[WARNING] Failed to parse 'ACQ_DATE': {e}")

    for ch in channels:
        channel_specific_telemetry = channel_specific_telemetries[ch]
        ch_id = reverse_channel_map[ch]
        try: 
            det_temp = channel_specific_telemetry['DET_TEMP']
            meta_data[f'AS{ch_id}_CCDTEMP'] = str(det_temp)
        except KeyError:
            print(f"[WARNING] 'DET_TEMP' missing for channel '{ch}' in telemetry.")
            meta_data[f'AS{ch_id}_CCDTEMP'] = 'UNK'
        try:
            fpi_temp1 = channel_specific_telemetry['FPI_TEMP1']
            meta_data[f'AS{ch_id}_FPI_TEMP1'] = str(fpi_temp1)
        except KeyError:
            print(f"[WARNING] 'FPI_TEMP1' missing for channel '{ch}' in telemetry.")
            meta_data[f'AS{ch_id}_FPI_TEMP1'] = 'UNK'
        try:
            fpi_temp2 = channel_specific_telemetry['FPI_TEMP2']
            meta_data[f'AS{ch_id}_FPI_TEMP2'] = str(fpi_temp2)
        except KeyError:
            print(f"[WARNING] 'FPI_TEMP2' missing for channel '{ch}' in telemetry.")
            meta_data[f'AS{ch_id}_FPI_TEMP2'] = 'UNK'

        if missphas == 'SIMULATED':
            meta_data[f'AS{ch_id}_CCDTEMP'] = 'N/A'
            meta_data[f'AS{ch_id}_FPI_TEMP1'] = 'N/A'
            meta_data[f'AS{ch_id}_FPI_TEMP2'] = 'N/A'
            
    return meta_data

def collect_instrument_specific_metadata(config_path: Path, channel: str, frame_numbers: list[str], missphas: str) -> Dict[str, Tuple[str, str]]:
    """
    Collect instrument specific metadata from config.json file. 
    This includes Exposure times and setpoint values. Also calculate the order of acquisition.

    Parameters: 
        config_path (Path): Path object to the config JSON file
        channel (str): Instrument channel
        frame_number_string (str): All frame numebr e.g. '000', '001' as a comma-separated string.
        missphas (str): Mission Phase ID, if SIMULATED the temperatures and setpoints are set tot N/A

    Returns:
        Dict[key, (value, comment)]
    """

    meta_data = {}
    channel_id = reverse_channel_map[channel]
    frame_number_string = ','.join(frame_numbers)
    meta_data[f'AS{channel_id}_FRAMES'] = (frame_number_string, f'{channel} frames')

    try: 
        config_data = json.loads(config_path.read_text(encoding='utf-8'))
    except Exception as e:
        print(f"[WARNING] Failed to read or parse config file '{config_path}': {e}")
        config_data = {}

    #read SP values for each image
    try:
        match channel:
            case 'Vis':
                taskFile = config_data['visTaskFile']
            case 'NIR1':
                taskFile = config_data['nir1TaskFile']
            case 'NIR2':
                taskFile = config_data['nir2TaskFile']
            case 'SWIR':
                taskFile = config_data['swirTaskFile']
    except KeyError as e:
        print(f"[WARNING] Missing task file entry for channel '{channel}' in config.json'")
        taskFile = []
    try:
        #Extract sp values from taskValues
        taskValues = [taskFile[i:i + 8] for i in range(0, len(taskFile), 8)]
        sp_expos_values = [task[1:5] for task in taskValues]
        task_number = len(taskValues)
        #Check the order based on SP1 index 3
        if task_number > 3:
            order = check_order(taskValues[3][0], channel)
        elif task_number >= 1:
            print(f"[WARNING] less than 4 acquisitions, the order determined from setpoint index < 3.")
            order = check_order(taskValues[-1][0], channel)
        else:
            print(f"[WARNING] Not enough setpoint values to determine order.")
            order = 'UNK'
    except Exception as e:
        print(f'[WARNING] Failed to parse task file values: {e}')
        sp_expos_values = ['UNK, UNK, UNK, UNK'] * len(frame_numbers)
        order = 'UNK'
    
    meta_data[f'AS{channel_id}_ORDER'] = (str(order), 'LOW / HIGH')
    meta_data[f'AS{channel_id}_TASK_NUMBER'] = (str(task_number), f'Number of {channel} imiging tasks')
    if task_number != len(frame_numbers):
        print(f'[WARNING] The number of tasks is different to number of frames')
    for i, task in enumerate(sp_expos_values):
        n = taskValues[i][0]
        num = f'{n:03d}' # e.g. 1 -> 001
        meta_data[f'AS{channel_id}_TASK_{num}'] = (' '.join(str(x) for x in task), 'SP1 SP2 SP3 ExpDn')

    # if missphas == 'SIMULATED':
    #     meta_data[f'{channel_id}_ORDER'] = ('N/A', 'LOW / HIGH')
    #     meta_data[f'{channel_id}_EXPOS'] = ('N/A', f'{channel} Exposure time(s) [DN]')
    #     meta_data[f'{channel_id}_SP1'] = ('N/A', f'{channel} setpoints 1 [DN]')
    #     meta_data[f'{channel_id}_SP2'] = ('N/A', f'{channel} setpoints 2 [DN]')
    #     meta_data[f'{channel_id}_SP3'] = ('N/A', f'{channel} setpoints 3 [DN]')

    return meta_data

def decompress_jp2(input_path: str | Path, output_dir: str | Path) -> Path:
    """
    Decompress a JPEG2000 .jp2 image using the C-based './decompress' program.

    Parameters:
        input_path (str | Path): Path object to the input .jp2 file.
        output_dir (str | Path): Directory to store the output .bin file.
    
    Returns:
        Path (Path): Path object to the decompressed .bin file.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True) # Cretate the output directory if does not exist
    if input_path.suffix == '.jp2':
        output_filename = input_path.stem 
    else:
        raise ValueError("Input file does not end with .jp2")

    output_path = Path(output_dir) / output_filename 

    # Resolve absolute path to decompress binary (relative to this script's location)
    script_dir = Path(__file__).resolve().parent
    decompress_path = Path(script_dir) / "decompress"

    with open(input_path, "rb") as f_in, open(output_path, "wb") as f_out:
        subprocess.run([str(decompress_path)], stdin=f_in, stdout=f_out, stderr=subprocess.DEVNULL, check=True)
    
    return output_path

def diff_decode(image_cube: np.ndarray, offsets: list[int], output_dir: str, channel: str, frame_numbers: list[str]):
    assert image_cube.ndim == 3, "Expected 3D array"
    gaps, height, width = image_cube.shape

    assert gaps == len(offsets), "Offset list must match number of frames"

    # Prepare raw binary input
    raw_bytes = image_cube.astype('<u2').tobytes() # Little-endian uint16
    offset_str = ','.join(str(o) for o in offsets)

    # Resolve absolute path to decompress binary (relative to this script's location)
    script_dir = Path(__file__).resolve().parent
    decoder_path = Path(script_dir) / "diff_bpp_decode"
    
    proc = subprocess.Popen(
        [decoder_path, '-w', str(width), '-h', str(height), '-g', str(gaps), offset_str],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    decoded_output, stderr = proc.communicate(input=raw_bytes)

    if proc.returncode != 0:
        print(stderr.decode(), file=sys.stderr)
        raise RuntimeError("Differential decoder failed.")
    
    decoded_cube = np.frombuffer(decoded_output, dtype='<u2').reshape((gaps, height, width))

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    for i, frame in enumerate(decoded_cube):
        out_path = output_dir / f'{channel}_decoded_{frame_numbers[i]}.bin'
        with open(out_path, 'wb') as f:
            f.write(frame.astype('<u2').tobytes())  # little-endian 16-bit

    return decoded_cube

def det_temp_conversion(value: float, channel: str) -> Tuple[str, str]:
    """
    Converts the 'DET_TEMP' entries from telemetry to Celcius and Kelvin. Not available for NIR-channels

    Parameters:
        value (float): Detector temperature DN value
        channel (str): Instrument channel

    Returns:
        Tuple(Celsius, Kelvin)
    """
    match channel:
        case 'Vis':
            if value == 0:
                return ('UNK', 'UNK')
            c = value * 0.6522 - 295.87
            k = c + kelvin
            return (f'{c:.2f}', f'{k:.2f}')
        case 'NIR1': return ('N/A', 'N/A')
        case 'NIR2': return ('N/A', 'N/A')
        case 'SWIR':
            if value == 0:
                return ('UNK', 'UNK')
            c = (-6e-11) * value**3 + 3e-6 * value**2 - 0.0188 * value + 17.291
            k = c + kelvin
            return (f'{c:.2f}', f'{k:.2f}')

def fpi_temp_conversion(value:float, channel: str, fpi: int) -> Tuple[str, str]:
    """
    Converts the FPI temperatures to Celcius and Kelvin. Not available for Vis-channel

    Parameters:
        value (float): Temperature DN value
        channel (str): Instrument channel

    Returns:
        Tuple(Celsius, Kelvin)
    """
    match channel:
        case 'Vis':
            return ('N/A', 'N/A')
        case ('NIR1' | 'NIR2' | 'SWIR'):
            if value == 0:
                return ('UNK', 'UNK')
            if fpi == 1:
                c = -0.034 * value + 110.93
            else:
                c = -0.026 * value + 81.01
            k = c + kelvin
            return (f'{c:.2f}', f'{k:.2f}')

def exposure_conversion(values: float, channel: str, task_number: int) -> float:
    """
    Converts exposure DN into seconds.

    Parameters:
        value (float): exposure DN
        channel (str): channel
    
    Returns: 
        float exposure in seconds
    """
    exposures = []
    # Conversion from DN to s
    match channel:
        case 'Vis':
            for value in values:
             x = ((value + 8.6123) / 155.04) / 1000
             exposures.append(f"{x:.6f}")
        case 'NIR1': 
            for value in values:
                x = value / 100000
                exposures.append(f"{x:.6f}")
        case 'NIR2': 
            for value in values:
                x = value / 100000
                exposures.append(f"{x:.6f}")
        case 'SWIR': 
            for value in values:
                x = value
                exposures.append(f"{x:.6f}")
    
    channel_id = reverse_channel_map[channel]
    ex_dict = {}

    for i in range(0, task_number):
        num = f'{i:03d}' # e.g. 1 -> 001
        ex_dict[f'AS{channel_id}_TASK_{num}_EXPOS'] = (str(exposures[i]), f'{channel} task {num} exposure [s]')
    
    return ex_dict

def wavelength_conversion(channel: str, order: str, sp_values: List[float], task_number: int, simulated: bool) -> Dict[str, Tuple[str, str]]:
    """
    Calculates the wavelengths from setpoint values

    Parameters:
        channel: instrument channel
        order: order of the acquisition (HIGH / LOW)
        sp_values: Piezo actuator setpoint values
        taks_number: number of tasks
        simulated: True if the data is simulated othewise False

    Returns:
        Dict[header entry, (wl, comment)]
    """
    channel_id = reverse_channel_map[channel]
    wavelengths = []
    
    if simulated:
        match (channel):
            case 'Vis':
                wavelengths = [675,690,705,720,735,750,765,780,795,810,825]
            case 'NIR1':
                wavelengths = [875,904,933,963,992,1021,1050,1079,1108,1138,1167,1196,1225]
            case 'NIR2':
                wavelengths = [1225,1254,1283,1313,1342,1371,1400,1429,1458,1488,1517,1546,1575]
            case 'SWIR':
                wavelengths = [1675, 1711, 1748, 1784, 1820, 1857, 1893, 1930, 1966, 2002, 2039, 2075, 2111, 2148, 2184, 2220, 2257, 2293, 2330, 2366, 2402, 2439, 2475]
    else:
        ## Before the correct values are added
        # unk_dict = {}
        # for i in range(0, len(frames)):
        #     unk_dict[f'{channel_id}_WL{frames[i]}'] = ('UNK', f'task setpoint: {sp_values[i]}')
        # return unk_dict
    
        match (channel, order):
            # The correct values for the corretion needed
            case 'Vis', 'HIGH':
                for i in range(0, len(sp_values)):
                    wavelength = round(0.0749 * sp_values[i] - 786.9)
                    wavelengths.append(wavelength)
            case 'Vis', 'LOW':
                for i in range(0, len(sp_values)):
                    wavelength = round(0.1244 * sp_values[i] - 1498.2)
                    wavelengths.append(wavelength)
            case 'NIR1', 'HIGH':
                for i in range(0, len(sp_values)):
                    wavelength = round(0.1331 * sp_values[i] - 1823.1)
                    wavelengths.append(wavelength)
            case 'NIR1', 'LOW':
                for i in range(0, len(sp_values)):
                    wavelength = round(0.2379 * sp_values[i] - 3190.5)
                    wavelengths.append(wavelength)
            case 'NIR2', 'HIGH':
                for i in range(0, len(sp_values)):
                    wavelength = round(0.1293 * sp_values[i] - 1619.4)
                    wavelengths.append(wavelength)
            case 'NIR2', 'LOW':
                for i in range(0, len(sp_values)):
                    wavelength = round(0.2366 * sp_values[i] - 2925.8)
                    wavelengths.append(wavelength)
            case 'SWIR', '':
                for i in range(0, len(sp_values)):
                    wavelength = round(0.2869 * sp_values[i] - 3847.2)
                    wavelengths.append(wavelength)
            case _ , _:
                print(f"[WARNING] invalid channel '{channel}' / order '{order}' inside wavelength conversion.")
                unk_dict = {}
                for i in range(0, task_number):
                    num = f'{i:03d}' # e.g. 1 -> 001
                    unk_dict[f'AS{channel_id}_WL_{num}'] = ('UNK', f'{channel} task {num} wavelength [nm]')
                return unk_dict
    
    wl_dict = {}
    for i in range(0, task_number):
        num = f'{i:03d}' # e.g. 1 -> 001
        wl_dict[f'AS{channel_id}_WL_{num}'] = (str(wavelengths[i]), f'{channel} task {num} wavelength [nm]')

    return wl_dict

def is_valid_fits_file(path:str) -> Tuple[bool, Optional[str]]:
    path = Path(path)
    if not path.exists():
        return False, f"File not found: {path}"
    if not path.is_file():
        return False, f"Path is not a file: {path}"
    if path.suffix.lower() != '.fits':
        return False, f"File does not have a .fits extension: {path}"
    
    return True, None

def combine_primary_headers(headers: list[Header]) -> Header:
    if not headers: 
        raise ValueError('No headers provided for combination')
    if len(headers) == 1:
        print(f"[WARNING] Only one header to combine in level 2.")
        return headers[0]
    
    combined_header = headers[0].copy()
    combined_header['ORIGFILE'] = ('dc_X_exp_XXX.bin', 'Original file name.')
    combined_header['PROCLEVL'] = ('2B', 'Calibration level')

    def copy_header_key(source: Header, target: Header, key: str, id: int):
        if key in source:
            value = source[key]
            comment = source.comments[key]
            if key in target:
                target[key] = (value, comment)
            else:
                insert_idx = target.index(f'{id}_FPI2')
                target.insert(insert_idx, (key, value, comment), after=True)
        else:
            channel = source.get('CHANNELS')
            print(f"[INFO] Key '{key}' not found in source '{channel}' header; skipping.")

    
    try:
        channel_0 = combined_header.get('CHANNELS')
        channels = [channel_0]
        for hdr in headers[1:]:
            channel = hdr.get('CHANNELS')
            channels.append(channel)
            channel_id = reverse_channel_map[channel]
            copy_header_key(hdr, combined_header, f'{channel_id}_WL', channel_id)
            copy_header_key(hdr, combined_header, f'{channel_id}_SP3', channel_id)
            copy_header_key(hdr, combined_header, f'{channel_id}_SP2', channel_id)
            copy_header_key(hdr, combined_header, f'{channel_id}_SP1', channel_id)
            copy_header_key(hdr, combined_header, f'{channel_id}_EXPOS', channel_id)
            copy_header_key(hdr, combined_header, f'{channel_id}_ORDER', channel_id)
            copy_header_key(hdr, combined_header, f'{channel_id}_frames', channel_id)
            copy_header_key(hdr, combined_header, f'{channel_id}_CCDTMP', channel_id)
            copy_header_key(hdr, combined_header, f'{channel_id}_FPI1', channel_id)
            copy_header_key(hdr, combined_header, f'{channel_id}_FPI2', channel_id)
        channels_string = ','.join(channels)
        combined_header['CHANNELS'] = (channels_string, 'Instrument channels')
    except Exception as e:
        print(f"[WARNING] Combining headers failed: {e}")
        print(f'Returning the first header.')
        headers[0]
    return combined_header

def append_header(hdu: ImageHDU, dict: Dict[str, Any]) -> ImageHDU:
    header = hdu.header
    for key, value in dict.items():
        header[key] = value
    return hdu

def normalize_to_8bit(img: np.ndarray) -> np.ndarray:
    # Compute min and max values in the image
    min_val = np.min(img)
    max_val = np.max(img)

    # Avoid division by zero in case all values are the same
    if max_val - min_val == 0:
        return np.zeros_like(img, dtype=np.uint8)

    # Normalize image to range 0-255
    normalized = (img - min_val) / (max_val - min_val) * 255.0

    # Convert to 8-bit integer
    return normalized.astype(np.uint8)

def convert_to_float64(hdul: HDUList, index: int = 0) -> HDUList:
    # Replace the .data with a float64
    hdu = hdul[index]
    if hdu.data is not None and np.issubdtype(hdu.data.dtype, np.number):
        hdu.data = hdu.data.astype(np.float64)
        if 'BITPIX' in hdu.header:
            hdu.header['BITPIX'] = -64
    else:
        print(f'[WARNING] HDU data is None or not convertable to float64')
    return hdul

def convert_to_float32(hdul: HDUList, index: int = 0) -> HDUList:
    # Replace the data with a float32 
    hdu = hdul[index]
    if hdu.data is not None and np.issubdtype(hdu.data.dtype, np.number):
        hdu.data = hdu.data.astype(np.float32)
        if 'BITPIX' in hdu.header:
            hdu.header['BITPIX'] = -32
    else:
        print(f'[WARNING] HDU data is None or not convertable to float32')
    return hdul

def extract_cds(image: np.ndarray) -> Tuple[np.ndarray, List[List[int]]]:
    # Define diagnostic pixel regions
    top = 5  # Five lines at the top
    bottom = 1  # One line at the bottom
    left = 4  # Four columns on the left
    right = 4  # Four columns on the right

    top_rows = image[:top, :]
    middle_rows = image[top:-bottom, :]
    left_cols = middle_rows[:, :left]
    right_cols = middle_rows[:, -right:]
    middle_sides = np.hstack((left_cols, right_cols))

    bottom_row = image[-1:, :]

    cds_pixels = np.concatenate([
        top_rows.flatten(),
        middle_sides.flatten(),
        bottom_row.flatten()
    ])

    # Remove diagnostic pixels to create the cleaned image
    cleanedImage = image[
        top:-bottom,  # Remove top and bottom rows
        left:-right  # Remove left and right columns
    ]
    return (cleanedImage, cds_pixels)

def read_cds(column: np.ndarray, row_inx: int, col_inx: int, count: int ) -> np.ndarray:
    """
    Read CDS pixels from a given column (Frame) give nthe desired range.
    If CDS pixels are desired from rows 5-516 the col_inx must be 0-7 such that 0-3 are the left side and 4-7 are the right side CDS.

    Parameters:
        column (np.ndarray): The column of the BintableHDU containing CDS pixels from the desired frame
        row_inx (int): The row that the pixels are wanted from (0-517)
        col_inx (int): The column that the range starts from (0-647) for rows 5-516 only 8 values!
        count (int): Count of how many values are wanted from the starting location

    Returns:
        np.ndarray: Array containing the desired CDS pixels.
    """
    if row_inx < 5:
        row_offset = row_inx*648
        total_offset = row_offset + col_inx 
        return column[total_offset : total_offset + count]
    elif row_inx < 517:
        row_offset = 5*648 + (row_inx - 5)* 8
        total_offset = row_offset + col_inx 
        return column[total_offset : total_offset + count]
    else:
        return column[-col_inx: col_inx + count]
    
def laplacian(img: np.ndarray) -> np.ndarray:

    # Check if the image was loaded successfully
    if img is None:
        print("Error: Image not found or unable to open")
    
    # Normalize the image to 8 bit integers
    img = normalize_to_8bit(img)

    # Apply gaussian blur
    img = cv2.GaussianBlur(img, (3, 3), sigmaX=0, sigmaY=0)

    # Apply Laplacian operator
    laplacian = cv2.Laplacian(img, cv2.CV_8U, ksize=3) 

    return laplacian

def filter_by_orientation(matches, keypoints1, keypoints2, threshold=10) -> List[cv2.DMatch]:
    filtered_matches = []
    for m in matches:
        angle1 = keypoints1[m.queryIdx].angle
        angle2 = keypoints2[m.trainIdx].angle
        angle_diff = abs(angle1 - angle2)
        if angle_diff < threshold or angle_diff > (360 - threshold):
            filtered_matches.append(m)
    return filtered_matches

def filter_by_distance(matches: List[List[cv2.DMatch]]) -> List[cv2.DMatch]:
    ratio_thresh = 0.90  # Adjustable
    good_matches = []
    for m in matches:
        if len(m) == 2:
            match1, match2 = m
            if match1.distance < ratio_thresh * match2.distance:
                good_matches.append(match1)
        elif len(m) == 1:
            match1 = m[0]
            good_matches.append(match1)
    distance_thresh = 65  # Adjustable
    good_matches = [m for m in good_matches if m.distance < distance_thresh]
    return good_matches

def estimate_matrix(vis: np.ndarray, nir: np.ndarray, filter: bool = True) -> np.ndarray:
    """
    if filter = true, the feature matches are filtered byt the distance between the 1st and 2nd match suggestion. (This method is recommended)
    if filter = false, the matches are filtered based on the angle of the keypoint calculated by ORB
    """
    # Step 1: Edge detection
    edges1 = laplacian(vis)
    edges2 = laplacian(nir)

    # Step 2: Feature detection using ORB
    orb = cv2.ORB_create(nfeatures=5000) # create ORB feature detector
    # keypoints and binary descriptions
    keypoints1, descriptors1 = orb.detectAndCompute(edges1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(edges2, None)

    # Step 3: Match features
    # FLANN
    index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                    table_number=30,  # Number of hash tables
                    key_size=20,     # Size of the key
                    multi_probe_level=2)  # Number of probes
        
    search_params = dict(checks=100)

    flann = cv2.FlannBasedMatcher(index_params, search_params) # Initialize the FLANN

    flann_matches = flann.knnMatch(descriptors1, descriptors2, k=2) # Match features

    #Filter the matches based on the distance. Other option is filter_by_orientation
    if filter: 
        matches = filter_by_distance(flann_matches)
    else:
        matches = filter_by_orientation(flann_matches)


    # Step 4: Extract location of good matches and estimate transformation matrix
    # arrays to store x and y coordinates
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    #Extract keypoint coordinates 
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Estimate transformation matrix
    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 10.0)

    
    #Return the transformation matrix
    return(H)

def cropND(img: np.ndarray, bounding: tuple[int, int]) -> np.ndarray:
    start = tuple(map(lambda a, da: (a - da) // 2, np.shape(img), bounding))
    end = tuple(map(np.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]

def get_simulated_wl(channel: str) -> str:
    wl_map = {
        'Vis' : '675,690,705,720,735,750,765,780,795,810,825',
        'NIR1': '875,904,933,963,992,1021,1050,1079,1108,1138,1167,1196,1225',
        'NIR2': '1225,1254,1283,1313,1342,1371,1400,1429,1458,1488,1517,1546,1575',
        'SWIR': '1675,1711,1748,1784,1820,1857,1893,1930,1966,2002,2075,2111,2148,2184,2220,2257,2293,2330,2366,2402,2439,2475'
    }
    return wl_map[channel]

"""
Testing functions that can be removed 
"""

def estimate_bit_depth(binary_file, width, height):
    print(f'file: {binary_file}')
    file_size = os.path.getsize(binary_file)
    pixels = width * height
    print(f'file size: {file_size}')
    print(f'pixels: {pixels}')
    bytes_per_pixel = file_size / pixels
    print(f'bytes per pixel: {bytes_per_pixel}')
    with open(binary_file, 'rb') as file:
            binaryData = file.read()
            image = np.frombuffer(binaryData, dtype='<u2')
            print(f'Image Array:')
            image = image.reshape((height, width))
            bit_depth = int(np.ceil(np.log2(np.max(image) + 1)))
    return bytes_per_pixel, bit_depth

def rename_bin_files(directory: str | Path):

    directory = Path(directory)

    for file in directory.iterdir():
        if file.is_file():
            match = re.search(r'-(Vis|NIR1|NIR2|SWIR)-', file.name)
            if not match:
                print(f'Skipping unrecognized channel: {file.name}')
                continue

            channel_name = match.group(1)
            channel_id = channel_map[channel_name]

            try:
                frame_str = file.stem[-3:]
                int(frame_str)
            except ValueError:
                print(f'Skipping file with invalid frame number: {file.name}')
                continue

            new_name = f'dc_{channel_id}_exp_{frame_str}.bin'
            new_path = file.with_name(new_name)

            file.rename(new_path)
            print(f'Renamed: {file.name} -> {new_name}')

def overlay_images(image1, image2, mode='red-green', title='Image Overlay'):
    """
    Overlay two aligned grayscale images using RGB channels to visualize alignment.

    Parameters:
    - image1, image2: 2D NumPy arrays (grayscale images)
    - mode: 'red-green' or 'red-blue' (channel assignment)
    - title: Title for the plot
    """
    # Normalize both images to [0, 1]
    def normalize(img):
        img = img.astype(np.float32)
        return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)

    img1_norm = normalize(image1)
    img2_norm = normalize(image2)

    # Create RGB composite
    rgb = np.zeros((*image1.shape, 3), dtype=np.float32)

    if mode == 'red-green':
        rgb[..., 0] = img1_norm  # Red
        rgb[..., 1] = img2_norm  # Green
    elif mode == 'red-blue':
        rgb[..., 0] = img1_norm  # Red
        rgb[..., 2] = img2_norm  # Blue
    else:
        raise ValueError("Mode must be 'red-green' or 'red-blue'.")

    return(rgb)

def plot_spectra_with_image(spectra_list, positions, image, all_wavelengths):
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
    labels = [f'({x}, {y})' for (x, y) in positions]

    # Create a 2x2 grid: top row is spectra/image, bottom row is just for labels under the image
    fig = plt.figure(figsize=(14, 7))
    # Reduce image column width (2 instead of 3), reduce label row height (0.6)
    gs = gridspec.GridSpec(1, 2, width_ratios=[6, 2], wspace=0.15)


    ax_spectra = fig.add_subplot(gs[0, 0])
    ax_image = fig.add_subplot(gs[0, 1])
    ax_image.set_xticks([])
    ax_image.set_yticks([])

    # Plot each spectrum with labels
    for spec, color, label in zip(spectra_list, colors, labels):
        ax_spectra.plot(all_wavelengths, spec, color=color, label=label)

    ax_spectra.set_xlabel("Wavelength (nm)")
    ax_spectra.set_ylabel("Intensity (DN / bit depth)")
    ax_spectra.set_title("Spectra from selected pixels", pad=10)

    # Add inline legend inside the plot (bottom-left corner)
    ax_spectra.legend(
        loc='lower right',
        fontsize=9,
        frameon=False,
        title="Pixel (x, y)",
        title_fontsize=10
    )

    # Show image and plot markers
    ax_image.imshow(image, cmap='gray' if image.ndim == 2 else None)
    ax_image.set_title("Selected pixel locations", pad=10)
    for (x, y), color in zip(positions, colors):
        ax_image.plot(x, y, 'o', color=color, markersize=6)

    return fig

def update_fits_exposure(path, save_as=None):
    exposure_map = {
        'Vis' : 0.01,
        'NIR1': 0.02,
        'NIR2': 0.02,
        'SWIR': 0.02
    }
    with fits.open(path, mode='update' if save_as is None else 'readonly') as hdul:
        channel = hdul[0].header.get('CHANNEL')
        for hdu in hdul:
            if 'EXPOSURE' in hdu.header:
                print(f"Old EXPOSURE: {hdu.header['EXPOSURE']}")
                hdu.header['EXPOSURE'] = exposure_map[channel]
                print(f"New EXPOSURE: {hdu.header['EXPOSURE']}")

        if save_as:
            hdul.writeto(save_as, overwrite=True)
            print(f"Saved updated file to {save_as}")
        else:
            print(f"Updated in place: {path}")

def update_fits_wl(path, save_as=None):
    wl_map = {
        'Vis' : '675,690,705,720,735,750,765,780,795,810,825',
        'NIR1': '875,904,933,963,992,1021,1050,1079,1108,1138,1167,1196,1225',
        'NIR2': '1225,1254,1283,1313,1342,1371,1400,1429,1458,1488,1517,1546,1575',
        'SWIR': '1675,1711,1748,1784,1820,1857,1893,1930,1966,2002,2075,2111,2148,2184,2220,2257,2293,2330,2366,2402,2439,2475'
    }
    with fits.open(path, mode='update' if save_as is None else 'readonly') as hdul:
        channel = hdul[0].header.get('CHANNEL')
        for hdu in hdul:
            if 'WAVELEN' in hdu.header:
                print(f"Old WL: {hdu.header['WAVELEN']}")
                hdu.header['WAVELEN'] = wl_map[channel]
                print(f"New WL: {hdu.header['WAVELEN']}")

        if save_as:
            hdul.writeto(save_as, overwrite=True)
            print(f"Saved updated file to {save_as}")
        else:
            print(f"Updated in place: {path}")