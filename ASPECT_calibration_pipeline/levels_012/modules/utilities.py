import cv2
import numpy as np
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import json
from astropy.io.fits import Header, PrimaryHDU, ImageHDU, BinTableHDU
import os
import re
import modules.hera_spice as hera_spice
from datetime import datetime, timezone
import subprocess
from pathlib import Path
import warnings
from modules._constants import kelvin
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

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
def verify_acquisition_directory(p: Path) -> Tuple[Path, Path, Path, Path]:
    """
    Verifies that the given acquisition directory contains:
     - a 'meta' subdirectory with 'telemetry.json' and 'config.json'
     - a subdirectory whose name starts with 'acq_'

    Parameters: 
        p (Path): Path to the acquisition directory.

    Returns:
        Tuple[Path, Path, Path, Path]: Paths to (acq_dir, meta_dir, telemetery_file, and config_file)

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
        warnings.warn(
            f"Multiple 'acq_' directories found in {p}. Using the first one: {acq_dirs[0].name}\nModify convertToFits and utilities/verify_acquisition_directory to handle more acquisitions at once",
            category=UserWarning
        )
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
    
    return(acq_dir, meta_dir, telemetry_file, config_file)

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
    channel_map = {
        0: 'VIS',
        1: 'NIR1',
        2: 'NIR2',
        3: 'SWIR'
    }
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

def get_current_utc_time_str():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')

def form_fits_name(channel: str, sc_clk: str, utc_time: str, calib_lvl: str) -> str:
    channel_map = {
            'VIS'   : 0,
            'NIR1'  : 1,
            'NIR2'  : 2,
            'SWIR'  : 3
        }
    asp_id = channel_map[channel]
    utc_format = datetime.strptime(utc_time, "%Y-%m-%dT%H:%M:%S.%f").strftime("%y%m%dT%H%M%S")
    if sc_clk == '':
        sc_clk = 'XXXXXX'
    file_name = f'AS{asp_id}_{sc_clk}_{utc_format}_{calib_lvl}.fits'
    return file_name

def collect_primary_metadata(
        swcreate: str, 
        orig_file: str, 
        missphas: str, 
        observph: str, 
        obstargt: str
    ) -> Dict[str, Tuple[str, str]]:
    """
    Generates the high level FITS metadata headers for primary HDU

    Parameters:
        swcreate (str): Software id 
        orig_file (str): original filename
        missphas (str): mission phase ID
        observph (str): Observation ID
        obstargt (str): observation target
    Returns: 
        Dict[header_keyword, Tuple(value, comment)]
    """
    date = get_current_utc_time_str()

    metadata = {
        'INSTRUME' : ('ASPECT', 'Camera ID'),
        'ORIGIN'   : ('ESA-HERA', ''),
        'DATE'     : (date, 'UTC time of file creation'),
        'FILENAME' : ('', 'Name of the actual fits file'), # generated later by form_fits_name. needs sc clock count.
        'SWCREATE' : (swcreate, 'Software identification'),
        'ORIGFILE' : (orig_file, 'Original file name.'),
        'PROCLEVL' : ('0', 'Calibration level'),
        'MISSPHAS' : (missphas, 'HERA Mission Phase ID'),
        'OBSERVPH' : (observph, 'HERA Observation ID'),
        'OBSTARGT' : (obstargt, 'Observation target'),
    }

    return metadata

def collect_instrument_metadata(
        telemetry_path: Path, 
        config_path: Path, 
        channel: str,
        object: str
    ) -> Dict[str, Tuple[str, str]]:
    """
    Collects instrument metadata.

    Parameters:
        telemetry_path (Path): Path object to the telemetry file
        config_path (Path): Path object to the config file
        channel (str): Channel information to look for
        object (str): Name for the observed object
    
    Returns:
         Dict[header_keyword, Tuple(value, comment)]
    """

    telemetry_path = Path(telemetry_path)
    config_path = Path(config_path)

    telemetry_data = json.loads(telemetry_path.read_text(encoding='utf=8')) # Telemetry etries
    channel_specific_telemetry = telemetry_data[channel]

    config_data = json.loads(config_path.read_text(encoding='utf-8')) # Config entries
    
    metadata = {}

    Acq_date = telemetry_data['ACQ_DATE']
    dt = datetime.strptime(Acq_date, "%a %b %d %H:%M:%S %Y")
    metadata['DATE-OB'] = (dt.strftime("%Y-%m-%dT%H:%M:%S.000"), 'UTC time of observation')

    metadata['OBJECT'] = (object, 'Observed object')

    match channel:
            case 'VIS': taskFile = config_data['visTaskFile']
            case 'NIR1': taskFile = config_data['nir1TaskFile']
            case 'NIR2': taskFile = config_data['nir2TaskFile']
            case 'SWIR': taskFile = config_data['swirTaskFile']
        
    task_values = [taskFile[i:i + 8] for i in range(0, len(taskFile), 8)]
    #Extract exposure times
    exposure_times = [task_values[i][4] for i in range(len(task_values))]
    if all(x == exposure_times[0] for x in exposure_times):
        exposure_times = [exposure_times[0]]
    exposures_str = ','.join(str(x) for x in exposure_times)
    metadata['EXPOSURE'] = (exposures_str, "Exposuretime(s) [DNs]")

    det_temp = channel_specific_telemetry['DET_TEMP']
    metadata['CCDTEMP'] = (det_temp, f'Detector temp [DNs]')

    metadata['AMBTEMP'] = ('', 'Ambient temperature')

    metadata['SC_CLK'] = ('', 'SC clock Hera instrument format')

    fault = channel_specific_telemetry['FAULT']
    metadata['ERRORFLG'] = (fault, 'Error flags for instrument')

    return metadata
    
def collect_instrument_specific_metadata(
        telemetry_path: Path, 
        config_path: Path, 
        channel: str,
    ) -> Dict[str, Tuple[str, str]]:
    """
    Collects instrument specific metadata.

    Parameters:
        telemetry_path (Path): Path object to the telemetry file
        config_path (Path): Path object to the config file
        channel (str): Channel information to look for
    
    Returns:
        Dict[header_keyword, Tuple(value, comment)]
    """

    metadata = {}

    #### Impelement here the ASPECT specific metadata

    return metadata

def collect_spice_metadata(
        telemetry_path: Path, 
        mk: str | Path,  
        target: str = 'DIDYMOS', 
        test: bool =True
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

    telemetry_data = json.loads(telemetry_path.read_text(encoding='utf-8')) # Telemetry etries

    utc_ob = telemetry_data['ACQ_DATE']
    if test:
        utc_ob = '2025-06-15T05:40:46.6666' # Testing

    spice_metadata = {}

    hera_spice.load_meta_kernel(mk) # Load the meta kernel

    et = hera_spice.utc_2_et(utc_ob)
    milani_frame = 'MILANI_SPACECRAFT'
    camera_frame = 'MILANI_NAVCAM'

    mk_id = hera_spice.query_mk_identifier() # Meta kernel version
    spice_metadata['SPICE_MK'] = (mk_id, 'SPICE meta kernel version')

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
    quaternions = hera_spice.query_spacecraft_quaternions(frame_name=milani_frame, et=et, tol=1, ref='J2000' )
    spice_metadata['SC_QUAT0'] = (quaternions[0], 'Spacecraft quaternion 0')
    spice_metadata['SC_QUAT1'] = (quaternions[1], 'Spacecraft quaternion 1')
    spice_metadata['SC_QUAT2'] = (quaternions[2], 'Spacecraft quaternion 2')
    spice_metadata['SC_QUAT3'] = (quaternions[3], 'Spacecraft quaternion 3')

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

def collect_calibration_metadata() -> Dict[str, Tuple[str, str]]:
    """
    Collects instrument specific metadata.

    Parameters:
    
    Returns:
        Dict[header_keyword, Tuple(value, comment)]
    """

    metadata = {}

    #### Impelement here the ASPECT calibration

    return metadata

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
        subprocess.run([str(decompress_path)], stdin=f_in, stdout=f_out, check=True)
    
    return output_path

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
        case 'VIS' | 'NIR1':
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
            return ''

# Read metadata from config file
def collect_image_metadata(config_path: Path, channel: str) -> Dict[str, Tuple[str, str]]:
    """
    Collect image specific metadata

    Parameters: 
        config_path (Path): Path object to the config file
        channel (str): Instrument channel

    Returns:
        Dict[header_keyword, Tuple(value, comment)]
    """
    
    meta_data = {}

    config_data = json.loads(config_path.read_text(encoding='utf-8')) # Config entries


    #read SP values for each image
    match channel:
        case 'VIS':
            taskFile = config_data['visTaskFile']
        case 'NIR1':
            taskFile = config_data['nir1TaskFile']
        case 'NIR2':
            taskFile = config_data['nir2TaskFile']
        case 'SWIR':
            taskFile = config_data['swirTaskFile']
    

    #Extract sp values from taskValues
    taskValues = [taskFile[i:i + 8] for i in range(0, len(taskFile), 8)]
    sp1Values = [taskValues[i][1] for i in range(0, len(taskValues))]
    sp2Values = [taskValues[i][2] for i in range(0, len(taskValues))]
    sp3Values = [taskValues[i][3] for i in range(0, len(taskValues))]
    #Extract exposure times
    exposureTimes = [taskValues[i][4] for i in range(0, len(taskValues))]
    exposureTimes = exposureTimes[0] if all(et == exposureTimes[0] for et in exposureTimes) else exposureTimes
    sp1 = ','.join(str(x) for x in sp1Values)
    sp2 = ','.join(str(x) for x in sp2Values)
    sp3 = ','.join(str(x) for x in sp3Values)
    #Check the order based on SP1 index 3
    order = check_order(sp1Values[3], channel)

    meta_data['CHANNEL'] = (channel, 'Instrument channel')
    meta_data['ORDER'] = (order, 'LOW / HIGH')
    meta_data['EXPOSURE'] = (exposureTimes, 'Exposuretime(s) [DNs].')
    meta_data['SP1'] = (sp1, 'Setpoint 1')
    meta_data['SP2'] = (sp2, 'Setpoint 2')
    meta_data['SP3'] = (sp3, 'Setpoint 3')

    return meta_data

def det_temp_conversion(value: float, channel: str) -> Tuple[float, float]:
    """
    Converts the 'DET_TEMP' entries from telemetry to Celcius and Kelvin

    Parameters:
        value (float): Detector temperature DN value
        channel (str): Instrument channel

    Returns:
        Tuple(Celsius, Kelvin)
    """
    match channel:
        case 'VIS':
            c = value * 0.6522 - 295.87
            return (c, c + kelvin)
        case 'NIR1': return (value, value)
        case 'NIR2': return (value, value)
        case 'SWIR':
            c = (-6e-11) * value**3 + 3e-6 * value**2 - 0.0188 * value + 17.291
            return (c, c + kelvin)

def exposure_conversion(value: float, channel: str) -> float:
    """
    Converts exposure DN into seconds.

    Parameters:
        value (float): exposure DN
        channel (str): channel
    
    Returns: 
        float exposure in seconds
    """
    match channel:
        case 'VIS':  return ((value + 8.6123) / 155.04) / 1000 # Conversion from DN to s
        case 'NIR1': return value / 100000
        case 'NIR2': return value / 100000
        case 'SWIR': return value 

def wavelength_conversion(channel: str, order: str, sp_values: List[float]) -> str:
    """
    Calculates the wavelengths from setpoint values

    Parameters:
        channel: instrument channel
        order: order of the acquisition (high / low)
        sp_values: Piezo actuator setpoint values

    Returns:
        str: wavelengths in string separated by comma
    """
    wavelengths = []
    
    match (channel, order):
        # The correct values for the corretion needed
        case 'VIS', 'HIGH':
            for i in range(0, len(sp_values)):
                wavelength = round(0.0749 * sp_values[i] - 786.9)
                wavelengths.append(wavelength)
        case 'VIS', 'LOW':
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
    
    return ",".join(map(str, wavelengths))


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
    
    combined_header = headers[0].copy()
    combined_header['ORIGFILE'] = ('dc_X_exp_XXX.bin', 'Original file name.')
    combined_header['PROCLEVL'] = ('2B', 'Calibration level')
    exposure = ''
    det_temp = ''
    det_temp_c = ''
    for i, hdr in enumerate(headers):
        exp = hdr.get('EXPOSURE')
        exposure += f'{exp}, '
        dt = hdr.get('CCDTEMP')
        det_temp += f'{dt}, '
        try:
            numeric = float(dt)
            det_temp_c += f'{numeric - kelvin}, '
        except (ValueError, TypeError):
            det_temp_c += f'UNK, '
    
    exposure = exposure.rstrip(',')
    det_temp = det_temp.rstrip(',')
    det_temp_c = det_temp_c.rstrip(',')
    combined_header['EXPOSURE'] = (exposure, 'Exposure times [s]')
    combined_header['CCDTEMP'] = (det_temp, f'Detector temp [K] ({det_temp_c} [C])')

    return combined_header

def combine_image_headers(headers: List[Header]) -> Header:
    if not headers: 
        raise ValueError('No headers provided for combination')
    
    combined_header = headers[0].copy()
    del combined_header['ORDER']
    del combined_header['WAVELEN']
    del combined_header['SP1']
    del combined_header['SP2']
    del combined_header['SP3']
    exposure = ''
    channels = ''
    for i, hdr in enumerate(headers):
        exp = hdr.get('EXPOSURE')
        exposure += f'{exp}, '
        channel = hdr.get('CHANNEL')
        channels += f'{channel}, '
        order = hdr.get('ORDER')
        combined_header[f'{channel}_O'] = (order, f'LOW / HIGH')
        wl = hdr.get('WAVELEN')
        combined_header[f'{channel}_WL'] = (wl, f'{channel} wavelengths')
        sp1 = hdr.get('SP1')
        combined_header[f'{channel}_SP1'] = (sp1, f'{channel} Setpoint 1')
        sp2 = hdr.get('SP2')
        combined_header[f'{channel}_SP2'] = (sp2, f'{channel} Setpoint 2')
        sp3 = hdr.get('SP3')
        combined_header[f'{channel}_SP3'] = (sp3, f'{channel} Setpoint 3')

    exposure = exposure.rstrip(',')
    channels = channels.rstrip(',')
    combined_header['EXPOSURE'] = (exposure, 'Exposure times [s]')
    combined_header['CHANNELS'] = (channels, 'Instrument channels')

    
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

def extract_diagnostics(image: np.ndarray) -> Tuple[np.ndarray, List[List[int]]]:
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
    orb = cv2.ORB_create(nfeatures=2000) # create ORB feature detector
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


print()
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

    channel_map = {
        'VIS' : '0',
        'NIR1': '1',
        'NIR2': '2',
        'SWIR': '3'
    }

    for file in directory.iterdir():
        if file.is_file():
            match = re.search(r'-(VIS|NIR1|NIR2|SWIR)-', file.name)
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