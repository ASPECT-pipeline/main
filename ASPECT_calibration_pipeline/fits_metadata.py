from astropy.io import fits
import numpy as np
import json
from datetime import datetime
import hera_spice as spice
import os

# Work in progress

"""
This is a file for retrieving and writing missing fits metadata.
You need to have HERA SPICE kernel dataset installed to use functions that query SPICE data.
HERA SPICE kernel dataset: https://s2e2.cosmos.esa.int/bitbucket/projects/SPICE_KERNELS/repos/hera/browse
Remember to add spice_metakernel_path below if retrieving SPICE info.
"main" funtion is the main function to be called, and it calls other
functions to retrieve metadata and creates an updated fits.
"""

telemetry_path = "test_data/[July_test_package]2024-07-25_15-41-18_nir2_h_nir2_ho_600w_7500/meta/telemetry.json"
config_path = "test_data/[July_test_package]2024-07-25_15-41-18_nir2_h_nir2_ho_600w_7500/meta/config.json" # Not used at the moment
# spice_metakernel_path = "/home/sysa/HERA/SPICE/hera_v180/hera/kernels/mk/hera_plan.tm" # Add metakernel path, for example /home/sysa/HERA/SPICE/HERA/kernels/mk/hera_plan.tm
spice_metakernel_path = "/Users/valtterimj/Downloads/Työ/Aalto/Hera/hera_spice/kernels/mk/hera_plan.tm" 
fits_path = "/home/sysa/HERA/github/main/test_data/levels_012_test/test_output/test_1/simulated_full_datacube.fits"
output_path = "/home/sysa/HERA/github/main/test_data/test_outputs/test.fits"

image_target = "Didymos" # Didymos or Dimorphos

test_main = False
test_dynamic_metadata_retrieval = False

# Manually add metadata
"""
This dictionary is serving as a temporary placeholder for some of the
metadata fields until an automatic retrieval from telemetry.json,
config.json, or elsewhere is implemented for them.
These variables are copied from AFC and HSH FITS files, and are examples
for ASPECT FITS files. All data fields are therefore not mandatory,
and their inclusion should be decided based on relevancy to ASPECT.
at the moment all of these fields are written into the primary HDU,
whereas some of them should possibly be in the image HDUs.

Feel free to implement automatic retrieval for any of these fields, or
delete irrelevant fields. There are some examples or data retrieval
functions below (e.g. get_acq_date which is called in retrieve_dynamic_metadata).

Structure: {keyword: (value, comment)}
"""
static_metadata = {
    'INSTRUME': ('ASPECT', ''),
    'ORIGIN': ('ESA - HERA/Milani', 'HERA/Milani imaging instruments'),
    # 'MISSPHAS': ('', 'HERA Mission Phase ID'), # For example 'COMMISSIONING'
    # 'OSERV_ID': ('', 'HERA Observation ID'), # For example '0001_EARTH_MOON'
    # 'FILENAME': ('', ''), # What are the correct naming conventions? AFC example: AF1_009FEQ_241010T212506_1A.fits (ORIGFILE: af1_009FEQ_241010T212506_0.FITS)
    # 'SWCREATE': ('', 'Software used'), # For example 'HERACAL' / BME-MOGI HeraCal (c) 2024 G. Kovacs
    # 'DATE': ('', 'File creation time UTC'), # Is this original creation time or last modification time?
    # 'PROCLEVL': ('', ''), # For example '1A'
    # 'OBJECT': ('', 'Observation target ID'), # For example 'UNK'. Is this provided by telemetry or SPICE or manual input?
    # 'EXPOSURE': ('', 'Exposure commanded [Sec]'), # For example 0.000416. Is this provided in config file ("exposurePrirotiy": [list])?
    # 'CCDTEMP': ('', 'Detector temp [K]'), # Temps in telemetry.json?
    # 'HIERARCH SPICE_SC_CLK_START_SEC': ('', 'Spacecraft clock seconds'), # SPICE spacecraft clock kernel (SCLK) contains fictional data (updated 2025-02-10).
    # 'HIERARCH SPICE_SC_CLK_START_FRACT': ('', 'Spacecraft clock fraction'), # SPICE spacecraft clock kernel (SCLK) contains fictional data (updated 2025-02-10).
    # 'CAM_RA': ('', 'Camera pointing'), # Is this SPICE data?
    # 'CAM_DEC': ('', 'Camera pointing'),
    # 'CAM_NAZ': ('', 'Camera pointing'),
    # 'CELN_CLK': ('', ''),
    # 'HIERARCH AFC_IMAGE_TYPE': ('', ''), # For example 'NAV' (Navigation image)
    # 'HIERARCH AFC_UNITID': ('', 'Tm afc_unitId'),
    # 'HIERARCH AFC_BIN_EN': ('', 'Tm afc_BIN_EN'),
    # 'HIERARCH AFC_BIN_FAC': ('', 'Tm afc_BIN_FAC'),
    # 'HIERARCH AFC_TEMP_RAW': ('', 'Tm afc_temp_raw'),
    # 'HIERARCH AFC_FULL_EXPOSURE': ('', 'Tm afc_Rdclfullexp us'),
    # 'HIERARCH AFC_PIXEL_CLK': ('', 'Tm afc_Rdfrpixclock'),
    # 'HIERARCH AFC_REFRLINES': ('', 'Tm afc_Rdfrlines'),
    # 'HIERARCH TEMP_AFC_A_TEMP': ('', 'AFC_A sensor temp [K]'), # Temps in telemetry.json?
}

def read_json_var(input_path: str, var: str):
	"""
	Read a variable from a JSON file.
	
	Parameters:
	path (str): Path to the JSON file.
	var (str): Variable name to read.
	
	Returns:
	The value of the variable.
	"""
	try:
		with open(input_path, "r") as f:
			data = json.load(f)
			return data[var]
	except FileNotFoundError:
		print(f"Error: File not found at {input_path}")
	except KeyError:
		print(f"Error: Variable '{var}' not found in the JSON file.")
	except Exception as e:
		print(f"An error occurred: {e}")

def parse_acq_date(acq_date: str):
    """
    Convert ACQ_DATE from telemetry.json format to 'YYYY-MM-DDTHH:MM:SS.sss'.
    """
    try:
        dt = datetime.strptime(acq_date, "%a %b %d %H:%M:%S %Y")
        return {"DATE-OBS": (str(dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]), 'Observation time UTC')}  # Truncate microseconds to milliseconds
    except ValueError as e:
        print(f"Error parsing ACQ_DATE: {e}")
        return None
    
test_time = '2027-01-02T05:40:46'

def get_acq_date(telemetry_path: str):
    """
    Retrieve the acquisition date from the telemetry JSON file.
    
    Parameters:
    telemetry_path (str): Path to the telemetry JSON file.
    
    Returns:
    dict: The acquisition date
    """
    acq_date = read_json_var(telemetry_path, "ACQ_DATE")
    return parse_acq_date(acq_date)

def print_fits_metadata_with_summary(file_path: str):
    """
    Reads a FITS file, prints metadata, summarizes data tables, and displays a sample of the first table.

    Parameters:
        file_path (str): The path to the FITS file.
    """
    try:
        # Open the FITS file
        with fits.open(file_path) as hdul:
            print("FITS File Structure:\n")
            print(hdul.info())  # Print the structure of the FITS file
            print("\n====================================\n")

            # Iterate through each HDU (Header/Data Unit)
            for i, hdu in enumerate(hdul):
                print(f"HDU {i}: {type(hdu)}")

                # Print header metadata
                print("Header Metadata:")
                print(repr(hdu.header))

                # Check if there is associated data
                if hdu.data is not None:
                    print("\n------------------------------------\n")
                    print("Data Type:", type(hdu.data))
                    print("Data Shape:", hdu.data.shape)

                    # If the HDU contains tabular data, summarize and display a sample
                    if isinstance(hdu.data, np.ndarray):
                        # print("Data Summary:")
                        # print(f"Min: {np.min(hdu.data)}, Max: {np.max(hdu.data)}, Mean: {np.mean(hdu.data)}")

                        # Display a small sample if it's the first HDU with data
                        if i == 1:  # Assuming HDU 1 is the first with table data
                            print("Sample Data:")
                            sample_shape = tuple(min(5, dim) for dim in hdu.data.shape)  # Get up to 5 elements per dimension
                            sample = hdu.data[tuple(slice(0, s) for s in sample_shape)]
                            print(sample)
                else:
                    print("No data associated with this HDU.")

                print("\n------------------------------------\n")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# print_fits_metadata_with_summary('/home/sysa/HERA/github/main/fits_examples_from_other_instruments/sftp_provi-hera/AFC1/AF1_009FEQ_241010T212506_1A.fits')

def update_fits_metadata(input_path: str, metadata: dict, output_path: str = None):
    """
    Update a FITS file with user-given metadata and save it to a new location.
    
    Parameters:
    input_path (str): Path to the original FITS file.
    metadata (dict): Dictionary containing metadata keys and values to be added.
    output_path (str, optional): Path to save the new FITS file.
                                    If None, it overwrites the original file.
    """
    try:
        # Open the FITS file
        with fits.open(input_path) as hdul:
            # Get the primary header
            header = hdul[0].header
            
            # Update the header with user-provided metadata
            for key, value in metadata.items():
                header[key] = value
            
            if output_path:
                # Save to a new file
                hdul.writeto(output_path, overwrite=True)
                print(f"Updated FITS file saved to {output_path}")
            else:
                hdul.flush()  # Save changes in-place
    
    except Exception as e:
        print(f"Error updating FITS metadata: {e}")

def delete_fits_metadata(input_path: str, keyword: str, output_path: str = None):
    """
    Deletes a metadata keyword from the header of a FITS file.

    Parameters:
        input_path (str): Path to the input FITS file.
        keyword (str): The metadata keyword to be removed.
        output_path (str, optional): Path to save the modified FITS file. 
                                     If None, it overwrites the original file.

    Returns:
        None
    """
    # Open the FITS file in update mode
    with fits.open(input_path, mode='update') as hdul:
        header = hdul[0].header  # Access primary header
        
        # Remove the keyword if it exists
        if keyword in header:
            del header[keyword]
            print(f"Deleted keyword '{keyword}' from FITS file.")
        else:
            print(f"Keyword '{keyword}' not found in FITS file.")
        
        # Write changes to the file (if output_path is not specified, overwrite)
        if output_path:
            hdul.writeto(output_path, overwrite=True)
        else:
            hdul.flush()  # Save changes in-place

# for keyword in ['SC_QUAT1', 'SC_QUAT2', 'SC_QUAT3', 'CAM_DEG']:
#     delete_fits_metadata(
#         input_path='/home/sysa/HERA/github2/main/test_data/test_outputs/example-1-vis-nir1-nir2-hyperstacks_deleted_metadata.fits',
#         keyword=keyword,
#         output_path='/home/sysa/HERA/github2/main/test_data/test_outputs/example-1-vis-nir1-nir2-hyperstacks_deleted_metadata.fits'
#     )
# print_fits_metadata_with_summary('/home/sysa/HERA/github2/main/test_data/test_outputs/example-1-vis-nir1-nir2-hyperstacks_deleted_metadata.fits')

def get_file_name(file_path: str):
    """
    Extracts the file name from a file path.
    
    Parameters:
    file_path (str): The full path to the file.
    
    Returns:
    str: The file name.
    """
    return file_path.split("/")[-1]

def get_origfile(file_path: str):
    """
    Extracts the origfile variable for metadata dict.
    
    Parameters:
    file_path (str): The full path to the file.
    
    Returns:
    dict: ORIGFILE.
    """
    return {"ORIGFILE": (get_file_name(file_path), 'Original file name')}

def get_target_position_vectors(
        spice_metakernel_path: str,
        target: str = 'Milani', # Milani or HERA
        utc_time: str = test_time,                            
        frame: str = "DIMORPHOS_FIXED",  # DIDYMOS_FIXED or DIMORPHOS_FIXED, or J2000 for inertial
        observer: str = "Dimorphos" # Didymos or Dimorphos, or Didymos_barycenter
    ):
    """
    Get the target position vectors based on provided attributes.
    
    Parameters:
    spice_metakernel_path (str): Path to the SPICE metakernel file.
    target (str): The spacecraft or celestial body of interest.
    utc_time (str): The UTC time for which the position vectors are required.
    frame (str): The reference frame for the position vectors.
    observer (str): The observing body (e.g., SUN or EARTH).
    
    Returns:
    dict: A dictionary containing position vectors.
    """
    try:
        # Ensure the metakernel path is provided
        if not spice_metakernel_path:
            raise ValueError("SPICE metakernel path is required.")
        
        # Query spacecraft position vectors using HERA SPICE toolkit
        position = spice.query_spacecraft_position_vectors(
            spice_metakernel_path,
            target=target,
            utc_time=utc_time,
            frame=frame,
            observer=observer
        )
        
        if target in ['HERA', 'Milani']:
            return {
                "SC_POSX": (position[0], f"Spacecraft position vector X (observer: {observer})"),
                "SC_POSY": (position[1], f"Spacecraft position vector Y (observer: {observer})"),
                "SC_POSZ": (position[2], f"Spacecraft position vector Z (observer: {observer})")
            }
        else:
            return {
                'TRG_POSX': (position[0], f'Target position vector X [km]'),
                'TRG_POSY': (position[1], f'Target position vector Y [km]'),
                'TRG_POSZ': (position[2], f'Target position vector Z [km]')
            }
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An error occurred while retrieving spacecraft position vectors: {e}")
    
    return None

def get_spacecraft_solar_distance(
        spice_metakernel_path: str,
        target: str = 'SUN',
        utc_time: str = test_time,                            
        frame: str = 'J2000',
        observer: str = 'Milani'
    ):
    """
    Get the distance between the spacecraft and the Sun.

    Parameters:
    spice_metakernel_path (str): Path to the SPICE metakernel file.
    target (str): The spacecraft or celestial body of interest.
    utc_time (str): The UTC time for which the position vectors are required.
    frame (str): The reference frame for the position vectors.
    observer (str): The observing body (e.g., SUN).
    
    Returns:
    dict: A dictionary containing distance.
    """
    try:
        # Ensure the metakernel path is provided
        if not spice_metakernel_path:
            raise ValueError("SPICE metakernel path is required.")
        
        # Query spacecraft position vectors using HERA SPICE toolkit
        distance = spice.query_spacecraft_solar_distance(
            spice_metakernel_path,
            target=target,
            utc_time=utc_time,
            frame=frame,
            observer=observer
        )
        
        return {
            "SOLAR_D": (distance, "Solar distance [AU]"),
        }
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An error occurred while retrieving spacecraft-solar distance: {e}")

def get_spacecraft_quaternions(
        spice_metakernel_path: str,
        utc_time: str = test_time,
        inertial_frame: str = "DIMORPHOS_FIXED",  # DIDYMOS_FIXED or DIMORPHOS_FIXED, or J2000 for inertial
        spacecraft_frame: str = 'MILANI_SPACECRAFT'
    ):
    """
    Get the spacecraft quaternions.

    Parameters:
    spice_metakernel_path (str): Path to the SPICE metakernel file.
    utc_time (str): The UTC time for which the position vectors are required.
    inertial_frame (str): The reference frame for the position vectors.
    spacecraft_frame (str): The spacecraft frame.

    Returns:
    dict: A dictionary containing quaternions.
    """
    try:
        # Ensure the metakernel path is provided
        if not spice_metakernel_path:
            raise ValueError("SPICE metakernel path is required.")
        
        # Query spacecraft position vectors using HERA SPICE toolkit
        x, y, z, quaternion = spice.query_attitude(
            spice_metakernel_path,
            utc_time,
            inertial_frame,
            spacecraft_frame
        )
        
        return {
            "SC_QUATW": (quaternion[3], f"Spacecraft quaternion (W)"),
            "SC_QUATX": (quaternion[0], f"Spacecraft quaternion (X)"),
            "SC_QUATY": (quaternion[1], f"Spacecraft quaternion (Y)"),
            "SC_QUATZ": (quaternion[2], f"Spacecraft quaternion (Z)"),
        }
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An error occurred while retrieving spacecraft quaternions: {e}")

def get_solar_elongation(
        spice_metakernel_path: str,
        utc_time: str = test_time,
		target: str = "Dimorphos", # Didymos or Dimorphos, or Didymos_barycenter
		observer: str = 'Milani'
	):
    """
    Get the solar elongation angle.

    Parameters:
    spice_metakernel_path (str): Path to the SPICE metakernel file.
    utc_time (str): The UTC time for which the position vectors are required.
    target (str): The target body.
    observer (str): The observing body.

    Returns:
    dict: A dictionary containing solar elongation angle.
    """
    try:
        # Ensure the metakernel path is provided
        if not spice_metakernel_path:
            raise ValueError("SPICE metakernel path is required.")
        
        # Query spacecraft position vectors using HERA SPICE toolkit
        solar_elongation = spice.query_solar_elongation(
            spice_metakernel_path,
            utc_time,
            target,
            observer
        )
        
        return {
            "SOL_ELNG": (solar_elongation, f"Solar elongation [deg]"),
        }
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An error occurred while retrieving solar elongation angle: {e}")

def get_phase_angle(solar_elongation: dict):
    """
    Derive the phase angle from the solar elongation angle.
    (Phase angle = 180 [Deg] - solar elongation [Deg])

    Parameters:
    solar_elongation (dict): A dictionary containing the solar elongation angle.

    Returns:
    dict: A dictionary containing the phase angle.
    """
    try:
        # Ensure the solar elongation angle is provided
        if "SOL_ELNG" not in solar_elongation:
            raise ValueError("Solar elongation angle (SOL_ELNG) is required.")
        
        # Calculate the phase angle
        phase_angle = 180 - solar_elongation["SOL_ELNG"][0]
        
        return {
            "PHAS_ANG": (phase_angle, "Phase angle [Deg]"),
        }
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An error occurred while calculating the phase angle: {e}")

def get_spacecraft_clock_start(
        spice_metakernel_path: str,
        pool_variable: str = "SCLK_PARTITION_START_9102"
    ):
    """
    Disclaimer: SCLK contains fictional data (updated 2025-02-10)

    Get the spacecraft clock start time.
    
    Parameters:
    spice_metakernel_path (str): Path to the SPICE metakernel file.
    pool_variable (str): The pool variable containing the spacecraft clock start time.

    Returns:
    dict: A dictionary containing the spacecraft clock start time.
    """
    try:
        # Ensure the metakernel path is provided
        if not spice_metakernel_path:
            raise ValueError("SPICE metakernel path is required.")
        
        # Query spacecraft clock start time using HERA SPICE toolkit
        spacecraft_clock_start = spice.query_spacecraft_clock_start(
            spice_metakernel_path,
            pool_variable
        )
        
        return {
            "SC_CLK": (spacecraft_clock_start, "Spacecraft clock start (fictional data)"),
        }
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An error occurred while retrieving spacecraft clock start time: {e}")

def get_spice_version(spice_metakernel_path: str):
    """
    Get the SPICE version from the metakernel file.

    Parameters:
    spice_metakernel_path (str): Path to the SPICE metakernel file.

    Returns:
    dict: A dictionary containing the SPICE version.
    """
    try:
        # Ensure the metakernel path is provided
        if not spice_metakernel_path:
            raise ValueError("SPICE metakernel path is required.")
        
        # Query SPICE version using HERA SPICE toolkit
        spice_version = spice.spice_dataset_version(spice_metakernel_path)
        
        return {
            "SPICEVER": (spice_version, "SPICE dataset version"),
        }
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An error occurred while retrieving the SPICE version: {e}")

def get_filename(file_path: str) -> str:
    """Returns the file name from a given file path."""
    return os.path.basename(file_path)

def get_spice_metakernel(spice_metakernel_path: str):
    """Returns the SPICE metakernel name and SPICE version."""
    try:
        # Ensure the metakernel path is provided
        if not spice_metakernel_path:
            raise ValueError("SPICE metakernel path is required.")
        
        # Query SPICE version using HERA SPICE toolkit
        filename = get_filename(spice_metakernel_path)
        # filename = filename.replace(".tm", "_")
        # spice_version = spice.spice_dataset_version(spice_metakernel_path)
        # filename = filename + spice_version
        
        return {"SPICE_MK": (filename, "SPICE metakernel")}
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An error occurred while retrieving the SPICE version: {e}")

def get_sun_position(
        spice_metakernel_path: str,
        utc_time: str = test_time,
        frame: str = 'J2000', # Is this the correct frame?
        observer: str = 'Milani'
    ):
    """
    Get the Sun's position vector.

    Parameters:
    spice_metakernel_path (str): Path to the SPICE metakernel file.
    utc_time (str): The UTC time for which the position vectors are required.
    frame (str): The reference frame for the position vectors.
    observer (str): The observing body.

    Returns:
    dict: A dictionary containing the Sun's position vector.
    """
    try:
        # Ensure the metakernel path is provided
        if not spice_metakernel_path:
            raise ValueError("SPICE metakernel path is required.")
        
        # Query Sun's position vector using HERA SPICE toolkit
        position = spice.query_sun_position_vectors(
            spice_metakernel_path,
            utc_time=utc_time,
            frame=frame,
            observer=observer
        )
        
        return {
            "SUN_POSX": (position[0], f"Sun position vector X [km]"),
            "SUN_POSY": (position[1], f"Sun position vector Y [km]"),
            "SUN_POSZ": (position[2], f"Sun position vector Z [km]")
        }
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An error occurred while retrieving the Sun's position vector: {e}")

def get_earth_position(
        spice_metakernel_path: str,
        utc_time: str = test_time,
        frame: str = 'J2000', # Is this the correct frame?
        observer: str = 'Milani'
    ):
    """
    Get the Earth's position vector.

    Parameters:
    spice_metakernel_path (str): Path to the SPICE metakernel file.
    utc_time (str): The UTC time for which the position vectors are required.
    frame (str): The reference frame for the position vectors.
    observer (str): The observing body.

    Returns:
    dict: A dictionary containing the Sun's position vector.
    """
    try:
        # Ensure the metakernel path is provided
        if not spice_metakernel_path:
            raise ValueError("SPICE metakernel path is required.")
        
        # Query Earth's position vector using HERA SPICE toolkit
        position = spice.query_earth_position_vectors(
            spice_metakernel_path,
            utc_time=utc_time,
            frame=frame,
            observer=observer
        )
        
        return {
            "EARTPOSX": (position[0], f"Earth position vector X [km]"),
            "EARTPOSY": (position[1], f"Earth position vector Y [km]"),
            "EARTPOSZ": (position[2], f"Earth position vector Z [km]")
        }
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An error occurred while retrieving the Earth's position vector: {e}")

def get_earth_distance(
        spice_metakernel_path: str,
        target: str = 'EARTH',
        utc_time: str = test_time,
        frame: str = 'J2000', # Is this the correct frame?
        observer: str = 'Milani'
    ):
    """
    Get the distance between the spacecraft and the Earth.

    Parameters:
    spice_metakernel_path (str): Path to the SPICE metakernel file.
    target (str): The spacecraft or celestial body of interest.
    utc_time (str): The UTC time for which the position vectors are required.
    frame (str): The reference frame for the position vectors.
    observer (str): The observing body.
    
    Returns:
    dict: A dictionary containing distance.
    """
    try:
        # Ensure the metakernel path is provided
        if not spice_metakernel_path:
            raise ValueError("SPICE metakernel path is required.")
        
        # Query spacecraft position vectors using HERA SPICE toolkit
        distance = spice.query_spacecraft_earth_distance(
            spice_metakernel_path,
            target=target,
            utc_time=utc_time,
            frame=frame,
            observer=observer
        )
        
        return {
            "EARTH_D": (distance, "Earth distance [AU]"),
        }
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An error occurred while retrieving spacecraft-Earth distance: {e}")

def get_target_distance(
        spice_metakernel_path: str,
        target: str = image_target,
        utc_time: str = test_time,
        frame: str = 'DIMORPHOS_FIXED',
        observer: str = 'Milani'
    ):
    """
    Get the distance between the spacecraft and the target body.
    
    Parameters:
    spice_metakernel_path (str): Path to the SPICE metakernel file.
    target (str): The target body.
    utc_time (str): The UTC time for which the position vectors are required.
    frame (str): The reference frame for the position vectors.
    observer (str): The observing body.

    Returns:
    dict: A dictionary containing distance.
    """
    try:
        # Ensure the metakernel path is provided
        if not spice_metakernel_path:
            raise ValueError("SPICE metakernel path is required.")
        
        # Query spacecraft position vectors using HERA SPICE toolkit
        distance = spice.query_target_distance(
            spice_metakernel_path,
            target=target,
            utc_time=utc_time,
            frame=frame,
            observer=observer
        )
        
        return {
            "TRG_DIST": (distance, "Target distance [AU]"),
        }
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An error occurred while retrieving spacecraft-target distance: {e}")

def get_right_ascension(
        spice_metakernel_path: str,
        target: str = "Milani",
        utc_time: str = test_time,
        frame: str = 'J2000',
        observer: str = 'EARTH'
    ):
    
    try:
        # Ensure the metakernel path is provided
        if not spice_metakernel_path:
            raise ValueError("SPICE metakernel path is required.")
        
        # Query spacecraft position vectors using HERA SPICE toolkit
        right_ascension = spice.query_right_ascension(
            spice_metakernel_path,
            target=target,
            utc_time=utc_time,
            frame=frame,
            observer=observer
        )
        
        return {
            "CAM_RA": (right_ascension, "Camera axis RA [deg]"),
        }
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An error occurred while retrieving spacecraft-target distance: {e}")

def get_declination(
        spice_metakernel_path: str,
        target: str = "Milani",
        utc_time: str = test_time,
        frame: str = 'J2000',
        observer: str = 'EARTH'
    ):
    
    try:
        # Ensure the metakernel path is provided
        if not spice_metakernel_path:
            raise ValueError("SPICE metakernel path is required.")
        
        # Query spacecraft position vectors using HERA SPICE toolkit
        declination = spice.query_declination(
            spice_metakernel_path,
            target=target,
            utc_time=utc_time,
            frame=frame,
            observer=observer
        )
        
        return {
            "CAM_DEC": (declination, "Camera axis DEC [deg]"),
        }
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An error occurred while retrieving spacecraft-target distance: {e}")

def retrieve_dynamic_metadata(telemetry_path: str, config_path: str, spice_metakernel_path: str, test: bool = False):
    """
    Retrieve dynamic metadata from telemetry, config, and SPICE.
    
    Parameters:
    telemetry_path (str): Path to the telemetry JSON file.
    config_path (str): Path to the configuration JSON file.
    spice_path (str): Path to the SPICE kernel directory.
    
    Returns:
    dict: A dictionary containing the dynamic metadata.
    """
    dynamic_metadata = {}

    if image_target == "Didymos":
        spacecraft_position_frame = "DIDYMOS_FIXED"
        quaternions_frame = "DIDYMOS_FIXED"
        target_position_frame = "DIDYMOS_FIXED"
    elif image_target == "Dimorphos":
        spacecraft_position_frame = "DIMORPHOS_FIXED"
        quaternions_frame = "DIMORPHOS_FIXED"
        target_position_frame = "DIMORPHOS_FIXED"
    else:
        raise ValueError("Invalid image target. Choose 'Didymos' or 'Dimorphos'.")

    date_obs = get_acq_date(telemetry_path)
    if test:
        date_obs = {'DATE-OBS': ('2027-03-24T00:00:00.0000', 'Observation time UTC')}
    original_file_name = get_origfile(fits_path)
    spacecraft_position = get_target_position_vectors(
        spice_metakernel_path,
        frame=spacecraft_position_frame,
        observer=image_target,
        target="Milani",
        utc_time=date_obs["DATE-OBS"][0]
    )
    spacecraft_solar_distance = get_spacecraft_solar_distance(
        spice_metakernel_path,
        utc_time=date_obs["DATE-OBS"][0]
    )
    quaternions = get_spacecraft_quaternions(
        spice_metakernel_path,
        inertial_frame=quaternions_frame,
        utc_time=date_obs["DATE-OBS"][0]
    )
    solar_elongation = get_solar_elongation(
        spice_metakernel_path,
        target=image_target,
        utc_time=date_obs["DATE-OBS"][0]
    )
    phase_angle = get_phase_angle(solar_elongation)
    target_position = get_target_position_vectors(
        spice_metakernel_path,
        target=image_target,
        frame=target_position_frame,
        observer='Milani',
        utc_time=date_obs["DATE-OBS"][0]
    )
    spacecraft_clock_start = get_spacecraft_clock_start(spice_metakernel_path)
    spice_metakernel = get_spice_metakernel(spice_metakernel_path)
    spice_version = get_spice_version(spice_metakernel_path)
    sun_position = get_sun_position(
        spice_metakernel_path,
        utc_time=date_obs["DATE-OBS"][0],
        frame='J2000',
        observer='Milani'
    )
    earth_position = get_earth_position(
        spice_metakernel_path,
        utc_time=date_obs["DATE-OBS"][0],
        frame='J2000',
        observer='Milani'
    )
    earth_distance = get_earth_distance(
        spice_metakernel_path,
        target='EARTH',
        utc_time=date_obs["DATE-OBS"][0],
        frame='J2000',
        observer='Milani'
    )
    target = {"TARGET": (image_target.upper(), "")}
    target_distance = get_target_distance(
        spice_metakernel_path,
        target=image_target,
        utc_time=date_obs["DATE-OBS"][0],
        frame=target_position_frame,
        observer='Milani'
    )
    dynamic_metadata.update(date_obs)
    dynamic_metadata.update(spacecraft_clock_start)
    dynamic_metadata.update(original_file_name)
    dynamic_metadata.update(spice_metakernel)
    dynamic_metadata.update(spice_version)
    dynamic_metadata.update(sun_position)
    dynamic_metadata.update(spacecraft_solar_distance)
    dynamic_metadata.update(solar_elongation)
    dynamic_metadata.update(phase_angle)
    dynamic_metadata.update(earth_position)
    dynamic_metadata.update(earth_distance)
    dynamic_metadata.update(target)
    dynamic_metadata.update(target_position)
    dynamic_metadata.update(target_distance)
    dynamic_metadata.update(spacecraft_position)
    dynamic_metadata.update(quaternions)

    if test:
        print(json.dumps(dynamic_metadata, indent=4))
    else:
        return dynamic_metadata

def retrieve_only_spice_dynamic_metadata(spice_metakernel_path: str, test: bool = False):
    """
    Retrieve dynamic metadata from SPICE.
    
    Parameters:
    telemetry_path (str): Path to the telemetry JSON file.
    config_path (str): Path to the configuration JSON file.
    spice_path (str): Path to the SPICE kernel directory.
    
    Returns:
    dict: A dictionary containing the dynamic metadata.
    """
    dynamic_metadata = {}

    if image_target == "Didymos":
        default_frame = "J2000"
    elif image_target == "Dimorphos":
        raise NotImplementedError()
    else:
        raise ValueError("Invalid image target. Choose 'Didymos' or 'Dimorphos'.")

    date_obs = {'DATE-OB': ('2027-03-23T06:00:00.0000', 'UTC time of observation')}
    # spacecraft_position = get_target_position_vectors(
    #     spice_metakernel_path,
    #     frame=spacecraft_position_frame,
    #     observer=image_target,
    #     target="Milani",
    #     utc_time=date_obs["DATE-OBS"][0]
    # )
    spacecraft_solar_distance = get_spacecraft_solar_distance(
        spice_metakernel_path,
        utc_time=date_obs["DATE-OB"][0]
    )
    quaternions = get_spacecraft_quaternions(
        spice_metakernel_path,
        utc_time=date_obs["DATE-OB"][0],
        inertial_frame=default_frame,
        spacecraft_frame='MILANI_SPACECRAFT'
    )
    solar_elongation = get_solar_elongation(
        spice_metakernel_path,
        target=image_target,
        utc_time=date_obs["DATE-OB"][0]
    )
    # phase_angle = get_phase_angle(solar_elongation)
    target_position = get_target_position_vectors(
        spice_metakernel_path,
        target=image_target,
        frame=default_frame,
        observer='Milani',
        utc_time=date_obs["DATE-OB"][0]
    )
    # spacecraft_clock_start = get_spacecraft_clock_start(spice_metakernel_path)
    spice_metakernel = get_spice_metakernel(spice_metakernel_path)
    spice_version = get_spice_version(spice_metakernel_path)
    sun_position = get_sun_position(
        spice_metakernel_path,
        utc_time=date_obs["DATE-OB"][0],
        frame='J2000',
        observer='Milani'
    )
    earth_position = get_earth_position(
        spice_metakernel_path,
        utc_time=date_obs["DATE-OB"][0],
        frame='J2000',
        observer='Milani'
    )
    earth_distance = get_earth_distance(
        spice_metakernel_path,
        target='EARTH',
        utc_time=date_obs["DATE-OB"][0],
        frame='J2000',
        observer='Milani'
    )
    target = {"TARGET": (image_target.upper(), "Observation target (SPICE)")}
    target_distance = get_target_distance(
        spice_metakernel_path,
        target=image_target,
        utc_time=date_obs["DATE-OB"][0],
        frame=default_frame,
        observer='Milani'
    )
    right_ascension = get_right_ascension(
        spice_metakernel_path,
        target='Milani',
        utc_time=date_obs["DATE-OB"][0],
        frame=default_frame,
        observer='EARTH'
    )
    declination = get_declination(
        spice_metakernel_path,
        target='Milani',
        utc_time=date_obs["DATE-OB"][0],
        frame=default_frame,
        observer='EARTH'
    )
    dynamic_metadata.update(date_obs)
    # dynamic_metadata.update(spacecraft_clock_start)
    # dynamic_metadata.update(original_file_name)
    dynamic_metadata.update(spice_metakernel)
    dynamic_metadata.update(spice_version)
    dynamic_metadata.update(sun_position)
    dynamic_metadata.update(spacecraft_solar_distance)
    dynamic_metadata.update(solar_elongation)
    # dynamic_metadata.update(phase_angle)
    dynamic_metadata.update(earth_position)
    dynamic_metadata.update(earth_distance)
    dynamic_metadata.update(target)
    dynamic_metadata.update(target_position)
    dynamic_metadata.update(target_distance)
    # dynamic_metadata.update(spacecraft_position)
    dynamic_metadata.update(quaternions)
    dynamic_metadata.update(right_ascension)
    dynamic_metadata.update(declination)

    if test:
        print(json.dumps(dynamic_metadata, indent=4))
    else:
        return dynamic_metadata

def main(fits_path: str, telemetry_path: str, config_path: str, spice_metakernel_path: str, output_path: str, static_metadata: dict):
    """
    Main function to update FITS metadata with static and dynamic metadata.
    
    Parameters:
    input_path (str): Path to the original FITS file.
    telemetry_path (str): Path to the telemetry JSON file.
    config_path (str): Path to the configuration JSON file.
    spice_metakernel_path (str): Path to the SPICE metakernel.
    output_path (str): Path to save the updated FITS file.
    """
    # dynamic_metadata = retrieve_dynamic_metadata(telemetry_path, config_path, spice_metakernel_path)
    dynamic_metadata = retrieve_only_spice_dynamic_metadata(spice_metakernel_path) # Test for simulated data
    
    # combined_metadata = {**static_metadata, **dynamic_metadata} # Optionally add or remove static metadata dictionary
    
    # update_fits_metadata(fits_path, combined_metadata, output_path)
    update_fits_metadata(fits_path, dynamic_metadata, output_path)

if test_main:
    print_fits_metadata_with_summary(fits_path)
    main(fits_path, telemetry_path, config_path, spice_metakernel_path, output_path, static_metadata)
    print_fits_metadata_with_summary(output_path)

if test_dynamic_metadata_retrieval:
    retrieve_dynamic_metadata(telemetry_path, config_path, spice_metakernel_path, test=True)


"""
Function to modify the mk/hera_plan.tm PATH_VALUES to point into the correct directory

"""
from pathlib import Path

def print_metakernel_with_line_numbers(metakernel_path: str):
    """
    Prints the contents of a .tm meta-kernel file with line numbers for easy inspection.
    """
    metakernel_path = Path(metakernel_path).expanduser().resolve()

    if not metakernel_path.exists():
        print(f"❌ File not found: {metakernel_path}")
        return

    print(f"\n📄 Meta-kernel: {metakernel_path}\n{'-' * 80}")

    with open(metakernel_path, 'r') as file:
        for idx, line in enumerate(file, start=1):
            print(f"{idx:4}: {line.rstrip()}")

    print('-' * 80 + "\n✅ Done.\n")

# print_metakernel_with_line_numbers(
#     "/Users/valtterimj/Downloads/Työ/Aalto/Hera/hera_spice/kernels/mk/hera_plan.tm"
# )

from pathlib import Path

def update_path_values_in_metakernel(metakernel_path: str, new_path_value: str):
    """
    Updates the PATH_VALUES entry in a SPICE .tm file to a new absolute path.

    Args:
        metakernel_path: Full path to the .tm file.
        new_path_value: Absolute path to use in PATH_VALUES.
    """
    metakernel_path = Path(metakernel_path).expanduser().resolve()
    new_path = Path(new_path_value).expanduser().resolve()

    if not metakernel_path.exists():
        print(f"❌ Meta-kernel file not found: {metakernel_path}")
        return

    backup_path = metakernel_path.with_suffix('.tm.bak_path')
    metakernel_path.rename(backup_path)

    updated_lines = []

    with open(backup_path, 'r') as f:
        for line in f:
            if 'PATH_VALUES' in line and '=' in line:
                indent = line[:line.index('P')]
                new_line = f"{indent}PATH_VALUES       = ( '{new_path}' )\n"
                updated_lines.append(new_line)
            else:
                updated_lines.append(line)

    with open(metakernel_path, 'w') as f:
        f.writelines(updated_lines)

    print(f"✅ PATH_VALUES updated to: {new_path}")
    print(f"🗂️  Original meta-kernel backed up as: {backup_path}")

# update_path_values_in_metakernel(
#     metakernel_path="/Users/valtterimj/Downloads/Työ/Aalto/Hera/hera_spice/kernels/mk/hera_plan.tm",
#     new_path_value="/Users/valtterimj/Downloads/Työ/Aalto/Hera/hera_spice/kernels"
# )