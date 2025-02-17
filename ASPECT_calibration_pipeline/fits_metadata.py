from astropy.io import fits
import numpy as np
import json
from datetime import datetime
import hera_spice as spice

# Work in progress

"""
This is a file for retrieving and writing missing fits metadata.
Remember to add spice_metakernel_path to this file and hera_spice.py if retrieving SPICE info.
"""

telemetry_path = "test_data/[July_test_package]2024-07-25_15-41-18_nir2_h_nir2_ho_600w_7500/meta/telemetry.json"
config_path = "test_data/[July_test_package]2024-07-25_15-41-18_nir2_h_nir2_ho_600w_7500/meta/config.json"
spice_metakernel_path = "" # Add metakernel path, for example ...SPICE/HERA/kernels/mk/hera_ops.tm
fits_path = "test_data/D1D2_simulated_cube.fits"
output_path = "test_data/test_outputs/D1D2_simulated_cube_updated.fits"

test_main = True
test_dynamic_metadata_retrieval = False

# Manually add metadata
"""
This dictionary is serving as a temporary placeholder for some of the
metadata fields until an automatic retrieval is implemented for them.
These fields are copied from AFC and HSH FITS files, and are examples
for ASPECT FITS files. All data fields are therefore not mandatory,
and their inclusion should be decided based on relevancy to ASPECT data.
at the moment all of these fields are wrote into the primary HDU,
whereas some of them should possibly be in the image HDUs.

Feel free to implement automatic retrieval for any of these fields, or
delete irrelevant fields. There are some examples or data retrieval
functions below (e.g. get_acq_date which is called in retrieve_dynamic_metadata).

Structure: {keyword: (value, comment)}
"""
static_metadata = {
    'INSTRUME': ('ASPECT', ''),
    'ORIGIN': ('ESA - HERA', 'HERA imaging instruments'),
    'MISSPHAS': ('', 'HERA Mission Phase ID'), # For example 'COMMISSIONING'
    'OSERV_ID': ('', 'HERA Observation ID'), # For example '0001_EARTH_MOON'
    'FILENAME': ('', ''), # What are the correct naming conventions? AFC example: AF1_009FEQ_241010T212506_1A.fits (ORIGFILE: af1_009FEQ_241010T212506_0.FITS)
    'SWCREATE': ('', 'Software used'), # For example 'HERACAL' / BME-MOGI HeraCal (c) 2024 G. Kovacs
    'DATE': ('', 'File creation time UTC'), # Is this original creation time or last modification time?
    'PROCLEVL': ('', ''), # For example '1A'
    'SC_CLK': ('', 'Spacecraft clock start'), # For example '310746:369199'. SPICE spacecraft clock kernel (SCLK) contains fictional data (updated 2025-02-10).
    'OBJECT': ('', 'Observation target ID'), # For example 'UNK'. Is this provided by telemetry or SPICE or manual input?
    'EXPOSURE': ('', 'Exposure command [Sec]'), # For example 0.000416. Is this provided in config file ("exposurePrirotiy": [list])?
    'CCDTEMP': ('', 'Detector temp [K]'),
    'SPICEVER': ('', 'SPICE version'), # For example '2024-10-10T21:25:06.789'
    'SPICE_MK': ('', 'SPICE metakernel'), # For example '2024-10-10T21:25:06.789'
    'HIERARCH SPICE_SC_CLK_START_SEC': ('', 'Spacecraft clock seconds'), # SPICE spacecraft clock kernel (SCLK) contains fictional data (updated 2025-02-10).
    'HIERARCH SPICE_SC_CLK_START_FRACT': ('', 'Spacecraft clock fraction'), # SPICE spacecraft clock kernel (SCLK) contains fictional data (updated 2025-02-10).
    'CAM_RA': ('', 'Camera pointing'),
    'CAM_DEC': ('', 'Camera pointing'),
    'CAM_NAZ': ('', 'Camera pointing'),
    'CELN_CLK': ('', ''),
    'SOL_ELNG': ('', 'Solar elongation'),
    'PHAS_ANG': ('', ''),
    'TRG_POSX': ('', 'Target position vector X'),
    'TRG_POSY': ('', 'Target position vector Y'),
    'TRG_POSZ': ('', 'Target position vector Z'),
    'HIERARCH AFC_IMAGE_TYPE': ('', ''), # For example 'NAV' (Navigation image)
    'HIERARCH AFC_UNITID': ('', 'Tm afc_unitId'),
    'HIERARCH AFC_BIN_EN': ('', 'Tm afc_BIN_EN'),
    'HIERARCH AFC_BIN_FAC': ('', 'Tm afc_BIN_FAC'),
    'HIERARCH AFC_TEMP_RAW': ('', 'Tm afc_temp_raw'),
    'HIERARCH AFC_FULL_EXPOSURE': ('', 'Tm afc_Rdclfullexp us'),
    'HIERARCH AFC_PIXEL_CLK': ('', 'Tm afc_Rdfrpixclock'),
    'HIERARCH AFC_REFRLINES': ('', 'Tm afc_Rdfrlines'),
    'HIERARCH TEMP_AFC_A_TEMP': ('', 'AFC_A sensor temp [K]'), # Temps in telemetry?
    'Other relevant metadata': ('', '')
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

def get_spacecraft_position_vectors(
        spice_metakernel_path: str,
        target: str = 'HERA',
        utc_time: str = '2025-12-27T00:00:00',                            
        frame: str = 'J2000',
        observer: str = 'EARTH'
    ):
    """
    Get the spacecraft position vectors based on provided attributes.
    
    Parameters:
    spice_metakernel_path (str): Path to the SPICE metakernel file.
    target (str): The spacecraft or celestial body of interest.
    utc_time (str): The UTC time for which the position vectors are required.
    frame (str): The reference frame for the position vectors.
    observer (str): The observing body (e.g., EARTH).
    
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
        
        return {
            "SC_POSX": (position[0], "Spacecraft position vector X"),
            "SC_POSY": (position[1], "Spacecraft position vector Y"),
            "SC_POSZ": (position[2], "Spacecraft position vector Z"),
        }
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An error occurred while retrieving spacecraft position vectors: {e}")
    
    return None

def get_spacecraft_solar_distance(
        spice_metakernel_path: str,
        target: str = 'HERA',
        utc_time: str = '2025-12-27T00:00:00',                            
        frame: str = 'J2000',
        observer: str = 'SUN'
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
            "SCSOLDST": (distance, "Spacecraft Solar distance"),
        }
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An error occurred while retrieving spacecraft-solar distance: {e}")

def get_spacecraft_quaternions(
        spice_metakernel_path: str,
        utc_time: str = '2025-12-27T00:00:00',
        inertial_frame: str = 'J2000',
        spacecraft_frame: str = 'HERA_DIDYMOS_NPO'
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
        quaternions = spice.query_spacecraft_quaternions(
            spice_metakernel_path,
            utc_time,
            inertial_frame,
            spacecraft_frame
        )
        
        return {
            "SC_QUATW": (quaternions[0], "Spacecraft quaternions"),
            "SC_QUATX": (quaternions[1], "Spacecraft quaternions"),
            "SC_QUATY": (quaternions[2], "Spacecraft quaternions"),
            "SC_QUATZ": (quaternions[3], "Spacecraft quaternions"),
        }
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An error occurred while retrieving spacecraft quaternions: {e}")

# Work in progress
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
    date_obs = get_acq_date(telemetry_path)
    origfile = get_origfile(fits_path)
    sc_pos = get_spacecraft_position_vectors(spice_metakernel_path, utc_time=date_obs["DATE-OBS"][0])
    scsoldst = get_spacecraft_solar_distance(spice_metakernel_path, utc_time=date_obs["DATE-OBS"][0])
    quaternions = get_spacecraft_quaternions(spice_metakernel_path, utc_time=date_obs["DATE-OBS"][0])
    # Continue with the rest of the dynamic metadata retrieval...
    
    dynamic_metadata.update(date_obs)
    dynamic_metadata.update(origfile)
    dynamic_metadata.update(sc_pos)
    dynamic_metadata.update(scsoldst)
    dynamic_metadata.update(quaternions)

    if test:
        print(dynamic_metadata)
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
    dynamic_metadata = retrieve_dynamic_metadata(telemetry_path, config_path, spice_metakernel_path)
    
    # Update the FITS file with the combined metadata
    combined_metadata = {**static_metadata, **dynamic_metadata}
    update_fits_metadata(fits_path, combined_metadata, output_path)

if test_main:
    print_fits_metadata_with_summary(fits_path)
    main(fits_path, telemetry_path, config_path, spice_metakernel_path, output_path, static_metadata)
    print_fits_metadata_with_summary(output_path)

if test_dynamic_metadata_retrieval:
    retrieve_dynamic_metadata(telemetry_path, config_path, spice_metakernel_path, test=True)