import spiceypy as spice
import numpy as np
from scipy.spatial.transform import Rotation as R
import modules.utilities as utilities
import os

"""
This is a Python file for using HERA SPICE kernels.
To use this file, you need to have HERA SPICE kernel dataset installed.
You may need to write the correct path to the kernel folder specified in metakernel.
The SPICE dataset has too large file sizes to be uploaded into github.
Refer to the readme files in SPICE dataset for more information about SPICE kernels.
"""

"""
HERA SPICE kernel dataset: https://s2e2.cosmos.esa.int/bitbucket/projects/SPICE_KERNELS/repos/hera/browse
SpiceyPy docs: https://spiceypy.readthedocs.io/en/stable/documentation.html#
WebGeocalc: http://spice.esac.esa.int/webgeocalc/#NewCalculation
"""

"""
Memo:
- hera_plan.tm metakernel provides Milani long term predicted trajectory
- hera_ops.tm is later updated with asteroid phase cubesat trajectories
"""

test_time = '2025-06-02T05:40:46.6666'

acq_path = os.path.join(os.getcwd(), 'test_data/ASPECT_fly_images/acqseq_100')
meta_folder = os.path.join(acq_path, 'meta')
telemetry_path = os.path.join(meta_folder, 'telemetry.json')

spice_mk_plan = "/Users/valtterimj/Downloads/Työ/Aalto/Hera/hera_spice/kernels/mk/hera_plan.tm"
spice_mk_ops = "/Users/valtterimj/Downloads/Työ/Aalto/Hera/hera_spice/kernels/mk/hera_ops.tm"

def load_meta_kernel(meta_kernel_path: str):
    """Load the given meta-kernel (.tm) file."""
    spice.furnsh(meta_kernel_path)
    print(f"Loaded meta-kernel: {meta_kernel_path}")

def unload_all_kernels():
    spice.kclear()
    print("Unloaded all kernels.")

def list_loaded_kernels():
    """Prints all currently loaded SPICE kernels and their types."""
    count = spice.ktotal("ALL")
    print(f"Total kernels loaded: {count}")
    for i in range(count):
        file, type_, mk_path, index = spice.kdata(i, "ALL")
        print(f"[{i}] Type: {type_:<5}  File: {file}")


def get_mk_identifier() -> str:
    """
    Query 'MK_IDENTIFIER' from the kernel pool.
    parameters
    identifier, start index, number of values, max length of string
    """

    mk_id = spice.gcpool("MK_IDENTIFIER", 0, 1, 80)

    if len(mk_id) > 0:
        return mk_id[0]
    else:
        return 'UNK'

def get_sc_id(frame_name:str = 'MILANI_SPACECRAFT'):
    frame_name = "MILANI_SPACECRAFT"
    frame_id = spice.namfrm(frame_name)
    return frame_id

def get_sclk(date_ob:str, frame_name:str = 'MILANI_SPACECRAFT') -> str:
    """
    converts the UTC observation time into SC clock SPICE format

    DISCLAIMER: Does not work with time form telemetry (1). For some reason -999 must be added to sc_id for MILANI sc id
    """
    # Convert UTC to ET
    et = spice.utc2et(date_ob)
    sc_id = get_sc_id(frame_name)
    sclk_str = spice.sce2s(sc_id - 999, et) # What is the correct sc_id value?
    return sclk_str

def query_sun_position_vectors(
        utc_time: str = test_time,
		frame: str = 'J2000', # Is this the correct frame?
		observer: str = 'MILANI_SPACECRAFT'
    ):
    """
	Query the position vectors of the Sun in a specified time frame.

	Parameters:
	utc_time (str): The UTC time for which the position vectors are required.
	frame (str): The reference frame.
	observer (str): The observing body.

	Returns:
	tuple: A tuple (X, Y, Z) representing the Sun's position vector.
	"""
    et = spice.str2et(utc_time)
	
    state, _ = spice.spkezr("SUN", et, frame, "NONE", observer)
    position = state[:3] # X, Y, Z

    return position

def query_solar_distance(
        target: str = 'SUN',
        utc_time: str = test_time,
        frame: str = 'J2000',
        observer: str = 'MILANI_SPACECRAFT'

    ):
    """
    Query the distance between a spacecraft and the Sun.
    
    Parameters:
    spice_metakernel_path (str): Path to the SPICE metakernel file.
    target (str): The spacecraft of interest.
    utc_time (str): The UTC time for which the distance is required.
    frame (str): The reference frame for calculations.
    observer (str): The observing body (default is the Sun).
    
    Returns:
    float: Distance between the spacecraft and the Sun in kilometers.
    
    """

    # Query spacecraft position relative to the Sun
    position = query_spacecraft_position_vectors(
        target, utc_time, frame, observer
    )


load_meta_kernel(spice_mk_ops)
# print(get_mk_identifier())
# print(get_sclk(utilities.read_telemetry(telemetry_path, 'VIS')['DATE-OB']))
# print(get_sclk(test_time))
# print(query_sun_position_vectors())

unload_all_kernels()



