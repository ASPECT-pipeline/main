import spiceypy as spice
import numpy as np
from scipy.spatial.transform import Rotation as R
import modules.utilities as utilities
from typing import Any, Dict, List, Tuple, Optional
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
# test_time = '2027-01-02T05:40:46.6666'


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


def km_to_au(km) -> float:
    AU_PER_KM = 149597870.7
    return km / AU_PER_KM

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

def get_sc_id(frame_name:str = 'MILANI_SPACECRAFT') -> int:
    frame_name = "MILANI_SPACECRAFT"
    frame_id = spice.namfrm(frame_name)
    return frame_id

def get_sclk(date_ob:str, frame_name:str = 'MILANI_SPACECRAFT') -> str:
    """
    converts the UTC observation time into SC clock SPICE format

    Parameters:
    date_ob (str): date and time of the observation retrieved from telemetry
    frame_name (str): frame of the et time to be converted

    DISCLAIMER: Does not work with time form telemetry (1). For some reason -999 must be added to sc_id for MILANI sc id
    """
    # Convert UTC to ET
    et = spice.utc2et(date_ob)
    sc_id = get_sc_id(frame_name)
    sclk_str = spice.sce2s(sc_id - 999, et) # What is the correct sc_id value?
    return sclk_str

def query_position_distance(
        target: str = 'SUN',
        utc_time: str = test_time,
		frame: str = 'J2000', # Is this the correct frame?
        abcorr: str = 'NONE',
		observer: str = 'MILANI_SPACECRAFT'
    )-> Tuple[np.ndarray, float]:
    """
	Query the position vectors of the target from the perspective of the observer in a specified time within a given reference frame.

	Parameters:
    target (str): The target body
	utc_time (str): The UTC time for which the position vectors are required.
	frame (str): The reference frame.
    abcorr (str): Aberration correction flag
	observer (str): The observing body.

	Returns:
	tuple:  (A tuple (X, Y, Z) representing the target's position vector relative to the observer im km.),
            (distnace between the target and the observer in km)
	"""
    et = spice.str2et(utc_time)
    state, _ = spice.spkezr(target, et, frame, abcorr, observer)
    position = state[:3] # X, Y, Z
    distance_km = np.linalg.norm(position)
    distance_au = km_to_au(distance_km)

    return position, distance_au


def query_spacecraft_quaternions(
        frame_name: str = 'HERA_SPACECRAFT',
        utc_time: str = test_time,
        tol: int = 1.0,
        ref: str = 'J2000'
    ):

    et = spice.utc2et(utc_time)
    inst_id = get_sc_id(frame_name=frame_name)
    print(f'inst_id: {inst_id}')
    result = spice.ckgpav(-9102001, et, tol, ref)
    print(type(result))
    print(result)


def list_ck_instruments():
    """
    Lists all currently loaded CK files and the NAIF instrument IDs they contain.
    """
    n_ck = spice.ktotal("CK")
    if n_ck == 0:
        print("No CK kernels are currently loaded.")
        return

    print(f"Loaded CK kernels: {n_ck}\n")

    for i in range(n_ck):
        try:
            ck_file, _, _, _ = spice.kdata(i, "CK")
            print(f"[{i}] CK File: {ck_file}")

            # Get instrument IDs from this CK file
            ids = spice.ckobj(ck_file)
            if ids:
                for inst_id in ids:
                    try:
                        name = spice.bodc2n(inst_id)
                    except spice.stypes.SpiceyError:
                        name = "(name not found)"
                    print(f"    Instrument ID: {inst_id} → {name}")
            else:
                print("    No instrument IDs found in this CK file.")
        except spice.stypes.SpiceyError as e:
            print(f"    Error reading CK file: {e}")

    print("\nDone.")



load_meta_kernel(spice_mk_ops)
# print(get_mk_identifier())
# print(get_sclk(utilities.read_telemetry(telemetry_path, 'VIS')['DATE-OB']))
# print(get_sclk(test_time))
# print(type(query_position_distance()[1]))
print(query_spacecraft_quaternions())
# list_ck_instruments()

unload_all_kernels()



# Python3 ASPECT_calibration_pipeline/levels_012/hera_spice.py