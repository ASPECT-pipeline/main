"""
This file contains values for the pipeline. Modify the values to match your usecase.
"""
from pathlib import Path
# Project directory
project_directory: str = "/Users/valtterimj/Downloads/Työ/Aalto/Hera/Pipeline/main/"

# INPUT path for the acquisition data
# input_directory: str = "/Users/valtterimj/Downloads/Työ/Aalto/Hera/Pipeline/main/test_data/ASPECT_simulated_images/2027-03-23_06_00_00-McEwen"
input_directory: str = "/Users/valtterimj/Downloads/Työ/Aalto/Hera/Pipeline/main/pipeline_results/ASPECT_simulated/D1D2_10km/acq_000"
# input_directory: str = "/Users/valtterimj/Downloads/Työ/Aalto/Hera/Pipeline/main/test_data/ASPECT_in-flight-dark_250225/acqseq_100"

# OUTPUT path where the results are saved
# output_directory: str = "/Users/valtterimj/Downloads/Työ/Aalto/Hera/Pipeline/main/pipeline_results/ASPECT_in-flight-dark_250225"
output_directory: str = "/Users/valtterimj/Downloads/Työ/Aalto/Hera/Pipeline/main/pipeline_results/ASPECT_simulated/D1D2_10km/acq_000"


# Is the data differetially encoded
differential: bool = False

# Meta data
INSTRUME:   str = 'ASPECT'              # Camera ID
ORIGIN:     str = 'ESA-HERA'            # Hera mission instruments
MISSPHAS:   str = 'SIMULATED'                    # Hera mission phase ID '002_CRUISE'
OSERV_ID:   str = 'D1D2_10km'                    # Hera observation ID                
SWCREATE:   str = 'ASPECTCAL'           # Software identification
SC_CLK:     str = 'UNK'                 # Spacecraft clock Hera instrument format: '13480572:349872'
OBJECT:     str = 'Didymos'                    # Observed object
TARGET:     str = 'DIDYMOS'                    # Observed target (SPICE)

sc_clock_seconds: int = 0 # Spacecraft clock in seconds
sc_clock_offset: int = 0  # Offset of sc_clock

# Adjust this to to point to the directory containing calibration files 
calibration_directory = "/Users/valtterimj/Downloads/Työ/Aalto/Hera/Pipeline/main/ASPECT_calibration_pipeline/calibration_data"

# Adjust these to point to the locations of spice metakernels
spice_mk_plan = "/Users/valtterimj/Downloads/Työ/Aalto/Hera/hera_spice/kernels/mk/hera_plan.tm"
spice_mk_ops = "/Users/valtterimj/Downloads/Työ/Aalto/Hera/hera_spice/kernels/mk/hera_ops.tm"

# Adjust this to point to the metakernel to be used for FITS header data
spice_mk = spice_mk_plan

pipeline = '3' # Separate with '-' e.g. '1-2-3'

# Which instrument channels want to include
instrument = 'Vis-NIR1-NIR2' 

models = 'T'

initGuess = [[0.1, 950, 150], [0.01, 1250, 50]] # for MGM


# parameters to tweak the data filtering
data_filtering = True
z_factor = 1.0 # increse -> smoother spectra (z_factor * fwhm)
z_threshold = 1.0 # increase -> less sensitive to outliers ((deriv - mu) / sigma) > z_threshold)


"""
Constants do not change unless you know what your are modifying.
"""

subdirs = {
    "pipeline"      : "ASPECT_calibration_pipeline",
    "calibration"   : "calibration_data",
    "flat_field"    : "FLATS",
    "dark_frames"   : "DARKS",
    "bad_pixels"    : 'BAD_PIXELS',
    "simulated"     : "SIMULATED",
    "solar"         : "SOLAR"
}

# Paths to calibration files
_path_dark = Path(project_directory) / subdirs['pipeline'] / subdirs['calibration'] / subdirs['dark_frames']
_path_flat = Path(project_directory) / subdirs['pipeline'] / subdirs['calibration'] / subdirs['flat_field']
_path_bad_pixels = Path(project_directory) / subdirs['pipeline'] / subdirs['calibration'] / subdirs['bad_pixels']

_path_solar_ssi = Path(project_directory) / subdirs['pipeline'] / subdirs['calibration'] / subdirs['solar'] / 'ssi_yearly_avg_e2024_c20250221.csv'

# Paths to simulated calibration files
_path_sim_dark = Path(project_directory) / subdirs['pipeline'] / subdirs['calibration'] / subdirs['simulated'] / subdirs['dark_frames']
_path_sim_flat = Path(project_directory) / subdirs['pipeline'] / subdirs['calibration'] / subdirs['simulated'] / subdirs['flat_field']
_path_sim_bad_pixels = Path(project_directory) / subdirs['pipeline'] / subdirs['calibration'] / subdirs['simulated'] / subdirs['bad_pixels']

_path_sim_coef = Path(project_directory) / subdirs['pipeline'] / subdirs['calibration'] / subdirs['simulated'] / 'COEF'


channel_map = {
    0 : 'Vis',
    1 : 'NIR1',
    2 : 'NIR2',
    3 : 'SWIR'
}

reverse_channel_map = {
    'Vis'  : 0,
    'NIR1' : 1,
    'NIR2' : 2,
    'SWIR' : 3
}


kelvin: float = 273.15