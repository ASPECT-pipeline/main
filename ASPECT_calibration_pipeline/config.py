"""
This file contains values for the pipeline. Modify the values to match your usecase.
"""
# INPUT path for the acquisition data
# input_directory: str = "/Users/valtterimj/Downloads/Työ/Aalto/Hera/Pipeline/main/test_data/ASPECT_in-flight-dark_250225/acqseq_100"

input_directory: str = "/Users/valtterimj/Downloads/Työ/Aalto/Hera/Pipeline/main/test_data/ASPECT_simulated_images/2027-03-23_06_00_00-McEwen"
input_directory: str = "/Users/valtterimj/Downloads/Työ/Aalto/Hera/Pipeline/main/pipeline_results/ASPECT_simulated_20270323_McEwen"

# input_directory: str = "/Users/valtterimj/Downloads/Työ/Aalto/Hera/Pipeline/main/test_data/ASPECT_Autoseq_20250820/Dark/acqseq_110"


# input_directory: str = "/Users/valtterimj/Downloads/Työ/Aalto/Hera/Pipeline/main/test_data/ASPECT_Autoseq_20240809/acqseq_503"

# OUTPUT path where the results are saved
output_directory: str = "/Users/valtterimj/Downloads/Työ/Aalto/Hera/Pipeline/main/pipeline_results/ASPECT_simulated_20270323_McEwen"

# output_directory: str = "/Users/valtterimj/Downloads/Työ/Aalto/Hera/Pipeline/main/pipeline_results/ASPECT_autosequence_20250820/Dark"

# output_directory: str = "/Users/valtterimj/Downloads/Työ/Aalto/Hera/Pipeline/main/pipeline_results/ASPECT_in-flight-dark_250225"


# Is the data differetially encoded
differential: bool = False

# Meta data
INSTRUME:   str = 'ASPECT'              # Camera ID
ORIGIN:     str = 'ESA-HERA'                
SWCREATE:   str = 'ASPECTCAL'           # Software identification
MISSPHAS:   str = 'SIMULATED'        # Hera mission phase ID
OBSERVPH:   str = ''                 # Hera observation ID
OBSTARGT:   str = 'DIDYMOS'                # Observation target
OBJECT:     str = 'Didymos'                # Observed object
TARGET:     str = 'DIDYMOS'             # Observed target (SPICE)
SC_CLK:     str = 'UNK'                 # Spacecraft clock Hera instrument format: '13480572:349872'

sc_clock_seconds: int = 0 # Spacecraft clock in seconds
sc_clock_offset: int = 0  # Offset of sc_clock

# Adjust this to to point to the directory containing calibration files 
calibration_directory = "/Users/valtterimj/Downloads/Työ/Aalto/Hera/Pipeline/main/ASPECT_calibration_pipeline/files"

# Adjust these to point to the locations of spice metakernels
spice_mk_plan = "/Users/valtterimj/Downloads/Työ/Aalto/Hera/hera_spice/kernels/mk/hera_plan.tm"
spice_mk_ops = "/Users/valtterimj/Downloads/Työ/Aalto/Hera/hera_spice/kernels/mk/hera_ops.tm"

# Adjust this to point to the metakernel to be used for FITS header data
spice_mk = spice_mk_plan

pipeline = '3' # Separate with '-' e.g. '1-2-3'

# Which instrument channels want to include
instrument = 'vis-nir1-nir2' 

models = 'M'

initGuess = [[0.1, 950, 150], [0.01, 1250, 50]] # for MGM


"""
Constants do not change unless you know what your are modifying.
"""


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