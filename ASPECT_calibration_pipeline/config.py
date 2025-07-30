"""
This file contains values for the pipeline. Modify the values to match your usecase.
"""
# INPUT path for the acquisition data
input_directory: str = "/Users/valtterimj/Downloads/Työ/Aalto/Hera/Pipeline/main/test_data/test_output/ASPECT_simulated/ASPECT_simulated_20270323_McEwen/"

# OUTPUT path where the results are saved
output_directory: str = "/Users/valtterimj/Downloads/Työ/Aalto/Hera/Pipeline/main/test_data/test_output/ASPECT_simulated/ASPECT_simulated_20270323_McEwen/"


# Is the data differetially encoded
differential: bool = False

# Meta data
software: str = 'ASPECTCAL v1.0'
missphase: str = 'SIMULATE'
observph: str = 'McEwen'
target: str = 'DIDYMOS'
object: str = 'dark'
sc_clock_seconds: int = 0 # Spacecraft clock in seconds
sc_clock_offset: int = 0  # Offset of sc_clock

# Adjust these to point to the locations of spice metakernels
spice_mk_plan = "/Users/valtterimj/Downloads/Työ/Aalto/Hera/hera_spice/kernels/mk/hera_plan.tm"
spice_mk_ops = "/Users/valtterimj/Downloads/Työ/Aalto/Hera/hera_spice/kernels/mk/hera_ops.tm"

# Adjust this to point to the metakernel to be used for FITS header data
spice_mk = spice_mk_plan

# INSTRUMENT DATA
instrument = 'vis-nir1-nir2' 

pipeline = '2' # Separate with '-' e.g. '1-2-3'

models = 'C'

initGuess = [[0.1, 950, 150], [0.01, 1250, 50]] # for MGM

channel_map = {
    0 : 'VIS',
    1 : 'NIR1',
    2 : 'NIR2',
    3 : 'SWIR'
}

# CONSTANTS

kelvin: float = 273.15