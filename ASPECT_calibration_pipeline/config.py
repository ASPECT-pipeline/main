"""
This file contains values for the pipeline. Modify the values to match your usecase.
"""
# INPUT path for the acquisition data
# input_directory: str = "/Users/valtterimj/Downloads/Työ/Aalto/Hera/Pipeline/main/test_data/ASPECT_Autoseq_20240809/acqseq_505"
# input_directory: str = "/Users/valtterimj/Downloads/Työ/Aalto/Hera/Pipeline/main/test_data/levels_012_test/test_output/ASPECT_simulated/2027-03-23_06_00_00"
input_directory: str = "/Users/valtterimj/Downloads/Työ/Aalto/Hera/Pipeline/main/test_data/ASPECT_simulated_images/2027-03-23_06_00_00-McEwen"
input_directory: str = "/Users/valtterimj/Downloads/Työ/Aalto/Hera/Pipeline/main/test_data/test_output/ASPECT_simulated/2027-03-23_06_00_00/example-3"

# OUTPUT path where the results are saved
# output_directory: str = "/Users/valtterimj/Downloads/Työ/Aalto/Hera/pipeline/main/test_data/test_output/ASPECT_DIFF"
output_directory: str = "/Users/valtterimj/Downloads/Työ/Aalto/Hera/Pipeline/main/test_data/test_output/ASPECT_simulated/2027-03-23_06_00_00"


# Is the data differetially encoded
differential: bool = False

# Meta data
software: str = 'ASPECTCAL v1.0'
missphase: str = 'SIMULATE'
observph: str = 'example-3'
target: str = 'DIDYMOS'
object: str = 'Didymos'

# Adjust these to point to the locations of spice metakernels
spice_mk_plan = "/Users/valtterimj/Downloads/Työ/Aalto/Hera/hera_spice/kernels/mk/hera_plan.tm"
spice_mk_ops = "/Users/valtterimj/Downloads/Työ/Aalto/Hera/hera_spice/kernels/mk/hera_ops.tm"

# Adjust this to point to the metakernel to be used for FITS header data
spice_mk = spice_mk_plan

# INSTRUMENT DATA
instrument = 'nir1-nir2' 

pipeline = '3'

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