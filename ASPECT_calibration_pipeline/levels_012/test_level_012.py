import os
import modules.utilities as utilities
import modules.convertToFits as convertToFits

acq_path = os.path.join(os.getcwd(), 'test_data/ASPECT_fly_images/acqseq_100')
meta_folder = os.path.join(acq_path, 'meta')

# Getting channels frame counts and original fiel names from acquisition folder
def test_channel_frames_names():
    acq_folder = utilities.get_acq_folder(acq_path)
    channels_and_frames = utilities.get_channel_frames_names(acq_folder)
    print(channels_and_frames)

# Getting acquisition ID and acquisition sequence ID 
def test_acqseq():
    acq_folder = utilities.get_acq_folder(acq_path)
    result = utilities.get_acqSeq(acq_folder)
    print(result)

# Read telemetry and retrive metadata
def test_telemetry():
    telemetry_path = os.path.join(meta_folder, 'telemetry.json')

    result = utilities.read_telemetry(telemetry_path, 'VIS')
    print(result)

def test_spice_metadata():
    telemetry_path = os.path.join(meta_folder, 'telemetry.json')
    utilities.collect_spice_metadata(telemetry_path, '', '')

# Test the fits file conversion as a whole
def test_convert_to_fits():
    convertToFits.convert_to_fits(acq_path, None)

"""
Function calls after this

"""
# test_acqseq()
# test_channel_frames_names()
# test_telemetry()
# test_convert_to_fits()
test_spice_metadata()





# Python3 ASPECT_calibration_pipeline/levels_012/test_level_012.py