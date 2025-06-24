import os
import utilities

acq_path = os.path.join(os.getcwd(), 'test_data/ASPECT_fly_images/acqseq_100')
meta_folder = os.path.join(acq_path, 'meta')

def test_channel_frames_names():

    acq_folder = utilities.get_acq_folder(acq_path)

    channels_and_frames = utilities.get_channel_frames_names(acq_folder)

    print(channels_and_frames)


def test_acqseq():
    acq_folder = utilities.get_acq_folder(acq_path)
    result = utilities.get_acqSeq(acq_folder)
    print(result)

def test_telemetry():
    telemetry_path = os.path.join(meta_folder, 'telemetry.json')

    result = utilities.read_telemetry(telemetry_path, 'VIS')
    print(result)

# test_acqseq()
# test_channel_frames_names()
test_telemetry()



# Python3 ASPECT_calibration_pipeline/calibration_levels_012/test_level_012.py