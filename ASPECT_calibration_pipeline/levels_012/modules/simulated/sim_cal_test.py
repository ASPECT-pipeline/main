import os
import numpy as np
import matplotlib.pyplot as plt

cal_data_folder = os.path.join(os.getcwd(), 'ASPECT_calibration_pipeline/levels_012/modules/simulated/data')

VISDarkFrameFile = os.path.join(cal_data_folder, 'dark-frame-VIS-10ms.bin')
VISCalCoefFile =  os.path.join(cal_data_folder, 'calibration-coefficients-VIS.dat')
VISExampleFrameFile =  os.path.join(cal_data_folder, 'example-1-vis-10ms-0000.bin')

VISDetectorResolution =  [1024, 1024]
VISIntegrationTime = 10 # milliseconds
NIRDetectorResolution =  [640, 512]
NIRIntegrationTime = 20 # milliseconds

distanceToSun = 1.0 # au

def display_vis_image(file_name):
        
    data = np.fromfile(file_name, dtype='int16')/0.16
    hi = np.max(data)
    mean = np.mean(data)
    lo = np.min(data)
    typ = data.dtype
    shape = data.shape
    head = data[:5]

    img = data.reshape(VISDetectorResolution[1],-1)

    print(f'Image DN:')
    print(f'min : {lo}')
    print(f'mean : {mean}')
    print(f'max : {hi}')
    print(f'type : {typ}')
    print(f'shape : {shape}')
    print(f'head : {head}')

    plt.figure()
    plt.imshow(img, cmap='gray', norm=None, vmin=0)
    # plt.colorbar(img)
    plt.show()

    darkVIS = np.fromfile(VISDarkFrameFile,dtype=np.uint16)
    print(f"Dark Min, mean, and max values: {np.min(darkVIS)}, {np.mean(darkVIS)}, {np.max(darkVIS)}")

    
    VISdata1 = data-darkVIS

    print(f"Min, mean, and max values: {np.min(VISdata1)}, {np.mean(VISdata1)}, {np.max(VISdata1)}")
    im=plt.imshow(VISdata1.reshape(VISDetectorResolution[1],-1), cmap="gray", norm=None, vmin=0)
    plt.colorbar(im)
    plt.show()

    wlInd = 0
    VIScc = np.loadtxt(VISCalCoefFile)
    print(f"Wavelength is {VIScc[wlInd,0]} nm")
    VISdata2 = VISdata1 * VIScc[wlInd,1] / (VISIntegrationTime * distanceToSun**2)

    print(f"Min, mean, and max values: {np.min(VISdata2)}, {np.mean(VISdata2)}, {np.max(VISdata2)} W sr^-1 m^-2")
    im=plt.imshow(VISdata2.reshape(VISDetectorResolution[1],-1), cmap="gray", norm=None, vmin=0)
    plt.colorbar(im)
    plt.show()
"""
Python3 ASPECT_calibration_pipeline/levels_012/modules/simulated/sim_cal_test.py
"""

data_folder = os.path.join(os.getcwd(), 'test_data/ASPECT_simulated_images/2027-03-23_06_00_00-McEwen/acq_000')
nir1_000 = os.path.join(data_folder, 'dc_1_exp_000.bin')
vis1_000 = os.path.join(data_folder, 'dc_0_exp_000.bin')

vis_example = os.path.join(os.getcwd(), 'ASPECT_calibration_pipeline/levels_012/modules/simulated/calibration/example-1-vis-10ms-0000.bin')
nir_example = os.path.join(os.getcwd(), 'ASPECT_calibration_pipeline/levels_012/modules/simulated/calibration/example-1-nir1-20ms-0000.bin')

display_vis_image(vis1_000)
# display_vis_image(vis_example)