import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import postprocessing_functions as post


cal_data_folder = os.path.join(os.getcwd(), 'ASPECT_calibration_pipeline/levels_012/modules/simulated/data')
cal_fits_data_folder = os.path.join(os.getcwd(), 'ASPECT_calibration_pipeline/calibration_data/SIMULATED/DARKS')

VISDarkFrameFile = os.path.join(cal_data_folder, 'dark-frame-VIS-10ms.bin')
vis_dark_fits = os.path.join(cal_fits_data_folder, 'AS0_DARK.fits')
VISCalCoefFile =  os.path.join(cal_data_folder, 'calibration-coefficients-VIS.dat')
VISExampleFrameFile =  os.path.join(cal_data_folder, 'example-1-vis-10ms-0000.bin')

NIRDarkFrameFile = os.path.join(cal_data_folder, 'dark-frame-NIR1or2-20ms.bin')
NIR_dark_fits = os.path.join(cal_fits_data_folder, 'AS1_DARK.fits')
NIRCalCoefFile =  os.path.join(cal_data_folder, 'calibration-coefficients-NIR1.dat')
NIRExampleFrameFile =  os.path.join(cal_data_folder, 'example-1-nir1-10ms-0000.bin')

VISDetectorResolution =  [1024, 1024]
VISIntegrationTime = 10 # milliseconds
NIRDetectorResolution =  [640, 512]
NIRIntegrationTime = 20 # milliseconds

distanceToSun = 1.0 # au

def display_vis_image(file_name):
        
    data = np.fromfile(file_name, dtype='int16')
    hi = np.max(data)
    mean = np.mean(data)
    lo = np.min(data)
    typ = data.dtype
    shape = data.shape
    head = data[:5]
    img = data.reshape(VISDetectorResolution[1],-1)

    integrationTime = 10
    femtoAmpereToQE = 6241.509074460763
    darkCurrent       =  0.0200272 # dark current in femtoamperes


    plt.figure()
    plt.imshow(img, cmap='gray', norm=None, vmin=0)
    # plt.colorbar(img)
    plt.show()

    darkVIS = np.fromfile(VISDarkFrameFile,dtype=np.uint16)
    print(f"Dark bg: {darkVIS}")
    with fits.open(vis_dark_fits) as hdul:
        dark = hdul[0].data
    print(f"FITS Dark Min, mean, and max values: {np.min(dark)}, {np.mean(dark)}, {np.max(dark)}")
    
    VISdata1 = data.reshape(1024,1024)-dark
    VISdata1 = np.array(VISdata1,dtype='float64')
    # dc = darkCurrent * femtoAmpereToQE * integrationTime * 10e-4
    # VISdata1 -= dc
    VISdata1 = np.clip(VISdata1, 0, None)

    print(f"Min, mean, and max values: {np.min(VISdata1)}, {np.mean(VISdata1)}, {np.max(VISdata1)}")
    im=plt.imshow(VISdata1.reshape(VISDetectorResolution[1],-1), cmap="gray", norm=None, vmin=0)
    plt.colorbar(im)
    plt.show()
    wlInd = 0
    VIScc = np.loadtxt(VISCalCoefFile)
    print(f"Wavelength is {VIScc[wlInd,0]} nm")
    VISdata2 = VISdata1 * VIScc[wlInd,1] / (VISIntegrationTime * distanceToSun**2)
    VISdata2 /= 0.16

    print(f"Min, mean, and max values: {np.min(VISdata2)}, {np.mean(VISdata2)}, {np.max(VISdata2)} W sr^-1 m^-2")
    im=plt.imshow(VISdata2.reshape(VISDetectorResolution[1],-1), cmap="gray", norm=None, vmin=0)
    plt.colorbar(im)
    plt.show()

    lambertRadianceAt1au = 217.0
    au = distanceToSun
    c = lambertRadianceAt1au / au**2

    VISdata3 = VISdata2 / c

    print(f"Min, mean, and max values: {np.min(VISdata3)}, {np.mean(VISdata3)}, {np.max(VISdata3)} I/F")
    im=plt.imshow(VISdata3.reshape(VISDetectorResolution[1],-1), cmap="gray", norm=None, vmin=0)
    plt.colorbar(im)
    plt.show()

def display_nir_image(file_name):
    data = np.fromfile(file_name, dtype='int16')
    hi = np.max(data)
    mean = np.mean(data)
    lo = np.min(data)
    typ = data.dtype
    shape = data.shape
    head = data[:5]

    img = data.reshape(NIRDetectorResolution[1],-1)

    print(f'Image DN:')
    print(f'type : {typ}')
    print(f'shape : {shape}')
    print(f'head : {head}')

    integrationTime = 20
    femtoAmpereToQE = 6241.509074460763
    darkCurrent       =  0.5 # dark current in femtoamperes

    print(f"min, mean, and max values: {lo}, {mean}, {hi}")
    plt.figure()
    plt.imshow(img, cmap='gray', norm=None, vmin=0)
    # plt.colorbar(img)
    plt.show()

    darkNIR = np.fromfile(NIRDarkFrameFile,dtype=np.uint16)
    with fits.open(NIR_dark_fits) as hdul:
        dark = hdul[0].data
    print(f"FITS Dark Min, mean, and max values: {np.min(dark)}, {np.mean(dark)}, {np.max(dark)}")
    NIRdata1 = np.array(data,dtype='float64')
    NIRdata1 /= 0.16
    dark = np.array(dark,dtype='float64')
    dark /= 0.16
    print(f"Dark after correction Min, mean, and max values: {np.min(dark)}, {np.mean(dark)}, {np.max(dark)}")
    print(f"NIR after correction Min, mean, and max values: {np.min(NIRdata1)}, {np.mean(NIRdata1)}, {np.max(NIRdata1)}")
    NIRdata1 = NIRdata1.reshape(512,640)-dark

    dc = darkCurrent * femtoAmpereToQE * integrationTime * 10e-4
    # NIRdata1 -= dc
    # NIRdata1 /= 0.16
    NIRdata1 = np.clip(NIRdata1, 0, None)

    print(f" After removing dcb Min, mean, and max values: {np.min(NIRdata1)}, {np.mean(NIRdata1)}, {np.max(NIRdata1)}")
    im=plt.imshow(NIRdata1.reshape(NIRDetectorResolution[1],-1), cmap="gray", norm=None, vmin=0)
    plt.colorbar(im)
    plt.show()

    wlInd = 0
    NIRcc = np.loadtxt(NIRCalCoefFile)
    print(f"Wavelength is {NIRcc[wlInd,0]} nm")
    NIRdata2 = NIRdata1 * NIRcc[wlInd,1] / (NIRIntegrationTime * distanceToSun**2)
    # NIRdata2 /= 0.16

    print(f"Min, mean, and max values: {np.min(NIRdata2)}, {np.mean(NIRdata2)}, {np.max(NIRdata2)} W sr^-1 m^-2")
    im=plt.imshow(NIRdata2.reshape(NIRDetectorResolution[1],-1), cmap="gray", norm=None, vmin=0)
    plt.colorbar(im)
    plt.show()

    lambertRadianceAt1au = 217.0
    au = distanceToSun
    c = lambertRadianceAt1au / au**2


    NIRdata3 = NIRdata2 / c

    print(f"Min, mean, and max values: {np.min(NIRdata3)}, {np.mean(NIRdata3)}, {np.max(NIRdata3)} I/F")
    im=plt.imshow(NIRdata3.reshape(NIRDetectorResolution[1],-1), cmap="gray", norm=None, vmin=0)
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

# display_vis_image(vis1_000)
# display_vis_image(vis_example)

display_nir_image(nir_example)
# display_nir_image(nir1_000)