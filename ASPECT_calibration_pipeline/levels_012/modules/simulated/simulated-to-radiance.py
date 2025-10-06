import numpy as np
import matplotlib.pyplot as plt
import postprocessing_functions as post
import os
from pathlib import Path

def display_vis_image(file_name):
    detectorResolution              =  [1024, 1024]
    FoV                             =  [0.174533, 0.174533]
    distanceToSun                   =  1.0 # au
    apertureDiameter                =  13.62 # mm
    opticsTransmission              =  0.9
    spectralTransmissionStrength    =  0.3 # Constant transmission factor through the window
    spectralTransmissionWindowWidth =  25 # The width of the transmission windon, in nanometers. Must be an integer
    extraWL = int(np.ceil(spectralTransmissionWindowWidth/2)) # Can be just computed, additional wl to counteract the edge effects in spectral transmission convolution later
    wlStart                         =  650-extraWL # The starting wavelength for the spectrometer, nm
    wlEnd                           =  900+extraWL # The ending wavelength for the spectrometer, nm
    qeDataName                      = (Path(__file__).parent / 'ASPECT-VIS-quantum-efficiency.txt').resolve()

    # NOISE
    readNoise         =  8 # read noise e-
    darkCurrent       =  0.0200272 # dark current in femtoamperes
    fullWellCapacity  =  10000 # in e-
    darkBackground    =  100 # fixed dark background level in e-
    bitDepth          =  2**12

    # Integration times
    intTime           = 10

    # Wavelengths
    wlVIS         =  np.array([675,690,705,720,735,750,765,780,795,810,825],np.float64)
    verbose = False
    wlArray = np.arange(wlStart+extraWL, wlEnd-extraWL, 1) # Range for 'good' wavelengths

    # Conversion from image values to radiant flux
    convFact1 = post.toRadiantFlux(au=distanceToSun, FoV=FoV, apertureDiameter=apertureDiameter, \
                            detectorResolution=detectorResolution, opticsTransmission=opticsTransmission, verbose=verbose)

    # Conversion from radiant flux to spectral flux
    [wla, convFact2] = post.toSpectralFlux(convFact1, wlStart, wlEnd, verbose=verbose)

    # Conversion from spectral flux to spectral transmission after the spectral filter
    convFact3 = post.toSpectralTransmission(convFact2, spectralTransmissionStrength, spectralTransmissionWindowWidth, verbose=verbose)

    # Final conversion from spectral transmission to e- charges at the detector pixel level
    finalConvFunc = post.toQEMonoFunction(convFact3, wla, qeDataName, verbose=verbose, QEOrder=1)
        
    data = np.fromfile(file_name, dtype='int16')
    hi = max(data)
    lo = min(data)
    typ = data.dtype
    shape = data.shape
    head = data[:5]

    wl = wlVIS[0]

    img = data.reshape(detectorResolution[1],-1)

    print(f'Image DN:')
    print(f'max : {hi}')
    print(f'min : {lo}')
    print(f'type : {typ}')
    print(f'shape : {shape}')
    print(f'head : {head}')

    plt.figure()
    plt.imshow(img, cmap='gray', norm=None, vmin=0)
    # plt.colorbar(img)
    plt.show()

    data1 = post.backToR(data, finalConvFunc, wl, darkCurrent, intTime, fullWellCapacity,
                       darkBackgroundQ=darkBackground, verbose=verbose, bitDepth=bitDepth, extraCoef=1/0.16)
    hi = max(data1)
    lo = min(data1)
    typ = data1.dtype
    shape = data1.shape
    head = data1[:5]

    img = data1.reshape(detectorResolution[1],-1)

    print()
    print(f'Image radiance factor:')
    print(f'max : {hi}')
    print(f'min : {lo}')
    print(f'type : {typ}')
    print(f'shape : {shape}')
    print(f'head : {head}')

    plt.figure()
    plt.imshow(img, cmap='gray', norm=None, vmin=0)
    # plt.colorbar(img)
    plt.show()

    data2 = post.backToRadiance(data1, distanceToSun)

    hi = max(data2)
    lo = min(data2)
    typ = data2.dtype
    shape = data2.shape
    head = data2[:5]

    img = data2.reshape(detectorResolution[1],-1)

    print()
    print(f'Image Radiance:')
    print(f'max : {hi}')
    print(f'min : {lo}')
    print(f'type : {typ}')
    print(f'shape : {shape}')
    print(f'head : {head}')

    plt.figure()
    plt.imshow(img, cmap='gray', norm=None, vmin=0)
    # plt.colorbar(img)
    plt.show()

def display_nir_image(file_name):

    detectorResolution              =  [640, 512]
    FoV                             =  [0.116588, 0.0935496]
    distanceToSun                   =  1.0 # au
    apertureDiameter                =  13.62 # mm
    opticsTransmission              =  0.9
    spectralTransmissionStrength    =  0.3 # Constant transmission factor through the window
    spectralTransmissionWindowWidth =  25 # The width of the transmission windon, in nanometers. Must be an integer
    extraWL = int(np.ceil(spectralTransmissionWindowWidth/2)) # Can be just computed, additional wl to counteract the edge effects in spectral transmission convolution later
    wlStart                         =  850-extraWL # The starting wavelength for the spectrometer, nm
    wlEnd                           =  1600+extraWL # The ending wavelength for the spectrometer, nm
    # qeDataName                      =  "ASPECT-NIR-quantum-efficiency.txt" # The quantum efficiency of the detector in a file
    # qeDataName                      = os.path.join(os.getcwd(), 'ASPECT_calibration_pipeline/levels_012/modules/simulated/ASPECT-NIR-quantum-efficiency.txt')
    qeDataName                      = (Path(__file__).parent / 'ASPECT-NIR-quantum-efficiency.txt').resolve()
    readNoise         =  85 # read noise e-
    darkCurrent       =  0.5 # dark current in femtoamperes
    fullWellCapacity  =  2**15 # in e-
    darkBackground    =  5900 # fixed dark background level in e-
    bitDepth          =  2**14

    # Integration times
    intTime          = 20

    # Wavelengths
    wlNIR1          =  np.array([875,904.1666667,933.3333333,962.5,991.6666667,1020.833333,1050,1079.166667,
                                1108.333333,1137.5,1166.666667,1195.833333,1225],np.float64)


    verbose = False
    wlArray = np.arange(wlStart+extraWL, wlEnd-extraWL, 1) # Range for 'good' wavelengths

    # Conversion from image values to radiant flux
    convFact1 = post.toRadiantFlux(au=distanceToSun, FoV=FoV, apertureDiameter=apertureDiameter, \
                            detectorResolution=detectorResolution, opticsTransmission=opticsTransmission, verbose=verbose)

    # Conversion from radiant flux to spectral flux
    [wla, convFact2] = post.toSpectralFlux(convFact1, wlStart, wlEnd, verbose=verbose)

    # Conversion from spectral flux to spectral transmission after the spectral filter
    convFact3 = post.toSpectralTransmission(convFact2, spectralTransmissionStrength, spectralTransmissionWindowWidth, verbose=verbose)

    # Final conversion from spectral transmission to e- charges at the detector pixel level
    finalConvFunc = post.toQEMonoFunction(convFact3, wla, qeDataName, verbose=verbose, QEOrder=1)
        
    data = np.fromfile(file_name, dtype='int16')
    hi = max(data)
    lo = min(data)
    typ = data.dtype
    shape = data.shape
    head = data[:5]

    wl = wlNIR1[0]

    img = data.reshape(detectorResolution[1],-1)

    print(f'Image DN:')
    print(f'max : {hi}')
    print(f'min : {lo}')
    print(f'type : {typ}')
    print(f'shape : {shape}')
    print(f'head : {head}')

    plt.figure()
    plt.imshow(img, cmap='gray', norm=None, vmin=0)
    # plt.colorbar(img)
    plt.show()

    data1 = post.backToR(data, finalConvFunc, wl, darkCurrent, intTime, fullWellCapacity,
                       darkBackgroundQ=darkBackground, verbose=verbose, bitDepth=bitDepth, extraCoef=1/0.16)
    hi = max(data1)
    lo = min(data1)
    typ = data1.dtype
    shape = data1.shape
    head = data1[:5]

    img = data1.reshape(detectorResolution[1],-1)

    print()
    print(f'Image radiance factor:')
    print(f'max : {hi}')
    print(f'min : {lo}')
    print(f'type : {typ}')
    print(f'shape : {shape}')
    print(f'head : {head}')

    plt.figure()
    plt.imshow(img, cmap='gray', norm=None, vmin=0)
    # plt.colorbar(img)
    plt.show()

    data2 = post.backToRadiance(data1, distanceToSun)

    hi = max(data2)
    lo = min(data2)
    typ = data2.dtype
    shape = data2.shape
    head = data2[:5]

    img = data2.reshape(detectorResolution[1],-1)

    print()
    print(f'Image Radiance:')
    print(f'max : {hi}')
    print(f'min : {lo}')
    print(f'type : {typ}')
    print(f'shape : {shape}')
    print(f'head : {head}')

    plt.figure()
    plt.imshow(img, cmap='gray', norm=None, vmin=0)
    # plt.colorbar(img)
    plt.show()
#### 
"""
Python3 ASPECT_calibration_pipeline/levels_012/modules/simulated/simulated-to-radiance.py
"""

data_folder = os.path.join(os.getcwd(), 'test_data/ASPECT_simulated_images/2027-03-23_06_00_00-McEwen/acq_000')
nir1_000 = os.path.join(data_folder, 'dc_1_exp_000.bin')
vis1_000 = os.path.join(data_folder, 'dc_0_exp_000.bin')

# display_nir_image(nir1_000)
# display_vis_image(vis1_000)