import numpy as np
from pathlib import Path
import levels_012.modules.simulated.postprocessing_functions as post


def sim_to_radiance_factor(data, channel, i):

    if channel in ['NIR1', 'NIR2']:
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
        qeDataName                      = (Path(__file__).parent / 'ASPECT-NIR-quantum-efficiency.txt').resolve()

        # NOISE
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
        wl = wlNIR1[i]
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

        data1 = post.backToR(data.flatten(), finalConvFunc, wl, darkCurrent, intTime, fullWellCapacity,
                    darkBackgroundQ=darkBackground, verbose=verbose, bitDepth=bitDepth, extraCoef=1/0.16)
        
        return data1.reshape(detectorResolution[1],-1)
    
    elif channel == 'Vis':
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

        verbose = False
        # Wavelengths
        wlVIS         =  np.array([675,690,705,720,735,750,765,780,795,810,825],np.float64)
        wl = wlVIS[i]
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

        data1 = post.backToR(data, finalConvFunc, wl, darkCurrent, intTime, fullWellCapacity,
                       darkBackgroundQ=darkBackground, verbose=verbose, bitDepth=bitDepth, extraCoef=1/0.16)

        return data1.reshape(detectorResolution[1],-1)

    else: 
        print(f'Invalid channel ({channel}) for simulated radiance conversion')