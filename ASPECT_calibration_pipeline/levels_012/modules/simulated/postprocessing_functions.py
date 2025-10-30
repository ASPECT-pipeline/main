import os
import re
import cv2
import glob
import time
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import io
from argparse import ArgumentParser
from IPython.display import Image, display

# TO DO:
#  2. Document each function properly, follow the example of the first functions under
#       'functions for image untis -> physical units conversion'


######################################################################
# Constants

## The radiance of Lambertian surface at the distance of
##   1 au from the Sun, in W m^-2 sr-1
lambertRadianceAt1au = 217.0

## File name for the file containing the spectral distribution of 
##   solar energy distribution
solarEnergyFileName = os.path.join(
        os.path.dirname(__file__), "data/solar-input-energy-spectral-distribution.txt")

##quantumEffData  =   os.path.join(
##        os.path.dirname(__file__), "../modules/data/ASPECT-NIR-quantum-efficiency.txt")

## Module-wide variable to hold the solar spectra once read from file
solarSpectra = None

## Conversion factor from flux energy into photons
energyToPhotonsCoef = 5.034116567542709e15

## Conversion factor from dark noise energy into unit charges (Qe) at the detector
femtoAmpereToQE = 6241.509074460763

######################################################################
# Functions

def isnotebook():
    """
    Detects if the script is being used from a notebook or a console.
    """

    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)

    except NameError:
        return False      # Probably standard Python interpreter


######################################################################
# Functions for image untis <-> physical units conversion

def aboveZero(data):
    """
    Filters out elements equal to zero.
    
    Input:
      data - (numpy) vector where above-zero elements are filtered
    
    Output:
      Filtered vector (is this a view (probably) or a copy of the vector?)
    """
    
    return data[data > 0]
    
#END aboveZero



def backToR(dn, qeFunc, wl, darkCurrent, integrationTime,  fullWellCapacity = None, darkBackgroundQ = None, darkBackgroundRC = None, verbose = False, bitDepth = None, extraCoef = None):
    """
    Converts the DN units back to radiance factor units.
    
    DOCUMENT BETTER...
    """

    vals = np.array(dn,dtype='float64')

    if (any(vals > bitDepth)):
        print(f"Error, some values are larger than bitdepth of {bitDepth}.")
        return None

    if (verbose):
        print(f"Starting from DN values {vals}.")

    if ((bitDepth is not None) and (fullWellCapacity is not None)):
        vals *= fullWellCapacity / bitDepth
        if (verbose):
            print(f'Scaling to bitdepth was from full-well {fullWellCapacity} to {bitDepth}.')
            print(f'Values scaled back to detector DNs: {vals}')

    
    # Dark constant background
    if darkBackgroundQ is None:
        if darkBackgroundRC is None:
            dcb = 0
        else:
            dcb = darkBackgroundRC * qeLoc * integrationTime * 10e-4
    else:
        dcb = darkBackgroundQ

    # Dark current in qe
    dc = darkCurrent * femtoAmpereToQE * integrationTime * 10e-4

    print(f'dark bg: {dcb}')
    vals -= dcb
    if (verbose):
        print(f"After removing dark background: {vals}.")

    print(f'dark current: {dc}')
    vals -= dc
    if (verbose):
        print(f"After removing dark current: {vals}.")

    wlLocal = np.array([wl]).flatten()
    qeLoc = qeFunc(wlLocal)

    vals /=  qeLoc * integrationTime * 10e-4
    vals = np.clip(vals, 0, None)
    
    if (extraCoef is not None):
        vals *= extraCoef
    
    return vals

#END backToR


def backToRadiance(rc, au, verbose = False):
    """
    Converts radiance factor back to radiance.
    
    DOCUMENT MORE...
    """

    # To radiance
    c = lambertRadianceAt1au / au**2
    if(verbose):
        print(f"*Radiance at {au} au is {c} W/m^2/sr")

    vals = rc * c

    return vals

#END backToRadiance


def computeSNR(rcData, qeFunc, wl, readNoise, darkCurrent, integrationTime, 
        fullWellCapacity = None, darkBackgroundQ = None, darkBackgroundRC = None, verbose = False):
    """
    Signal-to-noise ratio for specific radiance coefficient value. 
    
    Computes signal-to-noise ratio for given radiance coefficient value with
    given transformation function from radiance coefficients to qe charges at
    given wavelength.
    
    Input:
        rcData            - Pixel grayscale data (radiance coefficient)
                            of the region of interest.
        qeFunc            - e⁻ charges at the detector.
        wl                - Wavelength(s) of interest, in nm.
        readNoise         - Read noise on the sensor, in e⁻.
        darkCurrent       - Dark current on the dectector, in femptoamperes
        integrationTime   - Exposure time, in milliseconds
        fullWellCapacity  - Full-well capacity of the sensor, in e⁻. 
        darkBackgroundQ     - Constant dark background level in e-, default: None.
                              Note - overrides darkBackgroundRC if not None
        darkBackgroundRC    - Constant dark background level in radiance coefficient, default: None
                              Note - because of radiance-to-e- conversion, this is dependent on integration time.
        verbose           - True or False.
    
    Output:
        snr               - Signal-to-noise ratio
    """
    
    singleRc = not isinstance(rcData, (list, tuple, np.ndarray))
    singleWl = not isinstance(wl, (list, tuple, np.ndarray))
    single = singleRc and singleWl
    
    if (verbose):
        print("Computing SNR.")
        if (singleRc):
            print("*Single radiance coefficient.")
        else:
            print("*List of radiance coefficient values.")
        if (singleWl):
            print("*Single wavelength.")
        else:
            print("*List of wavelengths.")
    
    # Dark constant background
    if darkBackgroundQ is None:
        if darkBackgroundRC is None:
            dcb = 0
        else:
            dcb = darkBackgroundRC * qeLoc * integrationTime * 10e-4
    else:
        dcb = darkBackgroundQ


    if ((not singleRc) and (not singleWl)):
        signal = np.matmul(rcData.reshape(-1,1), qeFunc(wl).reshape(1,-1)).flatten()
    else:
        signal = rcData * qeFunc(wl)
    signal *= integrationTime * 10e-4
    signal += dcb
    if (fullWellCapacity is not None):
        if (np.any(signal > fullWellCapacity)):
            print("Warning, detector is saturated from signal")
    signal = np.clip(signal, 0, fullWellCapacity)
    
    # Dark current in qe
    dc = darkCurrent * femtoAmpereToQE * integrationTime * 10e-4
    if (fullWellCapacity is not None):
        if (np.any(dc > fullWellCapacity)):
            print("Warning, detector is saturated from dark current")
    dc = np.clip(dc, 0, fullWellCapacity)
    
    # Noise not including dark fixed pattern
    noise = np.sqrt(signal + readNoise**2)
    
    # Signal does not include dark fixed pattern
    snr = (signal-dcb)/noise
    
    if (not (singleRc and singleWl)):
        maxsig = np.max(signal)
        maxsnr = np.max(snr)
    
    if (verbose):
        if (single):
            print(f"*Dark-pattern-removed-signal is {signal-dcb}, Poisson noise component {np.sqrt(signal)}, read noise component {readNoise}, dark current component {np.sqrt(dc)}, dark background {dcb}.")
        else:
            print(f"*Maximum dark-pattern-removed signal is {maxsig-dcb}, Poisson noise component for that {np.sqrt(maxsig)}, read noise component {readNoise}, dark current component {np.sqrt(dc)}, dark background {dcb}.")
    
    if ((not singleRc) and (not singleWl)):
        return snr.reshape(len(wl),-1)
    else:
        return snr
    
#END computeSNR


def exportHsBin(hsData, BinBaseName, dataFolder):
    """
    Exports hyperspectral data into series of 16-bit raw binary streams.

    Input:
        hsData      -   Hyperspectral image data
        imageH      -   Image height, in pixels.
        BinBaseName -   String defining the name of the results.
        dataFolder  -   String defining the destination path for the results.

    Output:
        Bin streams saved to dataFolder.
    """

    # Check that folder exists, create otherwise
    if not os.path.exists(dataFolder):
        os.makedirs(dataFolder)
    
    for i in range(hsData.shape[1]):
        fn = dataFolder+"/"+BinBaseName+"-"+f"{i:0>4}"+".bin"
        bindata = hsData[:,i].astype(np.uint16)
        bindata.tofile(fn)

#END exportHsBin


def exportHsPNG(hsData, imageH, PNGBaseName, dataFolder):
    """
    Exports hyperspectral data into series of 16-bit PNG images.

    Input:
        hsData      -   Hyperspectral image data
        imageH      -   Image height, in pixels.
        PNGBaseName -   String defining the name of the results.
        dataFolder  -   String defining the destination path for the results.

    Output:
        PNGs saved to dataFolder.
    """

    # Check that folder exists, create otherwise
    if not os.path.exists(dataFolder):
        os.makedirs(dataFolder)
    
    for i in range(hsData.shape[1]):
        fn = dataFolder+"/"+PNGBaseName+"-"+f"{i:0>4}"+".png"
        cv2.imwrite(fn, hsData[:,i].reshape(imageH,-1).astype(np.uint16))

#END exportHsPNG


def exportHsMat(hsData, imageH, wl, intTime, MatName, dataFolder):
    """
    Exports hyperspectral data into Matlab hypercube

    Input:
        hsData      -   Hyperspectral image data
        imageH      -   Image height, in pixels.
        MatName     -   String defining the name of the results.
        dataFolder  -   String defining the destination path for the results.

    Output:
        MAT hypercube saved to dataFolder.
    """

    # Check that folder exists, create otherwise
    if not os.path.exists(dataFolder):
        os.makedirs(dataFolder)
    fn = dataFolder+"/"+MatName
    
    # Reshape data
    data = hsData.reshape(imageH,-1,(len(wl)))
    
    mdict = {
        'exposure_ms' : np.array([[intTime]],dtype=np.uint16),
        'setpoints' : np.array([[]]),
        'wavelengths' : np.array([wl]),
        'cube' : data.astype(np.uint16)
    }

    io.savemat(fn, mdict)
    
#END exportHsMat


def grayData2QE(wl, grayData, qefunction):
    """
    Converts image grayscale data to e⁻ charges at detector in given wavelength.

    Input:
        wl          -   Wavelength of interest, in nm.
        grayData    -   Grayscale image, as a numpy array?
        qeFunction  -   ?

    Output:
        Numpy array (?) containing e⁻ charges.

    """
    
    return qefunction(wl) * grayData

#END grayData2QE


def grayData2RadiantFlux(grayData, rfCoef):
    """
    Converts image grayscale data to radiant flux at detector pixel.

    Input:
        grayData    -   Grayscale image, as a numpy array?
        rfCoef      -   Radiant flux coefficient?

    Output:
        Numpy array (?) containing radiant flux.
    """
    
    return grayData * rfCoef

#END grayData2RadiantFlux


def grayData2SpectralFlux(grayData, spCoef):
    """
    Converts image grayscale data to spectral radiant flux at detector pixel.

    Input:
        grayData    -   Grayscale image, as a numpy array?
        spCoef      -   Spectral radiant flux coefficient?

    Output:
        Numpy array (?) containing spectral radiant flux.
    """
    
    return np.matmul(grayData.reshape([-1,1]), spCoef.reshape([1,-1]))

#END grayData2SpectralFlux


def importHsMat(MatName, dataFolder):
    """
    Imports hyperspectral data from Matlab hypercube

    Input:
        MatName     -   String defining the name of the results.
        dataFolder  -   String defining the destination path for the results.

    Output:
        (hsData, imH, wlSample, integrationTime)
    """
    
    mat = io.loadmat(dataFolder+"/"+MatName)
    
    wlSample = mat['wavelengths'].flatten()
    integrationTime = mat['exposure_ms'][0][0]

    # Reshape data
    hsData = mat['cube']
    ddim = hsData.shape
    imH = ddim[0]
    hsData = hsData.reshape((-1,len(wlSample)))
    
    return [hsData, imH, wlSample, integrationTime]
    
#END importHsMat


def importImageData(imagename, foldername = "Original", imageAlbedo = 0.5, 
        targetAlbedo = 1.0, verbose = False):
    """
    Imports image data and converts into radiance coefficient scale.

    Input:
        imagename       - String containing image name.
        foldername      - String containing name of the folder containing the image.
        imageAlbedo     - Albedo of the image.
        targetAlbedo    - Target albedo for the image.
        verbose         - True or False (optional, False by default)

    Output:
        grayData        -
        imageH          - Image height, in pixels.
        rcRange         - Maximum radiance coefficient value?
    """

    if (verbose):
        print("Importing an image and converting the RGB-values to radiance coefficient scale.")

    # Check that files exist
    ifn = os.path.join(foldername, imagename)
    if (not os.path.isfile(ifn)):
        print("Error: image file '"+ifn+"' not found")
        return
    
    # Import image
    img = cv2.imread(ifn, cv2.IMREAD_UNCHANGED)
    
    # Check image dimensions
    shape = np.shape(img)
    if (len(shape) != 3):
        print("Error: expecting an RBG image")
        return
    
    # Figure out image bit depth, but it should be 16
    dtype = img.dtype
    if (dtype == np.dtype('uint8')):
        btype = 8
    elif (dtype == np.dtype('uint16')):
        btype = 16
    else:
        print("Error: image bit depth was not 8 or 16")
        return
    maxval = 2**btype-1
    
    # Take grayscale data from G channel, convert to float and flatten to array
    grayData = img[:,:,2].flatten()/maxval

    if (verbose):
        print(f"*Read image from file '{imagename}'. Bit depth is {btype}, image resolution {img.shape[0:2]}.")
        print(f"*Image grayscale value range is {[np.min(grayData),np.max(grayData)]}.")

    
    # To radiance coefficient
    grayData *= targetAlbedo / imageAlbedo
    rcRange = [np.min(grayData),np.max(grayData)]
    if(verbose):
        print(f"*Image radiance coefficient value range is {rcRange}.")
    
    return [grayData, img.shape[0], rcRange[1]]

#END def importImageData


def importMetaData(folderpath):
    """
    Imports Blender metadata for the images, by reading the file created with the renders.
    "detectorResolution": numpy array of floats
    "FoV":  list of floats, in radians.

    Input:
        folderpath  - String with the name of the folder containing the json file.

    Output:
        Dictionary containing the following items:
            detectorResolution  - Array containing X and Y resolution in pixels.
            FoV                 - Array containing X and Y FoV in pixels.
            imageAlbedo         - Unitless.
    """

    parser = ArgumentParser()
    args, unknown = parser.parse_known_args()   # "unknown" required for ipython/jupyter.

    with open(f"{folderpath}/settings-copy.json", "r") as f:
        args.__dict__ = json.load(f)

    dict = {}

    dict.update({"detectorResolution": np.array(args.resolution, dtype=np.float64)}) # pixels

    dict.update({"FoV": [float(args.fov[0]), float(args.fov[1])]})

    dict.update({"imageAlbedo": args.imageAlbedo[0]})

    return dict

#END def importMetaData


def readSolarSpectra():
    """
    Reads solar spectra from file.

    Input:
        Hard-coded solar spectral distribution filepath.
    
    Output:
        ss  - Scipy's interpolation function containing solar spectra. 
    """
    
    with open(solarEnergyFileName,'r') as f:
        x = []
        y = []
        for line in f:
            txt = line.split()
            x.append(int(txt[0]))
            y.append(float(txt[1]))
    
    ss = interpolate.interp1d(x, y, 1)
    
    return ss

#END def readSolarSpectra


def readTargetSpectra(fn, verbose = False):
    """
    Reads target spectra from file.

    Input:
        fn              - Filepath of the target spectra file.
    
    Output:
        targetSpectra   - Scipy's interpolation function containing target spectra.
    """
    
    if (verbose):
        print(f"*Target spectra read from file '{fn}'")
    
    with open(fn,'r') as f:
        x = []
        y = []
        for line in f:
            txt = line.split()
            x.append(float(txt[0]))
            y.append(float(txt[1]))
    
    targetSpectra = interpolate.interp1d(x, y, 2)
    
    return targetSpectra

#END def readTargetSpectra


def readQuantumEfficiencyMono(qeName, order = 1, verbose = False, extraQECoef = None):
    """
    Reads the quantum efficiency of monochromatic detector.

    Input:
        qeName  - Path to the quantum efficiency datafile.

    Output:
        qe      - Scipy's interpolation function containing quantum efficiency data.
    """
    
    with open(qeName,'r') as f:
        x = []
        y = []
        for line in f:
            txt = line.split()
            x.append(float(txt[0]))
            y.append(float(txt[1]))
    
    if (extraQECoef is not None):
        y *= extraQECoef
    
    qe = interpolate.interp1d(x, y, order)
    
    if(verbose):
        print(f"*Quantum efficiency function read from file '{qeName}'")
        print(f"*Range is from {x[0]} to {x[-1]}, {len(x)} values.")
        plt.plot(x,y,'o',markerfacecolor='none')
        tx = np.arange(x[0],x[-1],(x[-1]-x[0])/100)
        ty = qe(tx)
        plt.plot(tx,ty,'-')
        plt.title("QE function points and interpolation")
    
    return qe

#END def readQuantumEfficiencyMono


def simulateNoiseless(rcData, qeFunc, wl, darkCurrent, integrationTime, 
        fullWellCapacity = None, darkBackgroundQ = None, darkBackgroundRC = None, verbose = False,
        bitDepth = None):
    """
    Simulates noiseless values for image. Image data is in radiance coefficients 
    with qeFunc giving the conversion into qe charges at detector. Noiseless image
    contains dark current contribution, as well as optionally constant dark background.
    
    Input:
        rcData              - Pixel grayscale data (radiance coefficient).
                              of the region of interest.
        qeFunc              - e⁻ charges at the detector.
        wl                  - Wavelength(s) of interest, in nm.
        darkCurrent         - Dark current on the dectector, in femptoamperes
        integrationTime     - Exposure time, in milliseconds
        fullWellCapacity    - Full-well capacity of the sensor, in e⁻.
        darkBackgroundQ     - Constant dark background level in e-, default: None.
                              Note - overrides darkBackgroundRC if not None
        darkBackgroundRC    - Constant dark background level in radiance coefficient, default: None
                              Note - because of radiance-to-e- conversion, this is dependent on integration time.
        verbose             - True or False.
        bitDepth            - The bit depth of the numbers for the signal. If given, assuming that fullWellCapacity 
                              charges will give max. number in bit depth presentation. Otherwise no conversion and
                              fullWellCapacity is in practise the max. number.
    
    Output:
        id                  - Noiseless (?) image data. 
    """
    
    if (verbose):
        print("Simulating noiseless image.")
    
    singleRc = not isinstance(rcData, (list, tuple, np.ndarray))
    singleWl = not isinstance(wl, (list, tuple, np.ndarray))
    single = singleRc and singleWl
    
    wlLocal = np.array([wl]).flatten()
    rcLocal = np.array([rcData]).flatten()
    qeLoc = qeFunc(wlLocal)
    
    # Dark constant background
    if darkBackgroundQ is None:
        if darkBackgroundRC is None:
            dcb = 0
        else:
            dcb = darkBackgroundRC * qeLoc * integrationTime * 10e-4
    else:
        dcb = darkBackgroundQ

    # Dark current in qe
    dc = darkCurrent * femtoAmpereToQE * integrationTime * 10e-4

    signal = np.array([rc * qeLoc * integrationTime * 10e-4 + dc + dcb for rc in rcLocal])

    # Image data
    if (fullWellCapacity is not None):
        if (np.any(signal > fullWellCapacity)):
            print("Warning, detector is saturated")
    id = np.clip(signal, 0, fullWellCapacity)
    if ((bitDepth is not None) and (fullWellCapacity is not None)):
        id *= bitDepth / fullWellCapacity
        if (verbose):
            print(f'Scaling {fullWellCapacity} to {bitDepth}.')
    
    if (single):
        id = id[0,0]
    elif (singleRc or singleWl):
        id = id.flatten()
    
    return id
    
#END simulateNoiseless


def simulateNoise(rcData, qeFunc, wl, readNoise, darkCurrent, integrationTime, 
        fullWellCapacity = None, darkBackgroundQ = None, darkBackgroundRC = None, verbose = False,
        bitDepth = None):
    """
    Simulates noise to image. Image data is in radiance coefficients 
    with qeFunc giving the conversion into qe charges at detector.

    Input:
        rcData              - Pixel grayscale data (radiance coefficient).
                              of the region of interest.
        qeFunc              - e⁻ charges at the detector.
        wl                  - Wavelength(s) of interest, in nm.
        readNoise           - Read noise on the sensor, in e⁻.
        darkCurrent         - Dark current on the dectector, in femptoamperes
        integrationTime     - Exposure time, in milliseconds
        fullWellCapacity    - Full-well capacity of the sensor, in e⁻. 
        darkBackgroundQ     - Constant dark background level in e-, default: None.
                              Note - overrides darkBackgroundRC if not None
        darkBackgroundRC    - Constant dark background level in radiance coefficient, default: None
                              Note - because of radiance-to-e- conversion, this is dependent on integration time.
        verbose             - True or False.
        bitDepth            - The bit depth of the numbers for the signal. If given, assuming that fullWellCapacity 
                              charges will give max. number in bit depth presentation. Otherwise no conversion and
                              fullWellCapacity is in practise the max. number.    
    Output:
        id                  - Noisy image data. 
    """
    
    if (verbose):
        print("Simulating noise to image.")
    
    singleRc = not isinstance(rcData, (list, tuple, np.ndarray))
    singleWl = not isinstance(wl, (list, tuple, np.ndarray))
    single = singleRc and singleWl
    
    wlLocal = np.array([wl]).flatten()
    rcLocal = np.array([rcData]).flatten()
    qeLoc = qeFunc(wlLocal)

    if (singleRc):
        n = 1
    else:
        n = len(rcData)
    if (singleWl):
        k = 1
    else:
        k = len(wl)

    signal = np.array([rc * qeLoc * integrationTime * 10e-4 for rc in rcLocal]).flatten()

    # Dark constant background
    if darkBackgroundQ is None:
        if darkBackgroundRC is None:
            dcb = 0
        else:
            dcb = darkBackgroundRC * qeLoc * integrationTime * 10e-4
    else:
        dcb = darkBackgroundQ
    
    # Add dark constant backgound to signal
    signal += dcb

    # Dark current in qe
    dc = darkCurrent * femtoAmpereToQE * integrationTime * 10e-4
    
    # Simulate Poisson dark current noise
    dnoise = np.random.poisson(dc, n*k)
    
    # Simulate Gaussian read noise
    rnoise = np.random.normal(0, readNoise, n*k)
        
    # Simulate Poisson photon noise, note that this includes the signal level and dark fixed level
    pnoise = [np.random.poisson(x) for x in signal]
    
    # Noisy image data
    allnoise = pnoise + dnoise + rnoise
    if (fullWellCapacity is not None):
        if (np.any(allnoise > fullWellCapacity)):
            print("Warning, detector is saturated")
    id = np.array(np.clip(allnoise, 0, fullWellCapacity))
    if ((bitDepth is not None) and (fullWellCapacity is not None)):
        id *= bitDepth / fullWellCapacity
        if (verbose):
            print(f'Scaling {fullWellCapacity} to {bitDepth}.')
    
    if (single):
        id = id[0]
    elif ((not singleRc) and (not singleWl)):
        id.shape = (n, k)
    
    return id
    
#END simulateNoise


def toQEMonoFunction(spectralTransmission, wlArray, QEData, verbose = False, QEOrder = 1, extraQECoef = None):
    """
    Converts spectral flux to photons with given QE of the detector.
    
    This method is for a monochromatic detector.

    Input:
        spectralTransmission    - Spectral transmission data.
        wlArray                 - Array with wavelengths of interest, in nm.
        QEData                  - Path to the quantum efficiency datafile.

    Output:
        ci                      - Scipy interpolation function containing qe charges
                                  at the detector.
    """
    
    if (verbose):
        print("Converting spectral flux into qe charges at the detector.")
    
    qe = readQuantumEfficiencyMono(QEData, order=QEOrder, verbose=verbose, extraQECoef = extraQECoef)
    if (verbose):
        print(f"*Detector quantum efficiency read from file '{QEData}'.")
        print(f"*Quantum efficiency data range is {vars(qe)['x'][[0,-1]]}.")
    
    # Limiting the computations to the joint range of wavelengths and qe wavelengths
    minWl = max(wlArray[0], vars(qe)['x'][0])
    maxWl = min(wlArray[-1], vars(qe)['x'][-1])
    wla = wlArray[np.logical_and(wlArray >= minWl, wlArray <= maxWl)]
    
    # Interpolation function for spectral transmission to cope with
    # possibly limited range of wavelengths.
    stf = interpolate.interp1d(wlArray, spectralTransmission, 1)
    
    c = [np.clip(qe(w), 0, None) * energyToPhotonsCoef * w * stf(w) for w in wla]
    
    ci = interpolate.interp1d(wla, c, QEOrder)
    
    return ci

#END toQEMonoFunction


def toRadiantFlux(au, FoV, apertureDiameter, detectorResolution, 
        opticsTransmission = 1.0, verbose = False):
    """
    Gives the conversion factor from radiance factor values into received radiant flux
    at the detector pixel in Watts.

    Input:
        au                  - Distance to Sun, in AUs.
        FoV                 - List with field of view, X and Y, in pixels.
        apertureDiameter    - Camera aperture diameter, in mm. 
        detectorResolution  - List with detector resolution, X and Y, in pixels.
        opticsTransmission  - Optics transmission coefficient? 
        verbose             - True of False, optional, defaults to False.
    """
    
    if (verbose):
        print("Computing conversion factor to radiant flux units at camera pixel for an radiance factor value of 1.")
    c = 1.0

    # To radiance
    c *= lambertRadianceAt1au / au**2
    if(verbose):
        print(f"*Radiance at {au} au is {c} W/m^2/sr")
    
    # To radiant flux
    x = np.tan(np.sqrt(FoV[0]**2 + FoV[1]**2)/2)**2
    x *= FoV[0] * apertureDiameter**2 * np.pi**2 * 1e-6
    x /= FoV[1] * (np.pi + FoV[0]**2 * np.pi / FoV[1]**2) * \
        detectorResolution[0] * detectorResolution[1]
    c *= x
    if(verbose):
        print(f"*Radiant flux received by camera pixel when the camera FoV is {FoV} and the detector resolution is {detectorResolution} is {c} W/sr.")
    
    # General optics transmission factor
    c *= opticsTransmission
    if (verbose):
        print(f"*After optics transmission factor of {opticsTransmission} the radiant flux is {c} W/sr.")
    
    return c

#END def toIrradiance


def toSpectralFlux(radiantFlux, wlStart, wlEnd, targetSpectra = None, verbose = False):
    """
    Spread the received radiant flux into spectral flux.

    Input:
        radiantFlux     -
        wlStart         - Start of the wavelength range, in nm.
        wlEnd           - End of the wavelength range, in nm.
        targetSpectra   - Output from readTargetSpectra function. Optional.
        verbose         - True of False, optional, defaults to False.

    Output:
        List containing:
            wla - Wavelength array, in nm.
            c   - Spectral flux?
    """
    
    global solarSpectra
    
    if (verbose):
        print("Computing the spectral distribution of radiant flux at detector pixel.")
    
    # If solar spectra is not yet read in, do it now.
    if(solarSpectra is None):
        solarSpectra = readSolarSpectra()

    # Target spectra
    if (targetSpectra is None):
        # Flat spectra of 1
        targetSpectra = interpolate.interp1d([0.0, 50000.0, 100000.0], [1.0, 1.0, 1.0], 0)
        if (verbose):
            print(f"*Constant target spectra assumed")
    
    if (wlStart < vars(targetSpectra)['x'][0]):
        print("Error: wavelength range goes below the target spectra")
        return
    if (wlEnd >  vars(targetSpectra)['x'][-1]):
        print("Error: wavelength range goes above the target spectra")
        return
    
    # Wavelength array, 1 nm steps
    wla = np.arange(wlStart, wlEnd+1)
    
    # Do spectral data
    c = solarSpectra(wla) * targetSpectra(wla)
    c *= radiantFlux
    
    if (verbose):
        print(f"*Flux distributed to {len(wla)} spectral channels from {wlStart} to {wlEnd}. Spectral flux at {wlStart} is {c[0]} W and at {wlEnd} it is {c[-1]} W")
    
    return [wla, c]


#END def toSpectralFlux


def toSpectralTransmission(spectralFlux, spectralFilterTransmission, 
        spectralFilterWidth, verbose = False):
    """
    Spectral flux after spectral transmission of the optics.
    
    At the moment, the only spectral transmission function option is the boxcar function.

    Input:
        spectralFlux                - Spectral flux, in ?
        spectralFilterTransmission  - Transmission factor through the window (?).
        spectralFilterWidth         - Width of the transmission window, in nm. Must be an integer.
        verbose                     - True of False, optional, defaults to False.
    """
    
    if (verbose):
        print("Computing the spectral flux after the spectral filter transmission function.")
    
    c = spectralFilterTransmission * np.convolve(spectralFlux, np.ones(spectralFilterWidth), 'same')

    return c
    
#END toSpectralTransmission



# Combos    ##################################################################


def eConversion(au, FoV, apertureD, detectorRes, opticsT, wlStart, wlEnd, 
        spectralTransmissionStrength, spectralTransmissionWindowWidth,
        #targetSpectra):
        targetSpectra, quantumEffData):
    
    verbose = False 


    convFact1 = toRadiantFlux(
            au = au, FoV = FoV, apertureDiameter = apertureD,
            detectorResolution = detectorRes, 
            opticsTransmission = opticsT, verbose = verbose)

    [wla, convFact20] = toSpectralFlux(convFact1, wlStart, wlEnd, 
            targetSpectra = None) 

    convFact30 = toSpectralTransmission(
        convFact20, spectralTransmissionStrength, spectralTransmissionWindowWidth) 

    flatConvFunc = toQEMonoFunction(convFact30, wla, quantumEffData)

    targetSpectraInterp = readTargetSpectra(targetSpectra, verbose = verbose)

    [wla, convFact2] = toSpectralFlux(convFact1, wlStart, wlEnd, 
            targetSpectra = targetSpectraInterp, verbose = verbose)

    convFact3 = toSpectralTransmission(convFact2, spectralTransmissionStrength, 
            spectralTransmissionWindowWidth, verbose = verbose)

    finalConvFunc = toQEMonoFunction(convFact3, wla, quantumEffData, verbose = verbose)

    #return finalConvFunc
    return flatConvFunc, finalConvFunc


#END def eConversion


def cameraSNRAnalysis(surfaceAlbedo, finalConvFunc, wlArray, albArr, itArr, readNoise, darkCurrent,
        integrationTime, fullWellCapacity, snrWl, verbose = False):

    snrSpec = computeSNR(surfaceAlbedo, finalConvFunc, wlArray, readNoise, darkCurrent, integrationTime, fullWellCapacity, verbose=verbose)

    #albArr = np.arange(0,0.21,0.01)

    snrAlb = computeSNR(albArr, finalConvFunc, snrWl, readNoise, darkCurrent, integrationTime, fullWellCapacity, verbose = verbose)

    #itArr = np.arange(0,101,1)

    snrIt = computeSNR(surfaceAlbedo, finalConvFunc, snrWl, readNoise, darkCurrent, itArr,
                      fullWellCapacity, verbose = verbose)

    return snrSpec, snrAlb, snrIt 

#END def cameraSNRAnalysis


def cameraSpectralRetrievalAnalysis(targetSpectra, wlArray, flatConvFunc, finalConvFunc, 
        surfaceAlbedo, wlSample, darkCurrent, integrationTime, readNoise, fullWellCapacity):

    # Extra:
    targetSpectraInterp = readTargetSpectra(targetSpectra, verbose = False)

    origSpectra = [targetSpectraInterp(w) for w in wlArray]

    flatFlux = simulateNoiseless(surfaceAlbedo, flatConvFunc, wlSample, darkCurrent, integrationTime)

    #finalConvFunc is the result of the e⁻ conversion.
    reflectedFlux = simulateNoise(surfaceAlbedo, finalConvFunc, wlSample, readNoise, darkCurrent, integrationTime, fullWellCapacity)

    noisySpectra = reflectedFlux/flatFlux

    return origSpectra, noisySpectra

#END def spectralRetrievalAnalysisCamera()


def imageSpectralRetrievalAnalysis(targetSpectra, wlArray, flatConvFunc, finalConvFunc,
        rcData, wlSample, darkCurrent, integrationTime, readNoise, fullWellCapacity):

    targetSpectraInterp = readTargetSpectra(targetSpectra, verbose = False)

    origSpectra = [targetSpectraInterp(w) for w in wlArray]

    flatFlux = simulateNoiseless(rcData, flatConvFunc, wlSample, darkCurrent, integrationTime)

    reflectedFlux = simulateNoise(rcData, finalConvFunc, wlSample, readNoise, darkCurrent, integrationTime, fullWellCapacity)

    noisySpectra = reflectedFlux/flatFlux

    aveSpectra = np.mean(noisySpectra, axis = 0)

    return origSpectra, aveSpectra

#END def imageSpectralRetrievalAnalysis


def imageSNRAnalysis(imageFilename, foldername, imageAlbedo, surfaceAlbedo, 
        finalConvFunc, snrWl, readNoise, darkCurrent, integrationTime, fullWellCapacity,
        verbose = False):

    [origIm, imH, maxRc] = importImageData(imageFilename, foldername, 
                        imageAlbedo=imageAlbedo, targetAlbedo=surfaceAlbedo, verbose=verbose)

    noisyIm = simulateNoise(origIm, finalConvFunc, snrWl, readNoise, darkCurrent, integrationTime, fullWellCapacity)

    imNormVal = finalConvFunc(snrWl) * 1.1*maxRc * integrationTime * 10e-4

    return noisyIm, imH, imNormVal

#END def imageSNRAnalysis


def imageToHypercube(imageFilename, foldername, imageAlbedo, surfaceAlbedo,
        finalConvFunc, wlSample, readNoise, darkCurrent, integrationTime, fullWellCapacity,
        outputFolder, outputFilename):

    [origIm, imH, maxRc] = importImageData(
        imagename = imageFilename, foldername = foldername, imageAlbedo = imageAlbedo, 
        targetAlbedo = surfaceAlbedo, verbose = True)

    hsData = simulateNoise(origIm, finalConvFunc, wlSample, readNoise, darkCurrent, 
        integrationTime, fullWellCapacity)

    exportHsPNG(hsData, imH, outputFilename, outputFolder)

#END def imageToHypercube

def fixedNoise(imageFilename, foldername):

    aspectWidth = 617
    aspectHeight = 489

    # First: Checking that all the images in question have ASPECT's size.

    # Check that files exist
    ifn = os.path.join(foldername, imageFilename)
    if (not os.path.isfile(ifn)):
        print("Error: image file '"+ifn+"' not found")
        return
    
    # Import image
    img = cv2.imread(ifn, cv2.IMREAD_UNCHANGED)
    
    # Check image dimensions
    shape = np.shape(img)
    imageWidth = print(shape[0])
    imageHeight = print(shape[1])

    # I'd like an and/or. Is an or enough?
    if imageWidth != aspectWidth or imageHeight != aspectHeight:
        print("Dimensions are not like ASPECT's")


#END def fixedNoise
