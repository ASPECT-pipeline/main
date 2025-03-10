import os
import numpy as np
import json
from astropy.io import fits

"""

This file is for converting a folder with binary
files and metadata into one FITS file that has 
all acquisitions of a single channel in a 3D-cube.

The function:
convertToFits(dirPath, outputFolder)
takes as parameters the path to the directory
containing the data. 
The folder at dirPath is expected to have the 
following structure:

dirPath/             
├── acq_000/                
│   ├── dc_1_exp_000.bin         
│   ├── dc_1_exp_001.bin         
│   └── ...                
│
└── meta/                   
    ├── calibration.json   
    ├── config.json         
    ├── hist_0.bin          
    ├── iqsOutput.json      
    └── telemetry.json     

The second parameter determines the folder 
where the fits file is created. 
Scroll down to see how to run.

"""

###################################################
#########        HELPER FUNCTIONS      ############
###################################################
# 
# These function are to help the convertToFits function
# to read the meta data and determine channels, 
# exposuretimes, and wavelenghts.

# Based on SP1 and channel determine the order.
# The sp value should be taken from the index 3
# to prevent miss identification.
def check_order(sp, channel):
    match channel:
        case 'VIS' | 'NIR1':
            if sp > 19000:
                return 'h'
            else:
                return 'l'
        case 'NIR2':
            if sp > 20000:
                return 'h'
            else:
                return 'l'
        case 'SWIR':
            return ''

#Extract the cahnnel from calib.json file
def read_channel(calibPath):
    with open(calibPath, 'r') as file:
        data = json.load(file)
        if data == None: # As the example SWIR files do not have config data
            return 'SWIR'
        firstKey = list(data.keys())[0]  # Access the first top-level key
        secondKey = list(data[firstKey].keys())[0]  # Access the first sub-key

        # Access the key indicating channel
        channel = list(data[firstKey][secondKey].keys())[0]

    return(channel)

#read the meta data from config file
def read_config(configPath, channel):
    with open(configPath, 'r') as file:
        data = json.load(file)

        #read SP values for each image
        match channel:
            case 'VIS':
                taskFile = data['visTaskFile']
            case 'NIR1':
                taskFile = data['nir1TaskFile']
            case 'NIR2':
                taskFile = data['nir2TaskFile']
            case 'SWIR':
                taskFile = data['swirTaskFile']
        
        #Extract sp values from taskValues
        taskValues = [taskFile[i:i + 8] for i in range(0, len(taskFile), 8)]
        sp1Values = [taskValues[i][1] for i in range(0, len(taskValues))]
        sp2Values = [taskValues[i][2] for i in range(0, len(taskValues))]
        sp3Values = [taskValues[i][3] for i in range(0, len(taskValues))]
        #Extract exposure times
        exposureTimes = [taskValues[i][4] for i in range(0, len(taskValues))]

        #Check the order based on SP1 index 3
        order = check_order(sp1Values[3], channel)

    return(order, exposureTimes, sp1Values, sp2Values, sp3Values)

###################################################
###### Reading binary and convering to FITS #######
###################################################

def convert_to_fits(dirPath, outputFolder):
    acquisitionPath = os.path.join(dirPath, "acq_000/") # path to acquisitions
    configPath = os.path.join(dirPath, "meta/config.json") # path to config file
    calibPath = os.path.join(dirPath, "meta/calib.json") # path to calib file

    #Extract channel
    channel = read_channel(calibPath)
    #Read config files and extract (order, exposuretimes[])
    config = read_config(configPath, channel)
    order = config[0] # l for low order, h for high order
    exposureTimes = config[1] # each element corresponds to the index in datacube
    sp1Values = config[2] # Piezo values of setpoint 1 that are used to calculate the wavelenghts
    sp2Values = config[3] # Piezo values of setpoint 2 that are used to calculate the wavelenghts
    sp3Values = config[4] # Piezo values of setpoint 3 that are used to calculate the wavelenghts

    #Create dimensions based on channel
    if channel == 'VIS':
        height = 1024
        width = 1024
    elif channel == 'NIR1' or channel == 'NIR2':
        height = 518
        width = 648
    
    #Read bin files from folder and short them in the right order
    binFiles = [f for f in os.listdir(acquisitionPath) if f.endswith('.bin')]
    binFiles.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    #List of HDU blocks
    #This will evetually containg the primary HDU and the Image HDU
    HDUs = []

    #Creating and empty primary HDU
    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header['EXTEND'] = True  # This indicates that there are extensions 
    
    HDUs.insert(0, primary_hdu)
    if channel == 'SWIR':
        #List that will contain the SWIR files
        swirList = []
        for i, binFile in enumerate(binFiles):
            filePath = os.path.join(acquisitionPath, binFile)
            with open(filePath, 'rb') as file:
                BinData = file.read()
                values = np.frombuffer(BinData, dtype=np.uint16)
                swirList.append(values)
        
        #Convert the swirList into NumPy Array
        swirArray = np.array(swirList, dtype=np.uint16)

        #Create FITS columns - one column for each file
        cols = []
        for i in range(swirArray.shape[1]):
            col_data = swirArray[:, i]
            col = fits.Column(name=f"SWIR_Frame_{i+1}", format="I", array=col_data) # I for 16-bit integers
            cols.append(col)

        #Create Binary Table HDU from the columns
        col_defs = fits.ColDefs(cols)
        hdu = fits.BinTableHDU.from_columns(col_defs)

    else: # For VIS and NIR channels create a 3D image cube

        #List that will contain the 2D images inside the block
        imageDataList = []
        #Read each binary data file and convert them into arrays of dimensions height x width.
        for i, binFile in enumerate(binFiles):
            filePath = os.path.join(acquisitionPath, binFile)
            with open(filePath, 'rb') as file:
                BinData = file.read()
                imgArray = np.frombuffer(BinData, dtype=np.uint16)
                imgArray = imgArray.reshape((height, width))
                imageDataList.append(imgArray)
        dataCube = np.stack(imageDataList, axis=0) # Stack the images into a cube

        #Create an image HDU
        hdu = fits.ImageHDU(dataCube)

    #Add metadata to the hdu header
    hdu.header['EXPOS'] = ','.join(map(str, exposureTimes)) #All exposure times as a string
    hdu.header['WAVELEN'] = '' #No wavelenghts at level 0
    hdu.header['CHANNEL'] = channel #Channel
    hdu.header['ORDER'] = order # Higher or lower order simply h or l
    hdu.header['PIEZO1'] = ','.join(map(str, sp1Values)) # set point 1 values
    hdu.header['PIEZO2'] = ','.join(map(str, sp2Values)) # set point 2 values
    hdu.header['PIEZO3'] = ','.join(map(str, sp3Values)) # set point 3 values

    HDUs.append(hdu)

    #create a folder for fits file
    os.makedirs(outputFolder, exist_ok=True)
    
    #Create a FITS file (in lack of better naming the file is called channle_order.fits)
    fileName = f'{channel}.fits' # Adjust to name the file 
    fitsFile = os.path.join(outputFolder, fileName)

    #Write the FITS file
    hdulist = fits.HDUList(HDUs)
    hdulist.writeto(fitsFile, overwrite=True)
    
    #return the path to the created fits file
    return(fitsFile)
