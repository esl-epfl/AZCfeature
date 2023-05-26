#!~/anaconda3/bin python

__authors__ = "Una Pale and Renato Zanetti"
__license__ = "LGPL"
__version__ = "1.0"
__email__ = "{una.pale, renato.zaneti} at epfl.ch"

'''script for:
    - converts edf files to csv files without any changes in data (for CHB-MIT) adding the labels
    - calculated features for each input file and saves them 
    - reorders feature data on a rolling basis (TSCV approach) and stores the a indexes vector to future train/test division
    - plots labels in time (before/after TSCV reordering)
'''

from parametersSetup import *
from ZeroCrossLibrary import *

###################################
# Initial setup
################################### 

Dataset='01_SWEC' #'01_SWEC', '02_CHB' #select the dataset to be processed
Server = 1 #0 - local execution, 1- server execution
parallelize=1 #chose whether to parallelize feature extraction per file per core
perc_cores= 0.5 #percetage of cores target to be charged with data processing

datasetPreparationType='Filt_1to20Hz'  # 'Raw', 'Filt_1to20Hz'

# DEFINING INPUT/OUTPUT FOLDERS
if Dataset=='01_SWEC':
    if Server:
        folderInData = '/shares/eslfiler1/databases/medical/swec-ethz/ieeg/long-term-dataset/'
        os.nice(20) #set low priority level for process running longer
    else:
        folderInData = '../iEEG/'
    patients  =['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18']

if Dataset=='02_CHB':
    if Server:
        folderInData = '/shares/eslfiler1/databases/medical/chb-mit/edf/' #expect all subject's files in the same directory
        os.nice(20) #set low priority level for process running longer
        samedir=1 #1- in case all chb files are in the same directory
    else:
        folderInData = '../chbmit/'
        samedir=0 #1- in case all chb files are in the same directory        
    patients  =['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24']

#folder that store partial results
if Server:
    folderResults = '/shares/eslfiler1/scratch/ZeroCrossFeatEpilepsy/'
else:
    folderResults = '../AZC_results/'

createFolderIfNotExists(folderResults)
createFolderIfNotExists( folderResults+Dataset)

#folder for feature files
folderOutFeatures= folderResults+Dataset+'/02_Features_' +datasetPreparationType + '/'
createFolderIfNotExists(folderOutFeatures)

###################################
# Parameters update regarding definition in 'parametersSetup.py'
GeneralParams.patients = patients

# ############################################################################################
#CHB-MIT edf files are converted to csv, adding extra column for labels
if Dataset=='02_CHB':
    folderOutCSV= folderResults+Dataset+'/01_DataCSV_raw/' 
    createFolderIfNotExists(folderOutCSV)
    # next function considers that all CHB-MIT edf files are in the same directory
    extractEDFdataToCSV_originalData(parallelize, perc_cores, samedir, folderInData, folderOutCSV, SigInfoParams, patients)

#########################################################################
# CALCULATE FEATURES FOR EACH DATA FILE
calculateCLFFeat = 1 #enable the calculation of each group of features
calculateAZCFeat = 1
if Dataset=='01_SWEC':
    calculateFeaturesPerEachFile_CLFAndAZCFeatures_SWEC(parallelize, perc_cores, ZeroCrossFeatureParams.EPS_thresh_arr,folderInData, folderOutFeatures, GeneralParams.patients,  SigInfoParams, ZeroCrossFeatureParams, calculateCLFFeat, calculateAZCFeat)

if Dataset=='02_CHB':
    calculateFeaturesPerEachFile_CLFAndAZCFeatures_CHB(parallelize, perc_cores, ZeroCrossFeatureParams.EPS_thresh_arr,folderOutCSV, folderOutFeatures, GeneralParams.patients,  SigInfoParams, ZeroCrossFeatureParams, calculateCLFFeat, calculateAZCFeat)

# # Plotting data labels before reorganizing files for TSCV approach
plotRawDataLabels(Dataset, folderOutFeatures, GeneralParams,SegSymbParams)

###########################################################################################
# Reorganizing feature files for TSCV (first file containing min 6h of data and a seizure)
windowSize=60*60 #target for files length in data
folderOutTSCV=folderResults+Dataset+'/03_Features_' +datasetPreparationType+'_TSCVprep_'+str(windowSize) +'s_DivBig/'
createFolderIfNotExists(folderOutTSCV)

#considering that CHB-MIT presents big time gaps in between some files for a subject and that some files have more than 1hour of data, we generate indexes
#for TSCV classification considering a first file with a minimum of 5hours and a seizure, trying to keep data grouped be origin file and only subdividing
#data of a file in case it's 1.5x bigger than the target 'windowSize'. Test data is always originated from a specific file, thus no data is merged from 
# multiple files.
generateTSCVindexesFromAllFiles_storeTSCVLabels(Dataset, GeneralParams, ZeroCrossFeatureParams, SegSymbParams, folderOutFeatures, folderOutTSCV, windowSize)

plotRearangedDataLabelsInTime(Dataset, folderOutTSCV,  GeneralParams,SegSymbParams)

