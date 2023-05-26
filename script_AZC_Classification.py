#!~/anaconda3/bin python

__authors__ = "Una Pale and Renato Zanetti"
__license__ = "LGPL"
__version__ = "1.0"
__email__ = "{una.pale, renato.zaneti} at epfl.ch"

''' script that performs seizure detection based on the TSCV approach'''

from VariousFunctionsLib import *
from parametersSetup import *
from PerformanceMetricsLib import *

import sys
import tracemalloc

###################################
# Initial setup
################################### 

Dataset='01_SWEC' #'01_SWEC', '02_CHB' #select the dataset to be processed
Server = 1 #0 - local execution, 1- server execution
bash = 1 #using bash to launch script execution: patient as input, hence we can parellelize in more cores
datasetPreparationType='Filt_1to20Hz'  # 'Raw', 'Filt_1to20Hz'

if bash: 
    if int(str(sys.argv[1]))<10:
        patients=['0'+str(sys.argv[1])]
    else:
        patients=[str(sys.argv[1])]

# DEFINING INPUT/OUTPUT FOLDERS
if Dataset=='01_SWEC':
    if Server:
        folderInData = '/shares/eslfiler1/databases/medical/swec-ethz/ieeg/long-term-dataset/'
        os.nice(20) #set low priority level for longer-running process
    else:
        folderInData = '../iEEG/'
    
    if bash==0: 
        patients  =['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18']

if Dataset=='02_CHB':
    if Server:
        folderInData = '/shares/eslfiler1/databases/medical/chb-mit/edf/' #expect all subject's files in the same directory
        os.nice(20) #set low priority level for longer-running process
    else:
        folderInData = '../chbmit/'
    
    if bash==0:
        patients  =['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24']

#folder that store partial results
if Server:
    folderResults = '/shares/eslfiler1/scratch/ZeroCrossFeatEpilepsy/'
else:
    folderResults = '../AZC_results/'

# ###########################################################################################
#SETUPS
GeneralParams.patients= patients
GeneralParams.plottingON=0
GeneralParams.PersGenApproach='personalized' #'personalized', 'generalized'
StandardMLParams.modelType = 'RandomForest'
FeaturesParams.numStandardFeat=62 #56 (Sopic2018) + 6 AZC

###############################
##  FEATURE SET
#features subset for evaluation
FeatSetNamesArray = ['StandardFeat'] #[ 'ZeroCross', 'StandardFeat']

# ###############################
# DEFINING INPUT/OUTPUT FOLDERS
windowSize=60*60 #files would have 1h data
folderOutTSCV=folderResults+Dataset+'/03_Features_' +datasetPreparationType+'_TSCVprep_'+str(windowSize) +'s_DivBig/'
folderOutFeatures= folderResults+Dataset+'/02_Features_' +datasetPreparationType + '/'
###############################################################################################################
# RUNNING FOR ALL COMBINATIONS OF FEATURES THAT ARE DEFINED ABOVE

featNamesAll=np.array(['meanAmpl', 'LineLength', 'samp_1_d7_1', 'samp_1_d6_1', 'samp_2_d7_1', 'samp_2_d6_1', 'perm_d7_3',
             'perm_d7_5', 'perm_d7_7', 'perm_d6_3', 'perm_d6_5', 'perm_d6_7', 'perm_d5_3', 'perm_d5_5', 'perm_d5_7',
             'perm_d4_3', 'perm_d4_5', 'perm_d4_7', 'perm_d3_3', 'perm_d3_5', 'perm_d3_7',
             'shannon_en_sig', 'renyi_en_sig', 'tsallis_en_sig', 'shannon_en_d7', 'renyi_en_d7', 'tsallis_en_d7',
             'shannon_en_d6', 'renyi_en_d6', 'tsallis_en_d6', 'shannon_en_d5', 'renyi_en_d5', 'tsallis_en_d5',
             'shannon_en_d4', 'renyi_en_d4', 'tsallis_en_d4', 'shannon_en_d3', 'renyi_en_d3', 'tsallis_en_d3',
             'p_dc_rel', 'p_mov_rel', 'p_delta_rel', 'p_theta_rel', 'p_alfa_rel', 'p_middle_rel', 'p_beta_rel',
             'p_gamma_rel', 'p_dc', 'p_mov', 'p_delta', 'p_theta', 'p_alfa', 'p_middle', 'p_beta', 'p_gamma', 'p_tot',
             'ZC_NoApprox', 'ZC_approx_16', 'ZC_approx_32', 'ZC_approx_64', 'ZC_approx_128', 'ZC_approx_256'])

for fs in range( len(FeatSetNamesArray)):
    FeatureSetName=FeatSetNamesArray[fs]

    if FeatureSetName=='All':
        inportCLF = 1
        inportAZC = 1
        featNames = featNamesAll
    elif FeatureSetName=='StandardFeat':
        inportCLF = 1
        inportAZC = 0
        featNames = np.array(['meanAmpl', 'LineLength', 'samp_1_d7_1', 'samp_1_d6_1', 'samp_2_d7_1', 'samp_2_d6_1', 'perm_d7_3',
             'perm_d7_5', 'perm_d7_7', 'perm_d6_3', 'perm_d6_5', 'perm_d6_7', 'perm_d5_3', 'perm_d5_5', 'perm_d5_7',
             'perm_d4_3', 'perm_d4_5', 'perm_d4_7', 'perm_d3_3', 'perm_d3_5', 'perm_d3_7',
             'shannon_en_sig', 'renyi_en_sig', 'tsallis_en_sig', 'shannon_en_d7', 'renyi_en_d7', 'tsallis_en_d7',
             'shannon_en_d6', 'renyi_en_d6', 'tsallis_en_d6', 'shannon_en_d5', 'renyi_en_d5', 'tsallis_en_d5',
             'shannon_en_d4', 'renyi_en_d4', 'tsallis_en_d4', 'shannon_en_d3', 'renyi_en_d3', 'tsallis_en_d3',
             'p_dc_rel', 'p_mov_rel', 'p_delta_rel', 'p_theta_rel', 'p_alfa_rel', 'p_middle_rel', 'p_beta_rel',
             'p_gamma_rel', 'p_dc', 'p_mov', 'p_delta', 'p_theta', 'p_alfa', 'p_middle', 'p_beta', 'p_gamma', 'p_tot'])

    elif FeatureSetName == 'ZeroCross':
        inportCLF = 0
        inportAZC = 1
        featNames = np.array(['ZC_NoApprox', 'ZC_approx_16', 'ZC_approx_32', 'ZC_approx_64', 'ZC_approx_128', 'ZC_approx_256'])
    else:
        inportCLF = 1
        inportAZC = 1
        featNames = np.array(
            ['meanAmpl', 'LineLength', 'samp_1_d7_1', 'samp_1_d6_1', 'samp_2_d7_1', 'samp_2_d6_1', 'perm_d7_3',
             'perm_d7_5', 'perm_d7_7', 'perm_d6_3', 'perm_d6_5', 'perm_d6_7', 'perm_d5_3', 'perm_d5_5', 'perm_d5_7',
             'perm_d4_3', 'perm_d4_5', 'perm_d4_7', 'perm_d3_3', 'perm_d3_5', 'perm_d3_7',
             'shannon_en_sig', 'renyi_en_sig', 'tsallis_en_sig', 'shannon_en_d7', 'renyi_en_d7', 'tsallis_en_d7',
             'shannon_en_d6', 'renyi_en_d6', 'tsallis_en_d6', 'shannon_en_d5', 'renyi_en_d5', 'tsallis_en_d5',
             'shannon_en_d4', 'renyi_en_d4', 'tsallis_en_d4', 'shannon_en_d3', 'renyi_en_d3', 'tsallis_en_d3',
             'p_dc_rel', 'p_mov_rel', 'p_delta_rel', 'p_theta_rel', 'p_alfa_rel', 'p_middle_rel', 'p_beta_rel',
             'p_gamma_rel', 'p_dc', 'p_mov', 'p_delta', 'p_theta', 'p_alfa', 'p_middle', 'p_beta', 'p_gamma', 'p_tot',
             'ZC_NoApprox', 'ZC_approx_16', 'ZC_approx_32', 'ZC_approx_64', 'ZC_approx_128', 'ZC_approx_256'])


    #creating output folder
    folderOut = folderResults + Dataset + '/05_RF_TSCV/'
    createFolderIfNotExists(folderOut)
    folderOut = folderOut + datasetPreparationType +FeatureSetName +'/'
    createFolderIfNotExists(folderOut)

    ############################################################################################
    seizureStableLenToTestIndx = int(GeneralParams.seizureStableLenToTest / SegSymbParams.slidWindStepSec)
    seizureStablePercToTest = GeneralParams.seizureStablePercToTest
    distanceBetweenSeizuresIndx = int(GeneralParams.distanceBetween2Seizures / SegSymbParams.slidWindStepSec)
    numLabelsPerHour = 60 * 60 / SegSymbParams.slidWindStepSec
    toleranceFP_bef = int(GeneralParams.toleranceFP_befSeiz / SegSymbParams.slidWindStepSec)
    toleranceFP_aft = int(GeneralParams.toleranceFP_aftSeiz / SegSymbParams.slidWindStepSec)
    tracemalloc.start()
    for patIndx, pat in enumerate(GeneralParams.patients):
        print('Patient:', pat)
        #load all files only once and mark where each file starts

        if Dataset=='01_SWEC':
            filesAll=np.sort(glob.glob(folderOutFeatures + 'ID' + pat + '/ID' + pat + '*Labels.csv.gz'))
            labelsAll = concatenateDataFromFiles(filesAll).reshape(-1)
            nrows = np.size(labelsAll)

            if inportCLF:
                filesAll=np.sort(glob.glob(folderOutFeatures + 'ID' + pat + '/ID' + pat + '*OtherFeat.csv.gz'))
                OtherFeat = concatenateDataFromFiles_AllocateFirst(filesAll, nrows)

            if inportAZC:
                filesAll=np.sort(glob.glob(folderOutFeatures + 'ID' + pat + '/ID' + pat + '*ZCFeat.csv.gz'))
                ZCFeat = concatenateDataFromFiles_AllocateFirst(filesAll, nrows)

            fileNameIndexes = folderOutTSCV + 'ID' + pat + '/ID' + pat + '_TSCV_indexes.csv'

        if Dataset=='02_CHB':
            filesAll = np.sort(glob.glob(folderOutFeatures +'chb' + pat+ '/chb' + pat + '*Labels.csv.gz'))            
            #chb02_16+ commes first than chb02_16 when sorting files
            if pat=='02' and '+' in filesAll[15][:]: #chb02_16+
                cpy = filesAll[15]
                filesAll[15] = filesAll[16]
                filesAll[16] = cpy

            labelsAll = concatenateDataFromFiles(filesAll).reshape(-1)
            nrows = np.size(labelsAll)
            if inportCLF:
                filesAll = np.sort(glob.glob(folderOutFeatures +'chb' + pat+ '/chb' + pat + '*OtherFeat.csv.gz'))
                if pat=='02' and '+' in filesAll[15][:]: #chb02_16+
                    cpy = filesAll[15]
                    filesAll[15] = filesAll[16]
                    filesAll[16] = cpy
                OtherFeat = concatenateDataFromFiles_AllocateFirst(filesAll, nrows)

            if inportAZC:
                filesAll = np.sort(glob.glob(folderOutFeatures +'chb' + pat+ '/chb' + pat + '*ZCFeat.csv.gz'))
                if pat=='02' and '+' in filesAll[15][:]: #chb02_16+
                    cpy = filesAll[15]
                    filesAll[15] = filesAll[16]
                    filesAll[16] = cpy
                ZCFeat = concatenateDataFromFiles_AllocateFirst(filesAll, nrows)

            fileNameIndexes = folderOutTSCV +'chb' + pat+ '/chb' + pat + '_TSCV_indexes.csv'
        
        folderOutPat = folderOut + pat + '/'
        createFolderIfNotExists(folderOutPat)
        #select the data to be used
        startIndxOfFiles = np.loadtxt(fileNameIndexes).astype(int) #startIndxOfFiles holde TSCV files indexes considering stacking all subject's data in the same matrix, 

        if inportAZC and inportCLF: #in case we use both features
            numCh=int(len(ZCFeat[0, :])/(len(ZeroCrossFeatureParams.EPS_thresh_arr)+1))
            dataAll= mergeFeatFromTwoMatrixes(OtherFeat, ZCFeat, numCh) 
            del OtherFeat #as all data is merged into another matrix, we can free previous allocated memory
            del ZCFeat
        elif inportAZC:
            dataAll = ZCFeat
        else:
            dataAll = OtherFeat

        n_cores  = mp.cpu_count()
        StandardMLParams.n_jobs = int(n_cores/24) #let's target half of the available cores, considering all the patients will be processed at once

        if StandardMLParams.n_jobs < 2:
            StandardMLParams.n_jobs=2 #at least 2 cores allocated for training 

        # remove nan and inf from matrix
        dataAll[np.where(np.isinf(dataAll))] = np.nan
        col_mean = np.nanmean(dataAll, axis=0)
        inds = np.where(np.isnan(dataAll))
        dataAll[inds] = np.take(col_mean, inds[1])
        # if still somewhere nan replace with 0
        dataAll[np.where(np.isnan(dataAll))] = 0


        current, peak = tracemalloc.get_traced_memory()
        print('peak memory (GB): ' + str(peak/1024**3))

        numCV = len(startIndxOfFiles) - 1 #first position is the begining of the testing data and last position is the length of total input data
        performanceTrain = np.zeros((numCV, 4*9))  # 3 for noSmooth, step1, step2, and 9 or 9 perf meausres
        performanceTest = np.zeros((numCV, 4*9))  # 3 for noSmooth, step1, step2, and 9 or 9 perf meausres
        performanceTest_justNext = np.zeros(( numCV-1, 4*9))   # 3 for noSmooth, step1, step2, and 9 or 9 perf meausres

        for cv in range(numCV):
            print('Pat', pat, 'CV', cv+1, 'Out of', numCV, 'CVs')

            # create train and test data
            dataTest = dataAll[startIndxOfFiles[cv]:startIndxOfFiles[cv + 1],  :]  # test data comes from only one file after this CV
            label_test = labelsAll[startIndxOfFiles[cv]:startIndxOfFiles[cv + 1]]
            dataTrain = dataAll[0:startIndxOfFiles[cv], :]
            label_train = labelsAll[0:startIndxOfFiles[cv]]
            dataSource_test = (cv + 1) * np.ones( (startIndxOfFiles[cv + 1] - startIndxOfFiles[cv]))  # value 1 means file after the train one

            # initializing model and then training
            MLmodel = train_StandardML_moreModelsPossible(dataTrain, label_train, StandardMLParams)

            #testing
            (predLabels_test, probabLab_test, acc_test, accPerClass_test)= test_StandardML_moreModelsPossible_v3(dataTest,label_test,  MLmodel)

            # save predictions
            fileName2='Subj' + pat + '_cv'+str(cv).zfill(3)
            dataToSave = np.vstack((label_test, probabLab_test, predLabels_test, dataSource_test)).transpose()  # added from which file is specific part of test set
            outputName = folderOutPat + fileName2 + '_TestPredictions.csv'
            saveDataToFile(dataToSave, outputName, 'gzip')



