#!~/anaconda3/bin python

__author__ = "Una Pale"
__credits__ = ["Una Pale", "Renato Zanetti"]
__license__ = "LGPL"
__version__ = "1.0"
__email__ = "una.pale at epfl.ch"

import os
import glob
import csv
import math
# import sklearn
import multiprocessing as mp
import time
from math import ceil
# from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pywt
import scipy
import sys
import pyedflib
import MITAnnotation as MIT
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
import scipy.io
# import random
from PerformanceMetricsLib import *
from scipy import signal
# from scipy import interpolate
import pandas as pd
# import seaborn as sns
# import wfdb
from antropy import *


########################################################
#global variables for multi-core operation
number_free_cores = 0
n_cores_semaphore = 1



def createFolderIfNotExists(folderOut):
    ''' creates folder if doesnt already exist
    warns if creation failed '''
    if not os.path.exists(folderOut):
        try:
            os.mkdir(folderOut)
        except OSError:
            print("Creation of the directory %s failed" % folderOut)
        # else:
        #     print("Successfully created the directory %s " % folderOut)


def calculateOtherMLfeatures_oneCh(X, ZeroCrossParams):
    numFeat = 56 #54 from Sopic2018 and LL and meanAmpl
    lenSig= len(X)
    segLenIndx = int(ZeroCrossParams.winLen * ZeroCrossParams.samplFreq)  # length of EEG segments in samples
    slidWindStepIndx = int( ZeroCrossParams.winStep * ZeroCrossParams.samplFreq)  # step of slidin window to extract segments in samples
    index = np.arange(0, lenSig - segLenIndx, slidWindStepIndx).astype(int)

    featureValues=np.zeros((len(index), numFeat))
    for i in range(len(index)):
        sig = X[index[i]:index[i] + segLenIndx]
        feat54 = calculateMLfeatures_oneDataWindow_54(sig, ZeroCrossParams.samplFreq)
        meanAmpl = np.mean(np.abs(sig))
        LL = np.mean(np.abs(np.diff(sig)))
        featureValues[i, :] = np.hstack((meanAmpl, LL, feat54))
    return (featureValues)


def calculateMLfeatures_oneDataWindow_54(data,  samplFreq):
    ''' function that calculates various features relevant for epileptic seizure detection
    from paper: D. Sopic, A. Aminifar, and D. Atienza, e-Glass: A Wearable System for Real-Time Detection of Epileptic Seizures, 2018
    but uses only features that are (can be) normalized
    _54 means that return also absolute values of power bands
    '''
    #some parameters
    DWTfilterName = 'db4'  # 'sym5'
    DWTlevel = 7
    n1 = 2  #num dimensions for sample entropy
    r1 = 0.2 # num of STD for sample entropy
    r2 = 0.35 # num of STD for sample entropy
    a = 2 # param for shannon, renyi and tsallis enropy
    q = 2 # param for shannon, renyi and tsallis enropy

    #DWT
    coeffs = pywt.wavedec(data, DWTfilterName, level=DWTlevel)
    a7, d7, d6, d5, d4, d3, d2, d1= coeffs

    # #sample entropy (Obs: the function sampen2 takes too long for executing, hence it was substituted with sample_entropy)
    # samp_1_d7_1 = sampen2(n1, r1 * np.std(d7), d7)
    # samp_1_d6_1 = sampen2(n1, r1 * np.std(d6), d6)
    # samp_2_d7_1 = sampen2(n1, r2 * np.std(d7), d7)
    # samp_2_d6_1 = sampen2(n1, r2 * np.std(d6), d6)

    #sample entropy: only allows r=0.2, does we zeroed samp_2_d7_1 an samp_2_d6_1 (after first tests)
    samp_1_d7_1 = sample_entropy(d7)
    samp_1_d6_1 = sample_entropy(d6)
    samp_2_d7_1 = 0 #sample entropy with r=0.35 was discarded for the use of 'antropy' library
    samp_2_d6_1 = 0

    #permutation entropy
    perm_d7_3 = perm_entropy(d7, order=3, delay=1, normalize=True)  # normalize=True instead of false as in paper
    perm_d7_5 = perm_entropy(d7, order=5, delay=1, normalize=True)
    perm_d7_7 = perm_entropy(d7, order=7, delay=1, normalize=True)
    perm_d6_3 = perm_entropy(d6, order=3, delay=1, normalize=True)
    perm_d6_5 = perm_entropy(d6, order=5, delay=1, normalize=True)
    perm_d6_7 = perm_entropy(d6, order=7, delay=1, normalize=True)
    perm_d5_3 = perm_entropy(d5, order=3, delay=1, normalize=True)
    perm_d5_5 = perm_entropy(d5, order=5, delay=1, normalize=True)
    perm_d5_7 = perm_entropy(d5, order=7, delay=1, normalize=True)
    perm_d4_3 = perm_entropy(d4, order=3, delay=1, normalize=True)
    perm_d4_5 = perm_entropy(d4, order=5, delay=1, normalize=True)
    perm_d4_7 = perm_entropy(d4, order=7, delay=1, normalize=True)
    perm_d3_3 = perm_entropy(d3, order=3, delay=1, normalize=True)
    perm_d3_5 = perm_entropy(d3, order=5, delay=1, normalize=True)
    perm_d3_7 = perm_entropy(d3, order=7, delay=1, normalize=True)

    #shannon renyi and tsallis entropy
    (shannon_en_sig, renyi_en_sig, tsallis_en_sig) = sh_ren_ts_entropy(data, a, q)
    (shannon_en_d7, renyi_en_d7, tsallis_en_d7)  = sh_ren_ts_entropy(d7, a, q)
    (shannon_en_d6, renyi_en_d6, tsallis_en_d6)  = sh_ren_ts_entropy(d6, a, q)
    (shannon_en_d5, renyi_en_d5, tsallis_en_d5)  = sh_ren_ts_entropy(d5, a, q)
    (shannon_en_d4, renyi_en_d4, tsallis_en_d4)  = sh_ren_ts_entropy(d4, a, q)
    (shannon_en_d3, renyi_en_d3, tsallis_en_d3)  = sh_ren_ts_entropy(d3, a, q)

    #band power
    p_tot = bandpower(data, samplFreq, 0,  45)
    p_dc = bandpower(data, samplFreq, 0, 0.5)
    p_mov = bandpower(data, samplFreq, 0.1, 0.5)
    p_delta = bandpower(data, samplFreq, 0.5, 4)
    p_theta = bandpower(data, samplFreq, 4, 8)
    p_alfa = bandpower(data, samplFreq, 8, 13)
    p_middle = bandpower(data, samplFreq, 12, 13)
    p_beta = bandpower(data, samplFreq, 13, 30)
    p_gamma = bandpower(data, samplFreq, 30, 45)
    p_dc_rel = p_dc / p_tot
    p_mov_rel = p_mov / p_tot
    p_delta_rel = p_delta / p_tot
    p_theta_rel = p_theta / p_tot
    p_alfa_rel = p_alfa / p_tot
    p_middle_rel = p_middle / p_tot
    p_beta_rel = p_beta / p_tot
    p_gamma_rel = p_gamma / p_tot

    featuresAll= [samp_1_d7_1, samp_1_d6_1, samp_2_d7_1, samp_2_d6_1, perm_d7_3, perm_d7_5, perm_d7_7, perm_d6_3, perm_d6_5, perm_d6_7,   perm_d5_3, perm_d5_5, \
             perm_d5_7, perm_d4_3, perm_d4_5, perm_d4_7, perm_d3_3, perm_d3_5, perm_d3_7, shannon_en_sig, renyi_en_sig, tsallis_en_sig, shannon_en_d7, renyi_en_d7, tsallis_en_d7, \
             shannon_en_d6, renyi_en_d6, tsallis_en_d6, shannon_en_d5, renyi_en_d5, tsallis_en_d5, shannon_en_d4, renyi_en_d4, tsallis_en_d4, shannon_en_d3, renyi_en_d3, tsallis_en_d3, \
             p_dc_rel, p_mov_rel, p_delta_rel, p_theta_rel, p_alfa_rel, p_middle_rel, p_beta_rel, p_gamma_rel,
             p_dc, p_mov, p_delta, p_theta, p_alfa, p_middle, p_beta, p_gamma, p_tot]
    return (featuresAll)


def sh_ren_ts_entropy(x, a, q):
    ''' function that calculates three different entropy meausres from given widow:
    shannon, renyi and tsallis entropy'''
    p, bin_edges = np.histogram(x)
    p = p/ np.sum(p)
    p=p[np.where(p >0)] # to exclude log(0)
    shannon_en = - np.sum(p* np.log2(p))
    renyi_en = np.log2(np.sum(pow(p,a))) / (1 - a)
    tsallis_en = (1 - np.sum(pow(p,q))) / (q - 1)
    return (shannon_en, renyi_en, tsallis_en)

def bandpower(x, fs, fmin, fmax):
    '''function that calculates energy of specific frequency band of FFT spectrum'''
    f, Pxx = scipy.signal.periodogram(x, fs=fs)
    ind_min = scipy.argmax(f > fmin) - 1
    ind_max = scipy.argmax(f >fmax) - 1
    return scipy.trapz(Pxx[ind_min: ind_max+1], f[ind_min: ind_max+1])

def sampen2(dim,r,data):
    ''' function that calculates sample entropy from given window of data'''
    epsilon = 0.001
    N = len(data)
    correl = np.zeros( 2)
    dataMat = np.zeros((dim + 1, N - dim))
    for i in range(dim+1):
        dataMat[i,:]= data[i: N - dim + i]

    for m in range(dim,dim + 2):
        count = np.zeros( N - dim)
        tempMat = dataMat[0:m,:]

        for i in range(N - m):
            #calculate distance, excluding self - matching case
            dist = np.max(np.abs(tempMat[:, i + 1: N - dim] - np.tile(tempMat[:, i],( (N - dim - i-1),1)).T  ), axis=0)
            D = (dist < r)
            count[i] = np.sum(D) / (N - dim - 1)

        correl[m - dim] = np.sum(count) / (N - dim)

    saen = np.log((correl[0] + epsilon) / (correl[1] + epsilon))
    return saen

def readEdfFile (fileName):
    ''' reads .edf file and returnes  data[numSamples, numCh], sampling frequency, names of channels'''
    f = pyedflib.EdfReader(fileName)
    n = f.signals_in_file
    channelNames = f.getSignalLabels()
    f.getSampleFrequency(0)
    samplFreq= data = np.zeros(( f.getNSamples()[0], n))
    for i in np.arange(n):
        data[:, i] = f.readSignal(i)
    return (data, samplFreq, channelNames)

def writeToCsvFile( data, labels,  fileName):
    outputName= fileName+'.csv'
    myFile = open(outputName, 'w',newline='')
    dataToWrite=np.column_stack((data, labels))
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(dataToWrite)

def writeToCompressedCsvFile( data, labels,  fileName):
    outputName= fileName+'.csv.gz'
    dataToWrite=np.column_stack((data, labels))
    df = pd.DataFrame(data=dataToWrite)
    df.to_csv(outputName, index=False, compression='gzip') 


def extractEDFdataToCSV_originalData(parallelize, perc_cores, samedir, folderIn, folderOut, SigInfoParams, patients):
    ''' converts data from edf format to csv
    20210705 UnaPale'''
    global number_free_cores
    print('Extracting .csv from CHB edf files')
    if parallelize:
        n_cores  = mp.cpu_count()
        n_cores = ceil(n_cores*perc_cores)

        if n_cores > len(patients):
            n_cores = len(patients)

        print('Number of used cores: ' + str(n_cores))

        pool = mp.Pool(n_cores)
        number_free_cores = n_cores
        
    # cutting segments
    for patIndx, pat in enumerate(patients):
        # print('-- Patient:', pat)
        PATIENT = pat if len(sys.argv) < 2 else '{0:02d}'.format(int(sys.argv[1]))
        #number of Seiz and nonSeiz files
        if samedir:
            SeizFiles=sorted(glob.glob(f'{folderIn}chb{PATIENT}*.seizures'))
            EDFNonSeizFiles=sorted(glob.glob(f'{folderIn}chb{PATIENT}*.edf'))
        else:
            SeizFiles=sorted(glob.glob(f'{folderIn}chb{PATIENT}/chb{PATIENT}*.seizures'))
            EDFNonSeizFiles=sorted(glob.glob(f'{folderIn}chb{PATIENT}/chb{PATIENT}*.edf'))

        folderOutFiles = folderOut + 'chb' + pat + '/'
        createFolderIfNotExists(folderOutFiles)
        print('Patient: ' + pat + '  Number of files: ' + str(len(EDFNonSeizFiles)))
        # create lists with just names, to be able to compare them
        SeizFileNames = list()
        for fIndx, f in enumerate(SeizFiles):
            justName = os.path.split(f)[1][:-13]
            if (fIndx == 0):
                SeizFileNames = [justName]
            else:
                SeizFileNames.append(justName)
        NonSeizFileNames = list()
        NonSeizFileFullNames = list()
        for fIndx, f in enumerate(EDFNonSeizFiles):
            justName = os.path.split(f)[1][:-4]
            if (justName not in SeizFileNames):
                if (fIndx == 0):
                    NonSeizFileNames = [justName]
                    NonSeizFileFullNames = [f]
                else:
                    NonSeizFileNames.append(justName)
                    NonSeizFileFullNames.append(f)

        if parallelize:
                pool.apply_async(extractDataLabels_CHB, args=(patIndx, SeizFiles, NonSeizFileFullNames, SigInfoParams, folderOutFiles), callback=collect_result) 
                number_free_cores = number_free_cores -1
                if number_free_cores==0:
                    while number_free_cores==0: #synced in the callback
                        time.sleep(0.1)
                        pass
        else:
            extractDataLabels_CHB(patIndx, SeizFiles, NonSeizFileFullNames, SigInfoParams, folderOutFiles)

    if parallelize:
        while number_free_cores < n_cores: #wait till all subjects have their data processed
            time.sleep(0.1)
            pass
        pool.close()
        pool.join()  

def extractDataLabels_CHB(patIndx, SeizFiles, NonSeizFileFullNames, SigInfoParams, folderOut):

    #EXPORT SEIZURE FILES
    for fileIndx,fileName in enumerate(SeizFiles):
        allGood=1

        fileName0 = os.path.splitext(fileName)[0]  # removing .seizures from the string
        # here replaced reading .hea files with .edf reading to avoid converting !!!
        (rec, samplFreq, channels) = readEdfFile(fileName0)
        # take only the channels we need and in correct order
        try:
            chToKeepAndInCorrectOrder=[channels.index(SigInfoParams.channels[i]) for i in range(len(SigInfoParams.channels))]
        except:
            print('Sth wrong with the channels in a file: ', fileName)
            allGood=0

        if (allGood==1):
            newData = rec[1:, chToKeepAndInCorrectOrder]
            (lenSig, numCh) = newData.shape
            newLabel = np.zeros(lenSig)
            # read times of seizures
            szStart = [a for a in MIT.read_annotations(fileName) if a.code == 32]  # start marked with '[' (32)
            szStop = [a for a in MIT.read_annotations(fileName) if a.code == 33]  # start marked with ']' (33)
            # for each seizure cut it out and save (with few parameters)
            numSeizures = len(szStart)
            for i in range(numSeizures):
                seizureLen = szStop[i].time - szStart[i].time
                newLabel[int(szStart[i].time):int(szStop[i].time)] = np.ones(seizureLen)

            # saving to csv file - saving all seizures to one file,with name of how many seizures is there
            pom, fileName1 = os.path.split(fileName0)
            fileName2 = os.path.splitext(fileName1)[0]

            fileName3 = folderOut +  fileName2 

            writeToCompressedCsvFile(newData, newLabel, fileName3)

    #EXPORT NON SEIZURE FILES
    for fileIndx,fileName in enumerate(NonSeizFileFullNames):
        allGood=1

        # # here replaced reading .hea files with .edf reading to avoid converting !!!
        (rec, samplFreq, channels) = readEdfFile(fileName)
        # take only the channels we need and in correct order
        try:
            chToKeepAndInCorrectOrder=[channels.index(SigInfoParams.channels[i]) for i in range(len(SigInfoParams.channels))]
        except:
            print('Sth wrong with the channels in a file: ', fileName)
            allGood=0

        if (allGood==1):
            newData = rec[1:, chToKeepAndInCorrectOrder]
            (lenSig, numCh) = newData.shape
            newLabel = np.zeros(lenSig)

            # saving to csv file
            pom, fileName1 = os.path.split(fileName)
            fileName2 = os.path.splitext(fileName1)[0]
            fileName3 = folderOut + fileName2
            writeToCompressedCsvFile(newData, newLabel, fileName3)

    return (patIndx)



def train_StandardML_moreModelsPossible(X_train, y_train,  StandardMLParams):
        #X_train0= np.float32(X_train)
    #y_train = np.float32(y_train)
    #y_train = y_train.reshape((len(y_train),))
    X_train0=X_train
    #replacing nan and inf values
    #X_train0[np.where(np.isinf(X_train0))] = np.nan

    if (np.size(X_train0,0)==0):
            print('X train size 0 is 0!!', X_train0.shape, y_train.shape)
    if (np.size(X_train0,1)==0):
            print('X train size 1 is 0!!', X_train0.shape, y_train.shape)
    col_mean = np.nanmean(X_train0, axis=0)
    inds = np.where(np.isnan(X_train0))
    X_train0[inds] = np.take(col_mean, inds[1])
    # if still somewhere nan replace with 0
    X_train0[np.where(np.isnan(X_train0))] = 0
    X_train=X_train0

    #MLmodels.modelType = 'KNN'
    if (StandardMLParams.modelType=='KNN'):
        model = KNeighborsClassifier(n_neighbors=StandardMLParams.KNN_n_neighbors, metric=StandardMLParams.KNN_metric)
        model.fit(X_train, y_train)
    elif (StandardMLParams.modelType=='SVM'):
        #model = svm.SVC(kernel=StandardMLParams.SVM_kernel, C=StandardMLParams.SVM_C, gamma=StandardMLParams.SVM_gamma)
        model = svm.SVC(kernel=StandardMLParams.SVM_kernel, C=StandardMLParams.SVM_C, gamma=StandardMLParams.SVM_gamma, probability=True) # when using bayes smoothing
        model.fit(X_train, y_train)
    elif (StandardMLParams.modelType=='DecisionTree'):
        if (StandardMLParams.DecisionTree_max_depth==0):
            model = DecisionTreeClassifier(random_state=0, criterion=StandardMLParams.DecisionTree_criterion, splitter=StandardMLParams.DecisionTree_splitter)
        else:
            model = DecisionTreeClassifier(random_state=0, criterion=StandardMLParams.DecisionTree_criterion, splitter=StandardMLParams.DecisionTree_splitter,  max_depth=StandardMLParams.DecisionTree_max_depth)
        model = model.fit(X_train, y_train)
    elif (StandardMLParams.modelType=='RandomForest'):
        if (StandardMLParams.DecisionTree_max_depth == 0):
            model = RandomForestClassifier(random_state=0, n_estimators=StandardMLParams.RandomForest_n_estimators, criterion=StandardMLParams.DecisionTree_criterion, n_jobs=StandardMLParams.n_jobs) # min_samples_leaf=10
        else:
            model = RandomForestClassifier(random_state=0, n_estimators=StandardMLParams.RandomForest_n_estimators, criterion=StandardMLParams.DecisionTree_criterion, n_jobs=StandardMLParams.n_jobs,  max_depth=StandardMLParams.DecisionTree_max_depth ) # min_samples_leaf=10
        model = model.fit(X_train, y_train)
    elif (StandardMLParams.modelType=='BaggingClassifier'):
        if (StandardMLParams.Bagging_base_estimator=='SVM'):
            model = BaggingClassifier(base_estimator=svm.SVC(kernel=StandardMLParams.SVM_kernel, C=StandardMLParams.SVM_C, gamma=StandardMLParams.SVM_gamma), n_estimators=StandardMLParams.Bagging_n_estimators,random_state=0)
        elif  (StandardMLParams.Bagging_base_estimator=='KNN'):
            model = BaggingClassifier(base_estimator= KNeighborsClassifier(n_neighbors=StandardMLParams.KNN_n_neighbors, metric=StandardMLParams.KNN_metric), n_estimators=StandardMLParams.Bagging_n_estimators,random_state=0)
        elif (StandardMLParams.Bagging_base_estimator == 'DecisionTree'):
            model = BaggingClassifier(DecisionTreeClassifier(random_state=0, criterion=StandardMLParams.DecisionTree_criterion, splitter=StandardMLParams.DecisionTree_splitter),
                n_estimators=StandardMLParams.Bagging_n_estimators, random_state=0)
        model = model.fit(X_train, y_train)
    elif (StandardMLParams.modelType=='AdaBoost'):
        if (StandardMLParams.Bagging_base_estimator=='SVM'):
            model = AdaBoostClassifier(base_estimator=svm.SVC(kernel=StandardMLParams.SVM_kernel, C=StandardMLParams.SVM_C, gamma=StandardMLParams.SVM_gamma), n_estimators=StandardMLParams.Bagging_n_estimators,random_state=0)
        elif  (StandardMLParams.Bagging_base_estimator=='KNN'):
            model = AdaBoostClassifier(base_estimator= KNeighborsClassifier(n_neighbors=StandardMLParams.KNN_n_neighbors, metric=StandardMLParams.KNN_metric), n_estimators=StandardMLParams.Bagging_n_estimators,random_state=0)
        elif (StandardMLParams.Bagging_base_estimator == 'DecisionTree'):
            model = AdaBoostClassifier(DecisionTreeClassifier(random_state=0, criterion=StandardMLParams.DecisionTree_criterion, splitter=StandardMLParams.DecisionTree_splitter),
                n_estimators=StandardMLParams.Bagging_n_estimators, random_state=0)
        model = model.fit(X_train, y_train)

    return (model)



def test_StandardML_moreModelsPossible_v3(data,trueLabels,  model):
    # number of clases
    (unique_labels, counts) = np.unique(trueLabels, return_counts=True)
    numLabels = len(unique_labels)
    if (numLabels==1): #in specific case when in test set all the same label
        numLabels=2

    # #PREDICT LABELS
    # X_test0 = np.float32(data)
    # X_test0[np.where(np.isinf(X_test0))] = np.nan
    # if (np.size(X_test0,0)==0):
    #         print('X test size 0 is 0!!', X_test0.shape)
    # if (np.size(X_test0,1)==0):
    #         print('X test size 1 is 0!!', X_test0.shape)
    # col_mean = np.nanmean(X_test0, axis=0)
    # inds = np.where(np.isnan(X_test0))
    # X_test0[inds] = np.take(col_mean, inds[1])
    # # if still somewhere nan replace with 0
    # X_test0[np.where(np.isnan(X_test0))] = 0
    # X_test=X_test0
    X_test = data
    #calculate predictions
    y_pred= model.predict(X_test)
    y_probability = model.predict_proba(X_test)

    #pick only probability of predicted class
    y_probability_fin=np.zeros(len(y_pred))
    indx=np.where(y_pred==1)
    if (len(indx[0])!=0):
        y_probability_fin[indx]=y_probability[indx,1]
    else:
        a=0
        print('no seiz predicted')
    indx = np.where(y_pred == 0)
    if (len(indx[0])!=0):
        y_probability_fin[indx] = y_probability[indx,0]
    else:
        a=0
        print('no non seiz predicted')

    #calculate accuracy
    diffLab=y_pred-trueLabels
    indx=np.where(diffLab==0)
    acc= len(indx[0])/len(trueLabels)

    # calculate performance and distances per class
    accPerClass=np.zeros(numLabels)
    distFromCorr_PerClass = np.zeros(numLabels)
    distFromWrong_PerClass = np.zeros(numLabels)
    for l in range(numLabels):
        indx=np.where(trueLabels==l)
        trueLabels_part=trueLabels[indx]
        predLab_part=y_pred[indx]
        diffLab = predLab_part - trueLabels_part
        indx2 = np.where(diffLab == 0)
        if (len(indx[0])==0):
            accPerClass[l] = np.nan
        else:
            accPerClass[l] = len(indx2[0]) / len(indx[0])

    return(y_pred, y_probability_fin, acc, accPerClass)


def calcHistogramValues_v2(sig, segmentedLabels, histbins):
    '''takes one window of signal - all ch and labels, separates seiz and nonSeiz and
    calculates histogram of values  during seizure and non seizure '''
    numBins=int(histbins)
    sig2 = sig[~np.isnan(sig)]
    sig2 = sig2[np.isfinite(sig2)]
    # maxValFeat=np.max(sig)
    # binBorders=np.arange(0, maxValFeat+1, (maxValFeat+1)/numBins)

    # sig[sig == np.inf] = np.nan
    indxs=np.where(segmentedLabels==0)[0]
    nonSeiz = sig[indxs]
    nonSeiz = nonSeiz[~np.isnan(nonSeiz)]
    try:
        nonSeiz_hist = np.histogram(nonSeiz, bins=numBins, range=(np.min(sig2), np.max(sig2)))
    except:
        print('Error with hist ')

    indxs = np.where(segmentedLabels == 1)[0]
    Seiz = sig[indxs]
    Seiz = Seiz[~np.isnan(Seiz)]
    try:
        Seiz_hist = np.histogram(Seiz, bins=numBins, range=(np.min(sig2), np.max(sig2)))
    except:
        print('Error with hist ')

    # normalizing values that are in percentage of total samples - to not be dependand on number of samples
    nonSeiz_histNorm=[]
    nonSeiz_histNorm.append(nonSeiz_hist[0]/len(nonSeiz))
    nonSeiz_histNorm.append(nonSeiz_hist[1])
    Seiz_histNorm=[]
    Seiz_histNorm.append(Seiz_hist[0]/len(Seiz))
    Seiz_histNorm.append(Seiz_hist[1])
    # Seiz_hist[0] = Seiz_hist[0] / len(Seiz_allCh)
    return( Seiz_histNorm, nonSeiz_histNorm)

def concatenateDataFromFiles(fileNames):
    dataAll = []
    for f, fileName in enumerate(fileNames):
        print(os.path.split(fileName)[-1])
        data = readDataFromFile(fileName)

        if (dataAll == []):
            dataAll = data
        else:
            dataAll = np.vstack((dataAll, data))

    return dataAll

def concatenateDataFromFiles_AllocateFirst(fileNames, nrows):

    data = readDataFromFile(fileNames[0]) #reads the first file to uncover the number of columns 

    dataAll = np.zeros((nrows, data.shape[1]))
    start=0
    for f, fileName in enumerate(fileNames):
        print(os.path.split(fileName)[-1])
        data = readDataFromFile(fileName)

        dataAll[start:start+data.shape[0]] = data
        start=start+data.shape[0]
        # if (dataAll == []):
        #     dataAll = data
        # else:
        #     dataAll = np.vstack((dataAll, data))

    return dataAll

def concatenateDataFromFiles_TSCVFilesInput(fileNames):
    dataAll = []
    startIndxOfFiles=np.zeros(len(fileNames))
    for f, fileName in enumerate(fileNames):
        data = readDataFromFile(fileName)
        data= np.float32(data)
        # reader = csv.reader(open(fileName, "r"))
        # data0 = list(reader)
        # data = np.array(data0).astype("float")
        # separating to data and labels
        # X = data[:, 0:-1]
        # y = data[:, -1]
        dataSource= np.ones(len(data[:, -1]))*f

        if (dataAll == []):
            dataAll = data[:, 0:-1]
            labelsAll = data[:, -1].astype(int)
            # startIndxOfFiles[f]=0
            lenPrevFile=int(len(data[:, -1]))
            startIndxOfFiles[f]=lenPrevFile
        else:
            dataAll = np.vstack((dataAll, data[:, 0:-1]))
            labelsAll = np.hstack((labelsAll, data[:, -1].astype(int)))
            # startIndxOfFiles[f]=int(lenPrevFile)
            lenPrevFile = lenPrevFile+ len(data[:, -1])
            startIndxOfFiles[f] = int(lenPrevFile)
    startIndxOfFiles = startIndxOfFiles.astype((int))
    return (dataAll, labelsAll, startIndxOfFiles)


def generateTSCVindexesFromAllFiles_storeTSCVLabels(dataset, GeneralParams, ZeroCrossFeatureParams, SegSymbParams, folderIn, folderOut, maxWinLen):
    '''Reads the names of the files with labels, generating a vector with the indexes for the TSCV considering all data of a subject stacked 
    in one matrix. It also generate files with the labels for each CV.'''
    
    maxWinLenIndx=int((maxWinLen-SegSymbParams.segLenSec)/SegSymbParams.slidWindStepSec)
    minFirsFileLen=int((3600-SegSymbParams.segLenSec)/SegSymbParams.slidWindStepSec*5) #number of windows of 1h x nHours min
    for patIndx, pat in enumerate(GeneralParams.patients):
        if dataset=='01_SWEC':
            allFiles=np.sort(glob.glob(folderIn + 'ID' + pat + '/'+'ID' + pat + '*Labels.csv.gz'))
        if dataset=='02_CHB':
            allFiles = np.sort(glob.glob(folderIn +'chb' + pat+ '/chb' + pat + '*Labels.csv.gz'))
        
        # print(os.path.split(allFiles[0][:])[-1])
        #chb02_16+ commes first than chb02_16 when ordering files
        if dataset=='02_CHB' and pat=='02':
            if '+' in allFiles[15][:]: #chb02_16+
                cpy = allFiles[15]
                allFiles[15] = allFiles[16]
                allFiles[16] = cpy

        numCVThisSubj=0
        firstFileCreated=0
        startIndxOfFiles=[]
        #LOAD ALL FILES ONE BY ONE
        for fIndx, fileName in enumerate(allFiles):
            print(os.path.split(fileName)[-1])

            labels = readDataFromFile(fileName)

            pom, fileName1 = os.path.split(fileName)
            fileNameOut = fileName1.split('_')[0] #for the labels output files to plot the TSCV sequence

            if 'chb17' in fileNameOut: #chb17 has files 'a', 'b', and 'c'
                fileNameOut='chb17'

            eachSubjDataOutFolder = folderOut+ fileNameOut+'/'
            createFolderIfNotExists(eachSubjDataOutFolder) #for each subject

            #if there is seizure in file find start and stops
            if (np.sum(labels)!=0):
                diffSig=np.diff(np.squeeze(labels))
                szStart=np.where(diffSig==1)[0]
                szStop= np.where(diffSig == -1)[0]


            if (firstFileCreated==0): #first file, append until at least one seizure
                if (fIndx==0):
                    newLabel=labels
                else: #appending to existing file
                    newLabel =np.vstack((newLabel,labels))
                
                if (np.sum(newLabel)>0 and len(newLabel)>=minFirsFileLen): #at least min of hours and 1 seizure in the first file
                    firstFileCreated=1
                    startIndxOfFiles = np.append(startIndxOfFiles, int(len(newLabel)))                  

                    fileNameOut2 = eachSubjDataOutFolder + fileNameOut + '_cv' + str(numCVThisSubj).zfill(3) + '_Labels'
                    saveDataToFile(newLabel.reshape((-1,1)), fileNameOut2, 'gzip')

                    numCVThisSubj = numCVThisSubj + 1

            else:  #not first file, just resave with different cv name
                newLabel=labels
                if len(newLabel) < 1.5*maxWinLenIndx: #up to 1.5x the target, we kepp the original file in full
                        startIndxOfFiles = np.append(startIndxOfFiles, startIndxOfFiles[-1] + int(len(newLabel)))

                        fileNameOut2 = eachSubjDataOutFolder + fileNameOut + '_cv' + str(numCVThisSubj).zfill(3) + '_Labels'
                        saveDataToFile(newLabel.reshape((-1,1)), fileNameOut2, 'gzip')

                        numCVThisSubj = numCVThisSubj + 1
                else:
                    n_new_files = int(len(labels)/maxWinLenIndx)
                    indxStart=0
                    dataMissing = maxWinLenIndx
                    for i in range(n_new_files-1):
                        #check if we would cut seizure in half
                        if (np.sum(labels)!=0):
                            for s in range(len(szStart)):
                                try:
                                    if ( szStart[s]<indxStart+dataMissing  and szStop[s]>indxStart+dataMissing ): #cut would be whenre seizure is
                                        dataMissing=szStop[s]- indxStart #move cut to the end of the seizure
                                except:
                                    print('error')

                        newLabel=labels[indxStart:indxStart+dataMissing,:]

                        #finished this new file to save
                        startIndxOfFiles = np.append(startIndxOfFiles, startIndxOfFiles[-1] + int(len(newLabel)))

                        fileNameOut2 = eachSubjDataOutFolder + fileNameOut + '_cv' + str(numCVThisSubj).zfill(3) + '_Labels'
                        saveDataToFile(newLabel.reshape((-1,1)), fileNameOut2, 'gzip')

                        indxStart = indxStart+dataMissing #start where we stopped
                        numCVThisSubj = numCVThisSubj + 1

                    #last part of the data of each file
                    newLabel=labels[indxStart:len(labels),:]
                    startIndxOfFiles = np.append(startIndxOfFiles, startIndxOfFiles[-1] + int(len(newLabel)))

                    fileNameOut2 = eachSubjDataOutFolder + fileNameOut + '_cv' + str(numCVThisSubj).zfill(3) + '_Labels'
                    saveDataToFile(newLabel.reshape((-1,1)), fileNameOut2, 'gzip')

                    numCVThisSubj = numCVThisSubj + 1 #for next file

        startIndxOfFiles = startIndxOfFiles.astype((int)) #contains indexes of test data
        fileNameOut2 = folderOut + fileNameOut + '/' + fileNameOut + '_TSCV_indexes.csv.gz'
        np.savetxt(fileNameOut2, startIndxOfFiles)




def concatenateFeatures_FirstFileNeedsSeizure_KeepOriginal(folderIn, folderOut,GeneralParams, SegSymbParams, SigInfoParams, patients, maxWinLen):
    '''Concatenate data from files till having the first seizure and write it to another file. The rest of the files are copied to the cross-validation ones as 
    they are '''

    createFolderIfNotExists(folderOut)
    for patIndx, pat in enumerate(GeneralParams.patients):
        print('Subj:'+ pat)
        allFiles=np.sort(glob.glob(folderIn + '/chb' + pat + '/*chb' + pat + '*_OtherFeat.csv.gz'))
        firstFileCreated=0
        numFilesThisSubj=0
        for fIndx, fileName in enumerate(allFiles):
            # reader = csv.reader(open(fileName, "r"))
            # data0 = np.array(list(reader)).astype("float")
            data0 = readDataFromFile(fileName)
            data=data0[:,0:-1]
            label=data0[:,-1]
            pom, fileName1 = os.path.split(fileName)
            fileNameOut = os.path.splitext(fileName1)[0][0:5]

            if (firstFileCreated==0): #first file, append until at least one seizure
                if (fIndx==0):
                    dataOut=data
                    labelOut=label
                else:
                    dataOut=np.vstack((dataOut,data))
                    labelOut = np.hstack((labelOut, label))
                if (np.sum(labelOut)>0 and fIndx>4): #at least 6 h or at least 1 seizure in first file
                    firstFileCreated=1
                    fileNameOut2 = folderOut + '/' + fileNameOut + '_cv' + str(numFilesThisSubj).zfill(3)
                    # writeToCsvFile(dataOut, labelOut, fileNameOut2)
                    saveDataToFile(np.hstack((dataOut, labelOut.reshape((-1,1)))), fileNameOut2, 'gzip')
                    numFilesThisSubj = numFilesThisSubj + 1
            else:  #not first file, just resave with different cv name
                fileNameOut2 = folderOut + '/' + fileNameOut + '_cv' + str(numFilesThisSubj).zfill(3)
                # writeToCsvFile(data, label, fileNameOut2)
                saveDataToFile(np.hstack((data,label.reshape((-1,1)))), fileNameOut2, 'gzip')
                numFilesThisSubj = numFilesThisSubj + 1



def concatenateFeatures_allDataInEqualWindows_FirstFileNeedsSeizure(parallelize, perc_cores, dataset, folderIn, folderOut, GeneralParams, SegSymbParams, ZeroCrossFeatureParams, maxWinLen):
    ''' '''
    print('Reorganizing features files to TSCV approach')
    global number_free_cores

    if parallelize:
        n_cores  = mp.cpu_count()
        n_cores = ceil(n_cores*perc_cores)

        if n_cores > len(GeneralParams.patients):
            n_cores = len(GeneralParams.patients)

        print('Number of used cores: ' + str(n_cores))

        pool = mp.Pool(n_cores)
        number_free_cores = n_cores
        

    maxWinLenIndx=int((maxWinLen-SegSymbParams.segLenSec)/SegSymbParams.slidWindStepSec)
    minFirsFileLen=int((3600-SegSymbParams.segLenSec)/SegSymbParams.slidWindStepSec*5) #number of windows of 1h x nHours min
    for patIndx, pat in enumerate(GeneralParams.patients):
        if dataset=='01_SWEC':
            allFiles=np.sort(glob.glob(folderIn + 'ID' + pat + '/'+'ID' + pat + '*_OtherFeat.csv.gz'))
        if dataset=='02_CHB':
            allFiles = np.sort(glob.glob(folderIn +'chb' + pat+ '/chb' + pat + '*_OtherFeat.csv.gz'))
        
        # print(os.path.split(allFiles[0][:])[-1])
        #chb02_16+ commes first than chb02_16 when ordering files
        if dataset=='02_CHB' and pat=='02':
            if '+' in allFiles[15][:]: #chb02_16+
                cpy = allFiles[15]
                allFiles[15] = allFiles[16]
                allFiles[16] = cpy

        if parallelize:
            pool.apply_async(concatenateFeatureFiles_SeizureInFirstFile_DivideBigFiles, args=(patIndx, allFiles, folderOut, maxWinLenIndx, ZeroCrossFeatureParams, minFirsFileLen), callback=collect_result) 
            number_free_cores = number_free_cores -1
            if number_free_cores==0:
                while number_free_cores==0: #synced in the callback
                    time.sleep(0.1)
                    pass
        else:
            concatenateFeatureFiles_SeizureInFirstFile_DivideBigFiles(patIndx, allFiles, folderOut, maxWinLenIndx, ZeroCrossFeatureParams, minFirsFileLen)

    if parallelize:
        while number_free_cores < n_cores: #wait till all subjects have their data processed
            time.sleep(0.1)
            pass
        
        pool.close()
        pool.join()  

        

def concatenateFeatureFiles_EqualSize_SeizureInFirstFile(patIndx, allFiles, folderOut, maxWinLenIndx, ZeroCrossFeatureParams, minFirsFileLen):
    indxStart=0
    dataMissing=maxWinLenIndx #how much data we target to have per file
    newFileToSave=1
    numFilesThisSubj=0
    firstFileCreated=0
    #LOAD ALL FILES ONE BY ONE
    for fIndx, fileName in enumerate(allFiles):
        dataOtherFeat=readDataFromFile(fileName)

        fileName2=fileName[0:-16]+'ZCFeat.csv.gz'
        dataZCFeat = readDataFromFile(fileName2)

        numCh=int(len(dataZCFeat[0, :])/(len(ZeroCrossFeatureParams.EPS_thresh_arr)+1))
        data= mergeFeatFromTwoMatrixes(dataOtherFeat, dataZCFeat, numCh)

        fileName2=fileName[0:-16]+'Labels.csv.gz'
        labels = readDataFromFile(fileName2)

        pom, fileName1 = os.path.split(fileName)
        fileNameOut = fileName1.split('_')[0]

        print('Patient:', str(patIndx+1), '  ', fileName1)

        #if there is seizure in file find start and stops
        if (np.sum(labels)!=0):
            diffSig=np.diff(np.squeeze(labels))
            szStart=np.where(diffSig==1)[0]
            szStop= np.where(diffSig == -1)[0]


        if (firstFileCreated==0): #first file, append until at least one seizure
            if (fIndx==0):
                newData=data
                newLabel=labels
            else: #appending to existing file
                newData= np.vstack((newData,data))
                newLabel =np.vstack((newLabel,labels))
            
            if (np.sum(newLabel)>0 and len(newLabel)>=minFirsFileLen): #at least 6 h or at least 1 seizure in first file
                firstFileCreated=1
                fileNameOut2 = folderOut + fileNameOut + '/' + fileNameOut + '_cv' + str(numFilesThisSubj).zfill(3)
                # writeToCsvFile(dataOut, labelOut, fileNameOut2)
                saveDataToFile(np.hstack((newData, newLabel.reshape((-1,1)))), fileNameOut2, 'gzip')
                
                fileNameOut2 = folderOut + 'Labels' + '/' + fileNameOut + '_cv' + str(numFilesThisSubj).zfill(3) + '_Labels'
                saveDataToFile(newLabel.reshape((-1,1)), fileNameOut2, 'gzip')
                
                numFilesThisSubj = numFilesThisSubj + 1
                del newData
                del newLabel
        else:  #not first file, just resave with different cv name
            thisFileStillHasData=1
            while (thisFileStillHasData==1):
                #if enough data in file
                if (indxStart + dataMissing <= len(labels)):
                    #in case we are to create a new file but the remaining data after its creating is too few, this remaing
                    #is added to the current file
                    rest = len(labels) - indxStart - dataMissing
                    if rest < int(0.05*maxWinLenIndx) and newFileToSave==1:
                        dataMissing = len(labels)

                    #check if we would cut seizure in half
                    if (np.sum(labels)!=0):
                        for s in range(len(szStart)):
                            try:
                                if ( szStart[s]<indxStart+dataMissing  and szStop[s]>indxStart+dataMissing ): #cut would be whenre seizure is
                                    dataMissing=szStop[s]- indxStart #move cut to the end of the seizure
                            except:
                                print('error')

                    if (newFileToSave==1):
                        newData=data[indxStart:indxStart+dataMissing,:]
                        newLabel=labels[indxStart:indxStart+dataMissing,:]
                    else: #appending to existing file
                        newData= np.vstack((newData,data[indxStart:indxStart+dataMissing,:]))
                        newLabel =np.vstack((newLabel,labels[indxStart:indxStart+dataMissing,:]))

                    #finished this new file to save
                    fileNameOut2 = folderOut + '/' + fileNameOut + '/' + fileNameOut + '_cv' + str(numFilesThisSubj).zfill(3)
                    saveDataToFile(np.hstack((newData, newLabel)), fileNameOut2, 'gzip')
                    
                    fileNameOut2 = folderOut + 'Labels' + '/' + fileNameOut + '_cv' + str(numFilesThisSubj).zfill(3) + '_Labels'
                    saveDataToFile(newLabel.reshape((-1,1)), fileNameOut2, 'gzip')

                    numFilesThisSubj = numFilesThisSubj + 1
                    newFileToSave = 1
                    indxStart = indxStart+dataMissing #start where we stopped
                    dataMissing = maxWinLenIndx
                    thisFileStillHasData=1
                    if indxStart < len(labels):
                        thisFileStillHasData=1
                    else:
                        thisFileStillHasData=0
                        indxStart = 0
                else: #not enough data in file
                    if (newFileToSave==1):
                        newData=data[indxStart:,:] #read until the end of the file
                        newLabel=labels[indxStart:,:]
                    else: #appending to existing file
                        newData= np.vstack((newData,data[indxStart:,:]))
                        newLabel =np.vstack((newLabel,labels[indxStart:,:]))
                    dataMissing = maxWinLenIndx - len(newLabel) #calculate how much data is missing
                    indxStart = 0 #in next file start from beginning
                    thisFileStillHasData=0 #this file has no more data, need to load new one
                    newFileToSave=0


    # save last file's remaining data 
    if (len(labels)- indxStart> int(0.05*maxWinLenIndx) and thisFileStillHasData): 
        fileNameOut2 = folderOut + '/' + fileNameOut + '/' + fileNameOut + '_cv' + str(numFilesThisSubj).zfill(3)
        saveDataToFile(np.hstack((newData, newLabel)), fileNameOut2, 'gzip')
        
        fileNameOut2 = folderOut + 'Labels' + '/' + fileNameOut + '_cv' + str(numFilesThisSubj).zfill(3) + '_Labels'
        saveDataToFile(newLabel.reshape((-1,1)), fileNameOut2, 'gzip')

    return patIndx


def concatenateFeatureFiles_SeizureInFirstFile_DivideBigFiles(patIndx, allFiles, folderOut, maxWinLenIndx, ZeroCrossFeatureParams, minFirsFileLen):
    
    numFilesThisSubj=0
    firstFileCreated=0
    #LOAD ALL FILES ONE BY ONE
    for fIndx, fileName in enumerate(allFiles):
        print(os.path.split(fileName)[-1])
        dataOtherFeat=readDataFromFile(fileName)

        fileName2=fileName[0:-16]+'ZCFeat.csv.gz'
        dataZCFeat = readDataFromFile(fileName2)

        numCh=int(len(dataZCFeat[0, :])/(len(ZeroCrossFeatureParams.EPS_thresh_arr)+1))
        data= mergeFeatFromTwoMatrixes(dataOtherFeat, dataZCFeat, numCh)

        fileName2=fileName[0:-16]+'Labels.csv.gz'
        labels = readDataFromFile(fileName2)

        pom, fileName1 = os.path.split(fileName)
        fileNameOut = fileName1.split('_')[0]

        if 'chb17' in fileNameOut: #chb17 has files 'a', 'b', and 'c'
            fileNameOut='chb17'

        eachSubjDataOutFolder = folderOut+fileNameOut + '/'
        createFolderIfNotExists(eachSubjDataOutFolder) #for each subject

        #if there is seizure in file find start and stops
        if (np.sum(labels)!=0):
            diffSig=np.diff(np.squeeze(labels))
            szStart=np.where(diffSig==1)[0]
            szStop= np.where(diffSig == -1)[0]


        if (firstFileCreated==0): #first file, append until at least one seizure
            if (fIndx==0):
                newData=data
                newLabel=labels
            else: #appending to existing file
                newData= np.vstack((newData,data))
                newLabel =np.vstack((newLabel,labels))
            
            if (np.sum(newLabel)>0 and len(newLabel)>=minFirsFileLen): #at least min of hours and 1 seizure in the first file
                firstFileCreated=1
                fileNameOut2 = eachSubjDataOutFolder + fileNameOut + '_cv' + str(numFilesThisSubj).zfill(3)
                # writeToCsvFile(dataOut, labelOut, fileNameOut2)
                saveDataToFile(np.hstack((newData, newLabel.reshape((-1,1)))), fileNameOut2, 'gzip')
                
                LabelsOutFolder = folderOut + 'Labels' + '/'
                createFolderIfNotExists(LabelsOutFolder) #for each subject                

                fileNameOut2 = LabelsOutFolder + fileNameOut + '_cv' + str(numFilesThisSubj).zfill(3) + '_Labels'
                saveDataToFile(newLabel.reshape((-1,1)), fileNameOut2, 'gzip')
                
                numFilesThisSubj = numFilesThisSubj + 1
                del newData
                del newLabel
        else:  #not first file, just resave with different cv name
            if len(labels) < 1.5*maxWinLenIndx: #up to 1.5x the target, we kepp the original file in full
                    fileNameOut2 = eachSubjDataOutFolder + fileNameOut + '_cv' + str(numFilesThisSubj).zfill(3)
                    saveDataToFile(np.hstack((data, labels.reshape((-1,1)))), fileNameOut2, 'gzip')
                    
                    fileNameOut2 =  LabelsOutFolder + fileNameOut + '_cv' + str(numFilesThisSubj).zfill(3) + '_Labels'
                    saveDataToFile(labels.reshape((-1,1)), fileNameOut2, 'gzip')

                    numFilesThisSubj = numFilesThisSubj + 1
            else:
                n_new_files = int(len(labels)/maxWinLenIndx)
                indxStart=0
                dataMissing = maxWinLenIndx
                for i in range(n_new_files-1):
                    #check if we would cut seizure in half
                    if (np.sum(labels)!=0):
                        for s in range(len(szStart)):
                            try:
                                if ( szStart[s]<indxStart+dataMissing  and szStop[s]>indxStart+dataMissing ): #cut would be whenre seizure is
                                    dataMissing=szStop[s]- indxStart #move cut to the end of the seizure
                            except:
                                print('error')

                    newData=data[indxStart:indxStart+dataMissing,:]
                    newLabel=labels[indxStart:indxStart+dataMissing,:]

                    #finished this new file to save
                    fileNameOut2 = fileNameOut2 = eachSubjDataOutFolder + fileNameOut + '_cv' + str(numFilesThisSubj).zfill(3)
                    saveDataToFile(np.hstack((newData, newLabel)), fileNameOut2, 'gzip')
                    
                    fileNameOut2 = LabelsOutFolder + fileNameOut + '_cv' + str(numFilesThisSubj).zfill(3) + '_Labels'
                    saveDataToFile(newLabel.reshape((-1,1)), fileNameOut2, 'gzip')

                    indxStart = indxStart+dataMissing #start where we stopped
                    numFilesThisSubj = numFilesThisSubj + 1

                #last part of the data of each file
                newData=data[indxStart:len(labels),:]
                newLabel=labels[indxStart:len(labels),:]

                #finished this new file to save
                fileNameOut2 = fileNameOut2 = eachSubjDataOutFolder + fileNameOut + '_cv' + str(numFilesThisSubj).zfill(3)
                saveDataToFile(np.hstack((newData, newLabel)), fileNameOut2, 'gzip')
                
                fileNameOut2 = LabelsOutFolder + fileNameOut + '_cv' + str(numFilesThisSubj).zfill(3) + '_Labels'
                saveDataToFile(newLabel.reshape((-1,1)), fileNameOut2, 'gzip')

                numFilesThisSubj = numFilesThisSubj + 1 #for next file


    return patIndx



def mergeFeatFromTwoMatrixes(mat1, mat2, numCh):
    numFeat1=int(len(mat1[0, :])/numCh)
    numFeat2=int(len(mat2[0, :])/numCh)
    numColPerCh=numFeat1+numFeat2
    matFinal=np.zeros((len(mat1[:,0]), numColPerCh*numCh))
    for ch in range(numCh):
        matFinal[:, ch*numColPerCh : ch*numColPerCh + numFeat1]=mat1[ :, ch*numFeat1: (ch+1)*numFeat1]
        matFinal[:,  ch * numColPerCh + numFeat1  : ch * numColPerCh + numColPerCh ] = mat2[:, ch * numFeat2: (ch + 1) * numFeat2]
    return matFinal

def plotRawDataLabels(dataset, folderIn,  GeneralParams,SegSymbParams):
    
    print('Print labels for all subjects extracted feature files')
    
    folderOut=folderIn +'LabelsInTime/'
    createFolderIfNotExists(folderOut)

    seizureStableLenToTestIndx = int(GeneralParams.seizureStableLenToTest / SegSymbParams.slidWindStepSec)
    seizureStablePercToTest = GeneralParams.seizureStablePercToTest
    distanceBetweenSeizuresIndx = int(GeneralParams.distanceBetween2Seizures / SegSymbParams.slidWindStepSec)
    numLabelsPerHour = 60 * 60 / SegSymbParams.slidWindStepSec
    toleranceFP_bef = int(GeneralParams.toleranceFP_befSeiz / SegSymbParams.slidWindStepSec)
    toleranceFP_aft = int(GeneralParams.toleranceFP_aftSeiz / SegSymbParams.slidWindStepSec)

    for patIndx, pat in enumerate(GeneralParams.patients):

        if dataset=='01_SWEC':
            allFiles=np.sort(glob.glob(folderIn + 'ID' + pat + '/ID' + pat + '*_Labels*'))
        if dataset=='02_CHB':
            allFiles = np.sort(glob.glob(folderIn +'chb' + pat+ '/chb' + pat +'*_Labels*'))

        if dataset=='02_CHB' and pat=='02':
            if '+' in allFiles[15][:]: #chb02_16+
                cpy = allFiles[15]
                allFiles[15] = allFiles[16]
                allFiles[16] = cpy

        #concatinatin predictions so that predictions for one seizure are based on train set with all seizures before it
        for fIndx, fileName in enumerate(allFiles):
            data=readDataFromFile(fileName)
            if fIndx==0:
                labels = np.squeeze(data)
                testIndx=np.ones(len(data))*(fIndx+1)
            else:
                labels = np.hstack((labels,  np.squeeze(data)))
                testIndx= np.hstack((testIndx, np.ones(len(data))*(fIndx+1)))

        (yPred_SmoothOurStep2, yPred_SmoothOurStep1) = smoothenLabels(labels, seizureStableLenToTestIndx, seizureStablePercToTest, distanceBetweenSeizuresIndx)


        #Plot predictions in time
        fig1 = plt.figure(figsize=(16, 8), constrained_layout=False)
        gs = GridSpec(4, 1, figure=fig1)
        fig1.subplots_adjust(wspace=0.2, hspace=0.2)
        xValues = np.arange(0, len(labels), 1) / (60*60*2)
        ax1 = fig1.add_subplot(gs[0,0])
        ax1.plot(xValues, labels , 'r')
        ax1.set_ylabel('TrueLabel')
        ax1.set_title('Subj'+pat)
        ax1.grid()
        ax1 = fig1.add_subplot(gs[1, 0])
        ax1.plot(xValues, yPred_SmoothOurStep1, 'b')
        ax1.set_ylabel('Step1')
        ax1.grid()
        ax1 = fig1.add_subplot(gs[2, 0])
        ax1.plot(xValues, yPred_SmoothOurStep2, 'm')
        ax1.set_ylabel('Step2')
        ax1.grid()
        ax1 = fig1.add_subplot(gs[3, 0])
        ax1.plot(xValues, testIndx , 'k')
        ax1.set_ylabel('FileNr')
        ax1.grid()
        ax1.set_xlabel('Time')
        fig1.show()
        fig1.savefig(folderOut + 'Subj' + pat + '_RawLabels.png', bbox_inches='tight')
        plt.close(fig1)


def kl_divergence(p,q):
    delta=0.000001
    deltaArr=np.ones(len(p))*delta
    p=p+deltaArr
    q=q+deltaArr
    res=sum(p[i] * math.log2(p[i]/q[i]) for i in range(len(p)))
    return res

def js_divergence(p,q):
    m=0.5* (p+q)
    res=0.5* kl_divergence(p,m) +0.5* kl_divergence(q,m)
    return (res)



def saveDataAndLabelsToFile( data, labels,  fileName, type):
    outputName= fileName+'.csv'
    df = pd.DataFrame(data=np.hstack((data, labels.reshape((-1, 1)))))
    if (type=='gzip'):
        df.to_csv(outputName + '.gz', index=False, compression='gzip')
    else:
        df.to_csv(outputName, index=False)

def saveDataToFile( data,  outputName, type):
    if ('.csv' not in outputName):
        outputName= outputName+'.csv'
    df = pd.DataFrame(data=data)
    if (type=='gzip'):
        df.to_csv(outputName + '.gz', index=False, compression='gzip')
    else:
        df.to_csv(outputName, index=False)

def readDataFromFile( inputName):

    if ('.h5' in inputName):
        df = pd.read_hdf(inputName)
    elif ('.csv.gz' in inputName):
        df= pd.read_csv(inputName, compression='gzip')
    else:
        df= pd.read_csv(inputName)
    data=df.to_numpy()
    return (data)


def func_calculatePerformance_AppendingTestData_TSCV(folderIn, GeneralParams, SegSymbParams, dataset):
    folderOut=folderIn +'PerformanceWithAppendedTests/'
    createFolderIfNotExists(folderOut)

    AllSubjDiffPerf_test = np.zeros((len(GeneralParams.patients), 4* 9))
    seizureStableLenToTestIndx = int(GeneralParams.seizureStableLenToTest / SegSymbParams.slidWindStepSec)
    seizureStablePercToTest = GeneralParams.seizureStablePercToTest
    distanceBetweenSeizuresIndx = int(GeneralParams.distanceBetween2Seizures / SegSymbParams.slidWindStepSec)
    numLabelsPerHour = 60 * 60 / SegSymbParams.slidWindStepSec
    toleranceFP_bef = int(GeneralParams.toleranceFP_befSeiz / SegSymbParams.slidWindStepSec)
    toleranceFP_aft = int(GeneralParams.toleranceFP_aftSeiz / SegSymbParams.slidWindStepSec)
    for patIndx, pat in enumerate(GeneralParams.patients):
        print('Patient: ' + pat)
        filesAll = np.sort(glob.glob(folderIn + pat + '/Subj' + pat + '*TestPredictions.csv.gz'))

        numFiles=len(filesAll)
        for cv in range(len(filesAll)):
            fileName2 = 'Subj' + pat + '_cv' + str(cv).zfill(3)
            print(fileName2)
            data= readDataFromFile(filesAll[cv])
            dataSource_test = data[:, -1]
            # indxs = np.where(dataSource_test == 1)[0]  # 0 is the first next seizure after the trained one
            if cv==0:
                trueLabels_AllCV = data[:, 0]
                probabLabels_AllCV=data[:,1]
                predLabels_AllCV=data[ :,2]
                #predictionsSmoothed=data[ indxs,3:-1]
                dataSource_AllCV= data[ :,3] #np.ones(len(indxs))*(cv+1)
            else:
                trueLabels_AllCV = np.hstack((trueLabels_AllCV, data[:, 0]))
                probabLabels_AllCV = np.hstack((probabLabels_AllCV, data[: ,1]))
                predLabels_AllCV=np.hstack((predLabels_AllCV,data[:,2]))
                #predictionsSmoothed = np.vstack((predictionsSmoothed,data[indxs, 3:-1]))
                dataSource_AllCV= np.hstack((dataSource_AllCV, data[ :,3]))#np.ones(len(indxs))*(cv+1)))


        (performanceTest, yPredTest_MovAvrgStep1, yPredTest_MovAvrgStep2,yPredTest_SmoothBayes) = calculatePerformanceAfterVariousSmoothing(predLabels_AllCV, trueLabels_AllCV,probabLabels_AllCV,
                                                                            toleranceFP_bef, toleranceFP_aft,
                                                                            numLabelsPerHour,
                                                                            seizureStableLenToTestIndx,
                                                                            seizureStablePercToTest,
                                                                            distanceBetweenSeizuresIndx,
                                                                            GeneralParams.bayesWind,
                                                                            GeneralParams.bayesProbThresh)
        dataToSave = np.vstack((trueLabels_AllCV, probabLabels_AllCV, predLabels_AllCV, yPredTest_MovAvrgStep1,yPredTest_MovAvrgStep2, yPredTest_SmoothBayes, dataSource_AllCV)).transpose()  # added from which file is specific part of test set
        outputName = folderOut + '/Subj' + pat + '_AppendedTest_Predictions.csv'
        saveDataToFile(dataToSave, outputName, 'gzip')

        AllSubjDiffPerf_test[patIndx, :] =performanceTest

        #plot predictions in time
        # Plot predictions in time
        fig1 = plt.figure(figsize=(16, 8), constrained_layout=False)
        gs = GridSpec(5, 1, figure=fig1)
        fig1.subplots_adjust(wspace=0.2, hspace=0.2)
        xValues = np.arange(0, len(trueLabels_AllCV), 1)*SegSymbParams.slidWindStepSec/3600
        ax1 = fig1.add_subplot(gs[0, 0])
        ax1.plot(xValues, predLabels_AllCV, 'k')
        ax1.set_ylabel('NoSmooth')
        ax1.set_title('Subj' + pat)
        ax1.grid()
        ax1 = fig1.add_subplot(gs[1, 0])
        ax1.plot(xValues, yPredTest_MovAvrgStep1 * 0.8, 'b')
        ax1.plot(xValues, yPredTest_MovAvrgStep2, 'c')
        ax1.set_ylabel('Step1&2')
        ax1.grid()
        ax1 = fig1.add_subplot(gs[2, 0])
        ax1.plot(xValues, yPredTest_SmoothBayes, 'm')
        ax1.set_ylabel('Bayes')
        ax1.grid()
        ax1 = fig1.add_subplot(gs[3, 0])
        ax1.plot(xValues, trueLabels_AllCV, 'r')
        ax1.set_ylabel('TrueLabel')
        ax1.grid()
        ax1 = fig1.add_subplot(gs[4, 0])
        ax1.plot(xValues, dataSource_AllCV, 'k')
        ax1.set_ylabel('FileNr')
        ax1.grid()
        ax1.set_xlabel('Time (h)')
        fig1.show()
        fig1.savefig(folderOut + 'Subj' + pat + '_Appended_TestPredictions.png', bbox_inches='tight')
        plt.close(fig1)

        # ------------------------------------------------------------------------------------------------
        font_size=11
        fig1 = plt.figure(figsize=(6, 3), constrained_layout=False)
        gs = GridSpec(2, 1, figure=fig1)
        fig1.subplots_adjust(wspace=0.2, hspace=0.2)

        xValues = np.arange(0, len(trueLabels_AllCV), 1)*SegSymbParams.slidWindStepSec/3600
        ax1 = fig1.add_subplot(gs[0, 0])
        ax1.plot(xValues, trueLabels_AllCV, color='g', alpha=0.75, linewidth=1)
        ax1.set_ylabel('Label', fontsize=font_size)
        # plt.yticks([0, 1], fontsize=font_size)
        ax1.grid(axis='x', alpha=0.5, linestyle='--', linewidth=1)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.set_xlim(left=0, right=np.max(xValues))  
        # ax1.set_ylim(0, 1)
        # ax1.legend('Predicted',fontsize=font_size-1) 
        plt.tick_params(
                        axis='both',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        bottom=False,      # ticks along the bottom edge are off
                        left=False,
                        labelbottom=False, # labels along the bottom edge are off
                        labelleft=False)

        ax1 = fig1.add_subplot(gs[1, 0])
        ax1.plot(xValues, yPredTest_SmoothBayes, color='r', alpha=0.75, linewidth=1)
        ax1.set_ylabel('Predicted', fontsize=font_size)
        
        plt.yticks([0, 1], fontsize=font_size)
        ax1.grid(axis='x', alpha=0.5, linestyle='--', linewidth=1)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        # ax1.spines['bottom'].set_visible(False)
        ax1.set_xlim(left=0, right=np.max(xValues))  
        # ax1.set_ylim(0, 1)

        plt.tick_params(
                        axis='y',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        left=False,      # ticks along the bottom edge are off
                        labelleft=False) # labels along the bottom edge are off
        # ax1.legend('Label',fontsize=font_size-1)
        ax1.set_xlabel('Time (h)', fontsize=font_size)
        # plt.xticks(xValues, xValues, fontsize=font_size)

        fig1.show()

        plt.savefig(folderOut+ dataset + '_Subj' + pat + '_Appended_TestPredictions_onlyBayes.pdf', bbox_inches='tight', dpi=200, format='pdf')
        # fig1.savefig(folderOut + '/Subj' + pat + '_Appended_TestPredictions_onlyBayes.png', bbox_inches='tight')
        plt.close(fig1)

    # SAVE PERFORMANCE MEASURES FOR ALL SUBJ
    outputName = folderOut + 'AllSubj_AppendedTest_AllPerfMeasures.csv'
    saveDataToFile(AllSubjDiffPerf_test, outputName, 'gzip')




def plotRearangedDataLabelsInTime(dataset, folderIn,  GeneralParams,SegSymbParams):
    folderOut=folderIn +'/LabelsInTime/'
    createFolderIfNotExists(folderOut)

    print('Printing labels for TSCV reorganized files')

    seizureStableLenToTestIndx = int(GeneralParams.seizureStableLenToTest / SegSymbParams.slidWindStepSec)
    seizureStablePercToTest = GeneralParams.seizureStablePercToTest
    distanceBetweenSeizuresIndx = int(GeneralParams.distanceBetween2Seizures / SegSymbParams.slidWindStepSec)
    numLabelsPerHour = 60 * 60 / SegSymbParams.slidWindStepSec
    toleranceFP_bef = int(GeneralParams.toleranceFP_befSeiz / SegSymbParams.slidWindStepSec)
    toleranceFP_aft = int(GeneralParams.toleranceFP_aftSeiz / SegSymbParams.slidWindStepSec)

    # CALCULATING SEIZURE INFORMATION FOR TABLE FOR PAPER
    numSubj=len(GeneralParams.patients)
    totalDataLenInHours=np.zeros((numSubj))
    testDataLenInHours=np.zeros((numSubj))
    firstFileDataLenInHours=np.zeros((numSubj))
    totalNumSeiz=np.zeros((numSubj))
    testNumSeiz=np.zeros((numSubj))

    for patIndx, pat in enumerate(GeneralParams.patients):
        if dataset=='01_SWEC':
            allFiles=np.sort(glob.glob(folderIn + 'ID'+ pat +'/ID' + pat + '*_Labels*'))
        if dataset=='02_CHB':
            allFiles = np.sort(glob.glob(folderIn +'chb' + pat +'/chb' + pat +'*_Labels*'))
        numFiles = len(allFiles)

        print('Pat = ', pat)
        #concatinatin predictions so that predictions for one seizure are based on train set with all seizures before it
        for fIndx, fileName in enumerate(allFiles):
            data=readDataFromFile(fileName)
            if fIndx==0:
                labels = np.squeeze(data[:,-1])
                testIndx=np.ones(len(data[:,-1]))*(fIndx+1)
                firstFileDataLenInHours[patIndx]= len(labels)/(2*60*60)
                startIndx=np.where(np.diff(labels)==1)[0]
                firstFileNumSeiz=len(startIndx)
            else:
                labels = np.hstack((labels,  np.squeeze(data[:,-1])))
                testIndx = np.hstack((testIndx, np.ones(len(data[:,-1]))*(fIndx+1)))

        #Information for paper table
        totalDataLenInHours[patIndx] = len(labels)/(2*60*60)        
        startIndx=np.where(np.diff(labels)==1)[0]
        totalNumSeiz[patIndx]=len(startIndx)
        testNumSeiz[patIndx]=totalNumSeiz[patIndx]-firstFileNumSeiz

        #Plot predictions in time
        fig1 = plt.figure(figsize=(16, 8), constrained_layout=False)
        gs = GridSpec(2, 1, figure=fig1)
        fig1.subplots_adjust(wspace=0.2, hspace=0.2)
        xValues = np.arange(0, len(labels), 1) / (60*60*2)
        ax1 = fig1.add_subplot(gs[0,0])
        ax1.plot(xValues, labels , 'r')
        ax1.set_ylabel('TrueLabel')
        ax1.set_title('Subj'+pat)
        ax1.grid()

        ax1 = fig1.add_subplot(gs[1, 0])
        ax1.plot(xValues, testIndx , 'k')
        ax1.set_ylabel('FileNr')
        ax1.grid()
        ax1.set_xlabel('Time [h]')
        fig1.show()
        fig1.savefig(folderOut + 'Subj' + pat + '_RawLabels.png', bbox_inches='tight')
        plt.close(fig1)

    testDataLenInHours = totalDataLenInHours - firstFileDataLenInHours
    dataToSave=np.vstack(( totalDataLenInHours, testDataLenInHours, totalNumSeiz, testNumSeiz  ))
    dataToSave=(dataToSave*10000).astype(int) /10000
    
    outputName = folderOut  +  '/SeizureInfo_ForPaperTable'
    saveDataToFile(dataToSave, outputName, 'csv')


#callback for the apply_async process paralization
def collect_result(result): 
    global number_free_cores
    global n_cores_semaphore

    while n_cores_semaphore==0: #block callback in case of multiple accesses
        pass

    if n_cores_semaphore:
        n_cores_semaphore=0
        number_free_cores = number_free_cores+1
        n_cores_semaphore=1

if __name__ == "__main__":
    pass

