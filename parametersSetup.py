#!~/anaconda3/bin python

__author__ = "Una Pale"
__credits__ = ["Una Pale", "Renato Zanetti"]
__license__ = "LGPL"
__version__ = "1.0"
__email__ = "una.pale at epfl.ch"

''' file with all parameters'''
import numpy as np
import pickle

class GeneralParams:
    #Label smoothing with moving average
    seizureStableLenToTest=5 #in seconds  - window for performing label voting
    seizureStablePercToTest=0.5 # 50% of 1 in last seizureStableLenToTest values that needs to be 1 to finally keep label 1
    distanceBetween2Seizures=30 #in seconds - if seizures are closer then this then they are merged
    timeBeforeSeizureConsideredAsSeizure=30 #in seconds - if seizure starts bit before true seizure to still consider ok
    numFPperDayThr=1 #for additional measure of performance what number of FP seizures per days we consider ok
    #Label smoothing with bayes
    bayesWind=10 #orignal paper uses 5 windows
    bayesProbThresh= 1.5 #smoothing with cummulative probabilities, threshold from Valentins paper
    #tolerance for FP before and after seizure not to considered as FP
    toleranceFP_befSeiz=10 #in sec
    toleranceFP_aftSeiz=30 #in sec

    patients=[]  #on which subjects to train and test
    plottingON=0  #determines whether some additional plots are plotted
    PersGenApproach='personalized' #'personalized', 'generalized' approaches - generalized not used here

class SigInfoParams:
    #CHB-MIT: common channels among all subjects
    channels = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
                'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',  'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 'FZ-CZ', 'CZ-PZ']
    chToKeep=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])  #which channels to keep - after reordering them in above order
    samplFreq=256 #sampling frequency of data

class SegSymbParams:
    # window discretization
    segLenSec=4 #length of discrete EEG windows on which to perform analysis
    slidWindStepSec=0.5 #step of sliding window
    labelVotingType='majority' #'majority', 'atLeastOne' or 'allOne' #defines how final label of a segment is chosen

class StandardMLParams:
    modelType='DecisionTree' #'KNN', 'SVM', 'DecisionTree', 'RandomForest','BaggingClassifier','AdaBoost'
    trainingDataResampling='NoResampling' #'NoResampling','ROS','RUS','TomekLinks','ClusterCentroids','SMOTE','SMOTEtomek'
    samplingStrategy='default' # depends on resampling, but if 'default' then default for each resampling type, otherwise now implemented only for RUS if not default
    #KNN parameters
    KNN_n_neighbors=5
    KNN_metric='euclidean' #'euclidean', 'manhattan'
    #SVM parameters
    SVM_kernel = 'linear'  # 'linear', 'rbf','poly'
    SVM_C = 1  # 1,100,1000
    SVM_gamma = 'auto' # 0  # 0,10,100
    #DecisionTree and random forest parameters
    DecisionTree_criterion = 'gini'  # 'gini', 'entropy'
    DecisionTree_splitter = 'best'  # 'best','random'
    DecisionTree_max_depth = 0  # 0, 2, 5,10,20
    RandomForest_n_estimators = 100 #10,50, 100,250
    #Bagging, boosting classifier parameters
    Bagging_base_estimator='SVM' #'SVM','KNN', 'DecisionTree'
    Bagging_n_estimators = 100  # 10,50, 100,250
    n_jobs = 1


class FeaturesParams:
    numStandardFeat=62  #56 features from Sopic2018 + 6 AZC features
    featNorm='noNorm' #'Norm','noNorm' #feature normalization flag


class ZeroCrossFeatureParams:
    EPS_thresh_arr = [16, 32, 64, 128, 256] #thresholds for AZC calculation
    buttFilt_order=4 #Butterworth filter order
    buttFilt_lfreq = 1 #low freq
    buttFilt_hfreq=20 #high freq
    samplFreq=256 #expected sampling frequency for the input signal
    winLen=4 # data window length in sec
    winStep=0.5 # non-overlapping data in sec 


#SAVING SETUP once again to update if new info
with open('../PARAMETERS.pickle', 'wb') as f:
    # pickle.dump([GeneralParams, SegSymbParams, SigInfoParams, EEGfreqBands, StandardMLParams, FeaturesParams, ZeroCrossFeatureParams, patients], f)
    pickle.dump([GeneralParams, SegSymbParams, SigInfoParams, StandardMLParams, FeaturesParams,  ZeroCrossFeatureParams], f)

    