#!~/anaconda3/bin python

'''script that includes all functions related to the use of the AZC feature'''

__authors__ = "Una Pale, Renato Zanetti, and Tomas Teijeiro"
__license__ = "LGPL"
__version__ = "1.0"
__email__ = "{una.pale, renato.zaneti} at epfl.ch"


import os
import glob
# import csv
import matplotlib
matplotlib.use('Agg') #to do not show figures
from scipy.io import loadmat
from scipy.signal import decimate
from VariousFunctionsLib import *

########################################################
#global variables for multi-core operation
files_processed=[]
n_files_per_patient=[]
number_free_cores = 0
number_free_cores_pat = 0
n_cores_semaphore = 1


###########################################################################
##defining functions needed
def polygonal_approx(arr, epsilon):
    """
    Performs an optimized version of the Ramer-Douglas-Peucker algorithm assuming as an input
    an array of single values, considered consecutive points, and **taking into account only the
    vertical distances**.
    """
    def max_vdist(arr, first, last):
        """
        Obtains the distance and the index of the point in *arr* with maximum vertical distance to
        the line delimited by the first and last indices. Returns a tuple (dist, index).
        """
        if first == last:
            return (0.0, first)
        frg = arr[first:last+1]
        leng = last-first+1
        dist = np.abs(frg - np.interp(np.arange(leng),[0, leng-1], [frg[0], frg[-1]]))
        idx = np.argmax(dist)
        return (dist[idx], first+idx)

    if epsilon <= 0.0:
        raise ValueError('Epsilon must be > 0.0')
    if len(arr) < 3:
        return arr
    result = set()
    stack = [(0, len(arr) - 1)]
    while stack:
        first, last = stack.pop()
        max_dist, idx = max_vdist(arr, first, last)
        if max_dist > epsilon:
            stack.extend([(first, idx),(idx, last)])
        else:
            result.update((first, last))
    return np.array(sorted(result))

def zero_crossings(arr):
    """Returns the positions of zero-crossings in the derivative of an array, as a binary vector"""
    return np.diff(np.sign(np.diff(arr))) != 0


def calculateMovingAvrgMeanWithUndersampling_v2(data, winLen, winStep):
    lenSig=len(data)
    index = np.arange(0, lenSig - winLen, winStep)

    segmData = np.zeros(len(index))
    for i in range(len(index)): #-1
        x = data[index[i]:index[i] + winLen]
        segmData[i]=np.mean(x)
    return(segmData)



def calculateFeaturesPerEachFile_CLFAndAZCFeatures_CHB(parallelize, perc_cores, EPS_thresh_arr,folderIn, folderOutFeatures, patients,  SigInfoParams, ZeroCrossFeatureParams, calculateOtherFeat, calculateZCFeat):
    global files_processed
    global n_files_per_patient
    global number_free_cores
    
    print('Extracting features from all subjects files')
    
    numFeat = len(EPS_thresh_arr) + 1
    
    if parallelize:
        n_cores  = mp.cpu_count()
        n_cores = ceil(n_cores*perc_cores)
        print('Number of cores: ' + str(n_cores))

        pool = mp.Pool(n_cores)
        number_free_cores = n_cores
        
    files_processed=np.zeros(len(patients))
    n_files_per_patient = np.zeros(len(patients))
        
    # go through all patients
    for patIndx, pat in enumerate(patients):
        filesIn=np.sort(glob.glob(folderIn + 'chb' + pat+ '/chb' + pat + '*.csv.gz')) #only the data files
        numFiles=len(filesIn)
        n_files_per_patient[patIndx]=numFiles

        print('Pat: ' + str(pat) + ' Nfiles: ' + str(numFiles))

        folderOutFeaturesSub = folderOutFeatures + 'chb' + pat + '/'

        if not os.path.exists(folderOutFeaturesSub):
            try:
                os.mkdir(folderOutFeaturesSub)
            except OSError:
                print("Creation of the directory %s failed" % folderOutFeaturesSub)

        for fileIndx, file in enumerate(filesIn):
            pom, fileName1 = os.path.split(file)
            fileName2 = fileName1[:-7]
            print('File: '+ fileName2 + '  NFilesprocessed: '+ str(files_processed[patIndx]) + '  Out of: ' + str(n_files_per_patient[patIndx]))
            fileName3 = fileName2.split('_')
            file_number_str = fileName3[1].zfill(3) #for file-reading proper ordering

            if '+' in fileName2 and pat=='02': #chb02_16+    
                file_number_str = '0'+ file_number_str

            numFeat = len(EPS_thresh_arr) + 1
            
            # reading data
            data = readDataFromFile(file)
            # separating to data and labels
            X = data[:, SigInfoParams.chToKeep]
            y = data[:, -1]
            (lenData, numCh) = X.shape
            labels = y[0:lenData - 2]

            index = np.arange(0, lenData - int(ZeroCrossFeatureParams.samplFreq * ZeroCrossFeatureParams.winLen),
                              int(ZeroCrossFeatureParams.samplFreq * ZeroCrossFeatureParams.winStep))
            
            labelsSegm = calculateMovingAvrgMeanWithUndersampling_v2(labels, int(
                ZeroCrossFeatureParams.samplFreq * ZeroCrossFeatureParams.winLen), int(
                ZeroCrossFeatureParams.samplFreq * ZeroCrossFeatureParams.winStep))
            labelsSegm = (labelsSegm > 0.5) * 1
            

            if parallelize:
                pool.apply_async(readFileCalculateFeatures, args=(patIndx, data, labelsSegm, index, numCh, numFeat, ZeroCrossFeatureParams, EPS_thresh_arr, calculateOtherFeat, calculateZCFeat, folderOutFeaturesSub, fileName3, file_number_str), callback=collect_result) 
                number_free_cores = number_free_cores -1
                if number_free_cores==0:
                    while number_free_cores==0: #synced in the callback
                        time.sleep(0.1)
                        pass
            else:
                readFileCalculateFeatures(patIndx, data, labelsSegm, index, numCh, numFeat, ZeroCrossFeatureParams, EPS_thresh_arr, calculateOtherFeat, calculateZCFeat, folderOutFeaturesSub, fileName3, file_number_str)

    if parallelize:
        while number_free_cores < n_cores: #wait till all subjects have their data processed
            time.sleep(0.1)
            pass
        pool.close()
        pool.join()      




def calculateFeaturesPerEachFile_CLFAndAZCFeatures_SWEC(parallelize, perc_cores, EPS_thresh_arr,folderIn, folderOutFeatures, patients,  SigInfoParams, ZeroCrossFeatureParams, calculateOtherFeat, calculateZCFeat):
    global files_processed
    global n_files_per_patient
    global number_free_cores
    
    print('Extracting features from all subjects files')

    numFeat = len(EPS_thresh_arr) + 1
    
    diff_files_seiz=0

    if parallelize:
        n_cores  = mp.cpu_count()
        n_cores = ceil(n_cores*perc_cores)
        print('Number of cores: ' + str(n_cores))

        pool = mp.Pool(n_cores)
        number_free_cores = n_cores
        
    files_processed=np.zeros(len(patients))
    n_files_per_patient = np.zeros(len(patients))
        
    # go through all patients
    for patIndx, pat in enumerate(patients):
        filesIn=np.sort(glob.glob(folderIn + 'ID' + pat + '/'+ 'ID' + pat + '*h.mat')) #only the data files
        numFiles=len(filesIn)
        n_files_per_patient[patIndx]=numFiles

        print('Pat: ' + str(pat) + ' Nfiles: ' + str(numFiles))

        f = loadmat(folderIn + 'ID' + pat + '/'+ 'ID' + pat +'_info.mat', simplify_cells=True)
        seizure_begin = np.array(f['seizure_begin']) #given in seconds
        seizure_end = np.array(f['seizure_end'])
        fs = np.array(f['fs']).astype(int)

        file_index = (seizure_begin/3600 + 1).astype(int) #sum 1 to fit calculate index to files indexes
        print('file_index: ' + str(file_index))
        seiz_count=0

        folderOutFeaturesSub = folderOutFeatures + 'ID' + pat + '/'

        if not os.path.exists(folderOutFeaturesSub):
            try:
                os.mkdir(folderOutFeaturesSub)
            except OSError:
                print("Creation of the directory %s failed" % folderOutFeaturesSub)

        for fileIndx, file in enumerate(filesIn):
            pom, fileName1 = os.path.split(file)
            fileName2 = os.path.splitext(fileName1)[0]
            print('File: '+ fileName2 + '  NFilesprocessed: '+ str(files_processed[patIndx]) + '  Out of: ' + str(n_files_per_patient[patIndx]))
            fileName3 = fileName2.split('_')
            file_number_str = fileName3[1][:-1]
            file_number = int(file_number_str)

            file_number_str = file_number_str.zfill(3) #for file-reading proper ordering

            numFeat = len(EPS_thresh_arr) + 1
            
            # reading data
            reader = loadmat(file, simplify_cells=True)
            data0 = np.array(reader['EEG']).astype("float").T

            #
            data = decimate(data0, int(fs/ZeroCrossFeatureParams.samplFreq), axis=0, zero_phase=True)

            (lenData, numCh) = data.shape
            y=np.zeros((lenData,))

            if diff_files_seiz: #previous file had a seizure which spams till the current file 
                y[old_beg:old_end]=1
                diff_files_seiz=0

            idx_seiz = np.where(file_index==file_number)[0]
            for i, v in enumerate(idx_seiz):
                print('Seizure found at file: ' + str(file_index[v]))
                beg = int((seizure_begin[v]%3600*fs)*(ZeroCrossFeatureParams.samplFreq/fs))
                end = int((seizure_end[v]%3600*fs)*(ZeroCrossFeatureParams.samplFreq/fs))

                if end < beg: #seizures in different files
                    diff_files_seiz = 1
                    old_end = end
                    old_beg = 0
                    end = int((3600*fs)*(ZeroCrossFeatureParams.samplFreq/fs))

                y[beg:end]=1


            labels = y[0:lenData - 2]
            # windowing the data: 'index' contains start:stop indexes
            index = np.arange(0, lenData - int(ZeroCrossFeatureParams.samplFreq * ZeroCrossFeatureParams.winLen),
                                int(ZeroCrossFeatureParams.samplFreq * ZeroCrossFeatureParams.winStep))

            #'labels' contains one value per datapoint. 'labelsSegm' contains one value per window (> 50% ictal data --> ictal label)
            labelsSegm = calculateMovingAvrgMeanWithUndersampling_v2(labels, int(
                ZeroCrossFeatureParams.samplFreq * ZeroCrossFeatureParams.winLen), int(
                ZeroCrossFeatureParams.samplFreq * ZeroCrossFeatureParams.winStep))
            labelsSegm = (labelsSegm > 0.5) * 1

            if parallelize:
                pool.apply_async(readFileCalculateFeatures, args=(patIndx, data, labelsSegm, index, numCh, numFeat, ZeroCrossFeatureParams, EPS_thresh_arr, calculateOtherFeat, calculateZCFeat, folderOutFeaturesSub, fileName3, file_number_str), callback=collect_result) 
                number_free_cores = number_free_cores -1
                if number_free_cores==0:
                    while number_free_cores==0: #synced in the callback 'collect_result'
                        time.sleep(0.1)
                        pass
            else:
                readFileCalculateFeatures(patIndx, data, labelsSegm, index, numCh, numFeat, ZeroCrossFeatureParams, EPS_thresh_arr, calculateOtherFeat, calculateZCFeat, folderOutFeaturesSub, fileName3, file_number_str)

    if parallelize:
        while number_free_cores < n_cores: #wait till all subjects have their data processed
            time.sleep(0.1)
            pass
        
        pool.close()
        pool.join()           
            


def readFileCalculateFeatures(patIndx, data, labelsSegm, index, numCh, numFeat, ZeroCrossFeatureParams, EPS_thresh_arr, calculateOtherFeat, calculateZCFeat, folderOutFeaturesSub, fileName3, file_number_str):

    sos = signal.butter(4, [1, 20], 'bandpass', fs=ZeroCrossFeatureParams.samplFreq, output='sos')
    allsigFilt = signal.sosfiltfilt(sos, data, axis=0)
    del data
    zeroCrossStandard = np.zeros((len(index), numCh))
    zeroCrossApprox = np.zeros((len(index), numCh))
    zeroCrossFeaturesAll = np.zeros((len(index), numFeat * numCh))
    for ch in range(numCh):
           
        # t=time.time()
        sigFilt=allsigFilt[:,ch]
        # calculate CLF
        if (calculateOtherFeat==1):
            featOther = calculateOtherMLfeatures_oneCh(np.copy(sigFilt), ZeroCrossFeatureParams)
            if (ch == 0):
                AllFeatures = featOther
            else:
                AllFeatures = np.hstack((AllFeatures, featOther))

        #AZC
        if (calculateZCFeat==1):
            # Zero-crossing of the original signal, counted in 1-second continuous sliding window
            # zeroCrossStandard[:,ch] = np.convolve(zero_crossings(sigFilt), np.ones(ZeroCrossFeatures.samplFreq), mode='same')
            x = np.convolve(zero_crossings(sigFilt), np.ones(ZeroCrossFeatureParams.samplFreq), mode='same')
            # zeroCrossStandard[:,ch] =calculateMovingAvrgMeanWithUndersampling_v2(x, ZeroCrossFeatureParams.samplFreq)
            zeroCrossStandard[:, ch] = calculateMovingAvrgMeanWithUndersampling_v2(x, int(
                ZeroCrossFeatureParams.samplFreq * ZeroCrossFeatureParams.winLen), int(
                ZeroCrossFeatureParams.samplFreq * ZeroCrossFeatureParams.winStep))
            zeroCrossFeaturesAll[:, numFeat * ch] = zeroCrossStandard[:, ch]


            for EPSthrIndx, EPSthr in enumerate(EPS_thresh_arr):
                # Signal simplification at the given threshold, and zero crossing count in the same way
                sigApprox = polygonal_approx(sigFilt, epsilon=EPSthr)
                # axs[0].plot(sigApprox, sigFilt[sigApprox], alpha=0.6)
                sigApproxInterp = np.interp(np.arange(len(sigFilt)), sigApprox, sigFilt[sigApprox])
                # zeroCrossApprox[:,ch] = np.convolve(zero_crossings(sigApproxInterp), np.ones(ZeroCrossFeatures.samplFreq), mode='same')
                x = np.convolve(zero_crossings(sigApproxInterp), np.ones(ZeroCrossFeatureParams.samplFreq),
                                mode='same')
                # zeroCrossApprox[:, ch] =  calculateMovingAvrgMeanWithUndersampling_v2(x, ZeroCrossFeatureParams.samplFreq)
                zeroCrossApprox[:, ch] = calculateMovingAvrgMeanWithUndersampling_v2(x, int(
                    ZeroCrossFeatureParams.samplFreq * ZeroCrossFeatureParams.winLen), int(
                    ZeroCrossFeatureParams.samplFreq * ZeroCrossFeatureParams.winStep))
                # save to matrix with all features that will be saved
                zeroCrossFeaturesAll[:, numFeat * ch + EPSthrIndx + 1] = zeroCrossApprox[:, ch]

    if (calculateZCFeat == 1):
        # save for this file Zero cross features
        df = pd.DataFrame(data=zeroCrossFeaturesAll)
        outputName = folderOutFeaturesSub + fileName3[0]+ '_' + file_number_str+ 'h_ZCFeat.csv.gz'
        #to read from disk: df = pd.read_csv('dfsavename.csv.gz', compression='gzip')
        df.to_csv(outputName, index=False, compression='gzip') 
        # np.savetxt(outputName, zeroCrossFeaturesAll, delimiter=",")

    if (calculateOtherFeat == 1):
        # save for this file all other features
        df = pd.DataFrame(data=AllFeatures)
        outputName = folderOutFeaturesSub + fileName3[0]+ '_' + file_number_str+ 'h_OtherFeat.csv.gz'        
        df.to_csv(outputName, index=False, compression='gzip')
        # np.savetxt(outputName, AllFeatures, delimiter=",")

    # save for this file labels
    df = pd.DataFrame(data=labelsSegm)
    outputName = folderOutFeaturesSub + fileName3[0]+ '_' + file_number_str+ 'h_Labels.csv.gz'
    df.to_csv(outputName, index=False, compression='gzip')
    # np.savetxt(outputName, labelsSegm, delimiter=",")

    return (patIndx)



#callback for the apply_async process paralization
def collect_result(result): 
    global files_processed
    global number_free_cores
    global n_cores_semaphore
    global n_files_per_patient

    while n_cores_semaphore==0: #block callback in case of multiple accesses
        pass

    if n_cores_semaphore:
        n_cores_semaphore=0
        files_processed[result] += 1
        number_free_cores = number_free_cores+1
        n_cores_semaphore=1

if __name__ == "__main__":
    pass
