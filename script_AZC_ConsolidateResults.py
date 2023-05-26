#!~/anaconda3/bin python

__authors__ = "Una Pale and Renato Zanetti"
__license__ = "LGPL"
__version__ = "1.0"
__email__ = "{una.pale, renato.zaneti} at epfl.ch"

''' script that consolidates the classification results obtained for both
datasets, hence partial results should be available before its execution '''

from VariousFunctionsLib import *
from parametersSetup import *


###################################
# Initial setup
################################### 

Server = 1 #0 - local execution, 1- server execution
datasetPreparationType='Filt_1to20Hz'  # 'Raw', 'Filt_1to20Hz'

#folder that store partial results
if Server:
    folderResults = '/shares/eslfiler1/scratch/ZeroCrossFeatEpilepsy/'
else:
    folderResults = '../AZC_results/'


#------------------------------------------------------------------------------------------------------
#Consolidating results 
#------------------------------------------------------------------------------------------------------
#chb-mit
Dataset='02_CHB'
patients  =['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24']
GeneralParams.patients = patients

folderOut = folderResults + Dataset + '/05_RF_TSCV/' + datasetPreparationType +'ZeroCross/'
func_calculatePerformance_AppendingTestData_TSCV(folderOut, GeneralParams, SegSymbParams, Dataset)

folderOut = folderResults + Dataset + '/05_RF_TSCV/' + datasetPreparationType +'StandardFeat/'
func_calculatePerformance_AppendingTestData_TSCV(folderOut, GeneralParams, SegSymbParams, Dataset)


#------------------------------------------------------------------------------------------------------
#swec-ethz
Dataset='01_SWEC' #02_CHBMIT, 01_iEEG_Bern
patients  =['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18']
GeneralParams.patients = patients

folderOut = folderResults + Dataset + '/05_RF_TSCV/' + datasetPreparationType +'ZeroCross/'
func_calculatePerformance_AppendingTestData_TSCV(folderOut, GeneralParams, SegSymbParams, Dataset)

folderOut = folderResults + Dataset + '/05_RF_TSCV/' + datasetPreparationType +'StandardFeat/'
func_calculatePerformance_AppendingTestData_TSCV(folderOut, GeneralParams, SegSymbParams, Dataset)

#---------------------------------------------------------------------------------------------------------
#plots for paper
#---------------------------------------------------------------------------------------------------------
font_size=11
mksize=7
fig1 = plt.figure(figsize=(15, 6), constrained_layout=False)
gs = GridSpec(2, 2, figure=fig1, width_ratios=[10, 7.5])
fig1.subplots_adjust(wspace=0.15, hspace=0.4)

#-------------------------------------------------------------------------------------------------------
#Subplot CHB - AZC
Dataset='02_CHB'
patients  =['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24']
GeneralParams.patients = patients

folderOut = folderResults + Dataset + '/05_RF_TSCV/' + datasetPreparationType +'ZeroCross/'
folderOut=folderOut +'PerformanceWithAppendedTests/'
outputName = folderOut + 'AllSubj_AppendedTest_AllPerfMeasures.csv.gz'
AllSubjDiffPerf_test= readDataFromFile(outputName)

f1 = np.zeros(len(GeneralParams.patients))
recall = np.zeros(len(GeneralParams.patients))
precision = np.zeros(len(GeneralParams.patients))
far = np.zeros(len(GeneralParams.patients))
for patIndx, pat in enumerate(GeneralParams.patients):        
    recall[patIndx] = AllSubjDiffPerf_test[patIndx, 27]
    precision[patIndx] = AllSubjDiffPerf_test[patIndx, 28]
    f1[patIndx] = AllSubjDiffPerf_test[patIndx, 29]
    far[patIndx] = AllSubjDiffPerf_test[patIndx, 35]

f1_azc_chb = f1
recall_azc_chb = recall
precision_azc_chb = precision

idx_sorted = np.argsort(-f1_azc_chb) #always ascending, so the minus is used to invert the order
p = np.arange(np.size(GeneralParams.patients))
# xValues = GeneralParams.patients[idx_sorted]
ax1 = fig1.add_subplot(gs[0,0])
ax1.plot(f1[idx_sorted]*100, '*', color='r', alpha=0.5, markersize=mksize)
ax1.plot(recall[idx_sorted]*100, '+', color='b', alpha=0.75, markersize=mksize)
ax1.plot(precision[idx_sorted]*100, '.', color='g', alpha=0.75, markersize=mksize)

ax1.set_ylabel('%', fontsize=font_size)
# ax1.set_xlabel('Subjects', fontsize=font_size+1)
ax1.set_title('AZC - CHB-MIT', fontsize=font_size+2)
# ax1.set_xlabel('a) AZC on CHB-MIT', fontsize=font_size+1)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(alpha=0.5, linestyle='--', linewidth=1)
# plt.plot(xValues, far, 'o', color='r')
perfNames = ['F1= ' + str(np.mean(f1_azc_chb)*100)[0:4] + ' ± ' + str(np.std(f1_azc_chb)*100)[0:4],\
        'Sens= ' + str(np.mean(recall_azc_chb)*100)[0:4] + ' ± ' + str(np.std(recall_azc_chb)*100)[0:4],\
            'Prec= ' + str(np.mean(precision_azc_chb)*100)[0:4] + ' ± ' + str(np.std(precision_azc_chb)*100)[0:4]]
ax1.legend(perfNames,fontsize=font_size-1)
plt.xticks(p, p[idx_sorted]+1, fontsize=font_size)
plt.yticks([0, 20, 40, 60, 80, 100], fontsize=font_size)


#-------------------------------------------------------------------------------------------------------
#Subplot CHB - CLF
folderOut = folderResults + Dataset + '/05_RF_TSCV/' + datasetPreparationType +'StandardFeat/'
folderOut=folderOut +'PerformanceWithAppendedTests/'
outputName = folderOut + 'AllSubj_AppendedTest_AllPerfMeasures.csv.gz'
AllSubjDiffPerf_test= readDataFromFile(outputName)

f1 = np.zeros(len(GeneralParams.patients))
recall = np.zeros(len(GeneralParams.patients))
precision = np.zeros(len(GeneralParams.patients))
far = np.zeros(len(GeneralParams.patients))
for patIndx, pat in enumerate(GeneralParams.patients):        
    recall[patIndx] = AllSubjDiffPerf_test[patIndx, 27]
    precision[patIndx] = AllSubjDiffPerf_test[patIndx, 28]
    f1[patIndx] = AllSubjDiffPerf_test[patIndx, 29]
    far[patIndx] = AllSubjDiffPerf_test[patIndx, 35]

f1_std_chb = f1
recall_std_chb = recall
precision_std_chb = precision
# idx_sorted = np.argsort(-f1) #always ascending, so the minus is used to invert the order
# p = np.arange(np.size(GeneralParams.patients))
# xValues = GeneralParams.patients[idx_sorted]
ax1 = fig1.add_subplot(gs[1,0])
ax1.plot(f1[idx_sorted]*100, '*', color='r', alpha=0.5, markersize=mksize)
ax1.plot(recall[idx_sorted]*100, '+', color='b', alpha=0.75, markersize=mksize)
ax1.plot(precision[idx_sorted]*100, '.', color='g', alpha=0.75, markersize=mksize)

# ax1.set_title('CHB-MIT')
ax1.set_ylabel('%', fontsize=font_size)
ax1.set_xlabel('Subjects', fontsize=font_size+1)
ax1.set_title('CLF - CHB-MIT', fontsize=font_size+2)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(alpha=0.5, linestyle='--', linewidth=1)
# plt.plot(xValues, far, 'o', color='r')
perfNames = ['F1= ' + str(np.mean(f1_std_chb)*100)[0:4] + ' ± ' + str(np.std(f1_std_chb)*100)[0:4],\
    'Sens= ' + str(np.mean(recall_std_chb)*100)[0:4] + ' ± ' + str(np.std(recall_std_chb)*100)[0:4],\
        'Prec= ' + str(np.mean(precision_std_chb)*100)[0:4] + ' ± ' + str(np.std(precision_std_chb)*100)[0:4]]
ax1.legend(perfNames,fontsize=font_size-1)
plt.xticks(p, p[idx_sorted]+1, fontsize=font_size)
plt.yticks([0, 20, 40, 60, 80, 100], fontsize=font_size)



#-------------------------------------------------------------------------------------------------------
#Subplot SWEC - AZC

Dataset='01_SWEC'
patients  =['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18']
GeneralParams.patients = patients
folderOut = folderResults + Dataset + '/05_RF_TSCV/' + datasetPreparationType +'ZeroCross/'
folderOut=folderOut +'PerformanceWithAppendedTests/'
outputName = folderOut + 'AllSubj_AppendedTest_AllPerfMeasures.csv.gz'
AllSubjDiffPerf_test= readDataFromFile(outputName)

p = np.arange(np.size(GeneralParams.patients))

f1 = np.zeros(len(GeneralParams.patients))
recall = np.zeros(len(GeneralParams.patients))
precision = np.zeros(len(GeneralParams.patients))
far = np.zeros(len(GeneralParams.patients))
for patIndx, pat in enumerate(GeneralParams.patients):        
    recall[patIndx] = AllSubjDiffPerf_test[patIndx, 27]
    precision[patIndx] = AllSubjDiffPerf_test[patIndx, 28]
    f1[patIndx] = AllSubjDiffPerf_test[patIndx, 29]
    far[patIndx] = AllSubjDiffPerf_test[patIndx, 35]

f1_azc_ieeg = f1
recall_azc_ieeg = recall
precision_azc_ieeg = precision
idx_sorted = np.argsort(-f1_azc_ieeg) #always ascending, so the minus is used to invert the order

ax1 = fig1.add_subplot(gs[0,1])
ax1.plot(f1[idx_sorted]*100, '*', color='r', alpha=0.5, markersize=mksize)
ax1.plot(recall[idx_sorted]*100, '+', color='b', alpha=0.75, markersize=mksize)
ax1.plot(precision[idx_sorted]*100, '.', color='g', alpha=0.75, markersize=mksize)
ax1.grid(alpha=0.5, linestyle='--', linewidth=1)
ax1.set_title('AZC - SWEC-ETHZ', fontsize=font_size+2)
# ax1.set_ylabel('%', fontsize=font_size)
# ax1.set_xlabel('Subjects', fontsize=font_size+1)
# ax1.set_title('SWEC-ETHZ')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

perfNames = ['F1= ' + str(np.mean(f1_azc_ieeg)*100)[0:4] + ' ± ' + str(np.std(f1_azc_ieeg)*100)[0:4],\
    'Sens= ' + str(np.mean(recall_azc_ieeg)*100)[0:4] + ' ± ' + str(np.std(recall_azc_ieeg)*100)[0:4],\
        'Prec= ' + str(np.mean(precision_azc_ieeg)*100)[0:4] + ' ± ' + str(np.std(precision_azc_ieeg)*100)[0:4]]
ax1.legend(perfNames,fontsize=font_size-1)
plt.xticks(p, p[idx_sorted]+1, fontsize=font_size)
plt.yticks([0, 20, 40, 60, 80, 100], fontsize=font_size)


#-------------------------------------------------------------------------------------------------------
#Subplot SWEC -  CLF
folderOut = folderResults + Dataset + '/05_RF_TSCV/' + datasetPreparationType +'StandardFeat/'
folderOut=folderOut +'PerformanceWithAppendedTests/'
outputName = folderOut + 'AllSubj_AppendedTest_AllPerfMeasures.csv.gz'
AllSubjDiffPerf_test= readDataFromFile(outputName)

p = np.arange(np.size(GeneralParams.patients))

f1 = np.zeros(len(GeneralParams.patients))
recall = np.zeros(len(GeneralParams.patients))
precision = np.zeros(len(GeneralParams.patients))
far = np.zeros(len(GeneralParams.patients))
for patIndx, pat in enumerate(GeneralParams.patients):        
    recall[patIndx] = AllSubjDiffPerf_test[patIndx, 27]
    precision[patIndx] = AllSubjDiffPerf_test[patIndx, 28]
    f1[patIndx] = AllSubjDiffPerf_test[patIndx, 29]
    far[patIndx] = AllSubjDiffPerf_test[patIndx, 35]

# idx_sorted = np.argsort(-f1) #always ascending, so the minus is used to invert the order
f1_std_ieeg = f1
recall_std_ieeg = recall
precision_std_ieeg = precision

ax1 = fig1.add_subplot(gs[1,1])
ax1.plot(f1[idx_sorted]*100, '*', color='r', alpha=0.5, markersize=mksize)
ax1.plot(recall[idx_sorted]*100, '+', color='b', alpha=0.75, markersize=mksize)
ax1.plot(precision[idx_sorted]*100, '.', color='g', alpha=0.75, markersize=mksize)
ax1.grid(alpha=0.5, linestyle='--', linewidth=1)
# ax1.set_ylabel('%', fontsize=font_size)
ax1.set_xlabel('Subjects', fontsize=font_size+1)
ax1.set_title('CLF - SWEC-ETHZ', fontsize=font_size+2)
# ax1.set_title('SWEC-ETHZ')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

perfNames = ['F1= ' + str(np.mean(f1_std_ieeg)*100)[0:4] + ' ± ' + str(np.std(f1_std_ieeg)*100)[0:4],\
    'Sens= ' + str(np.mean(recall_std_ieeg)*100)[0:4] + ' ± ' + str(np.std(recall_std_ieeg)*100)[0:4],\
        'Prec= ' + str(np.mean(precision_std_ieeg)*100)[0:4] + ' ± ' + str(np.std(precision_std_ieeg)*100)[0:4]]
ax1.legend(perfNames,fontsize=font_size-1)
plt.xticks(p, p[idx_sorted]+1, fontsize=font_size)
plt.yticks([0, 20, 40, 60, 80, 100], fontsize=font_size)


print('AZC- Sen > 0.7: ' + str((np.sum(recall_azc_chb>0.7) + np.sum(recall_azc_ieeg>0.7))/42))
print('STD- Sen > 0.7: ' + str((np.sum(recall_std_chb>0.7) + np.sum(recall_std_ieeg>0.7))/42))

print('AZC- F1 > 0.7: ' + str((np.sum(f1_azc_chb>0.7) + np.sum(f1_azc_ieeg>0.7))/42))
print('STD - F1 > 0.7: ' + str((np.sum(f1_std_chb>0.7) + np.sum(f1_std_ieeg>0.7))/42))

# plt.show()
plt.savefig(folderResults+'AllSubj_Consolidate_PaperPerfMeasures.pdf', bbox_inches='tight', dpi=200, format='pdf')
plt.close()
