#!~/anaconda3/bin python

__authors__ = "Una Pale and Renato Zanetti"
__license__ = "LGPL"
__version__ = "1.0"
__email__ = "{una.pale, renato.zaneti} at epfl.ch"

''' script that :
- calculates KL divergence values (KL_NSS - non seizure to seizure, JS- jensen shannon) 
- calculated from raw feature values and saves it for each subject 
- calculated average for all subjects and plots in different ways
'''

from parametersSetup import *
from VariousFunctionsLib import *

from matplotlib.patches import Polygon

###################################
# Initial setup
################################### 

Dataset='01_SWEC' #'01_SWEC', '02_CHB' #select the dataset to be processed
Server = 1 #0 - local execution, 1- server execution
datasetPreparationType='Filt_1to20Hz'  # 'Raw', 'Filt_1to20Hz'

# DEFINING INPUT/OUTPUT FOLDERS
if Dataset=='01_SWEC':
    if Server:
        folderInData = '/shares/eslfiler1/databases/medical/swec-ethz/ieeg/long-term-dataset/'
        os.nice(20) #set low priority level for longer-running process
    else:
        folderInData = '../iEEG/'
    patients  =['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18']


if Dataset=='02_CHB':
    if Server:
        folderInData = '/shares/eslfiler1/databases/medical/chb-mit/edf/' #expect all subject's files in the same directory
        os.nice(20) #set low priority level for longer-running process
    else:
        folderInData = '../chbmit/'
    patients  =['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24']

#folder that store partial results
if Server:
    folderResults = '/shares/eslfiler1/scratch/ZeroCrossFeatEpilepsy/'
else:
    folderResults = '../AZC_results/'

folderOutFeatures= folderResults+Dataset+'/02_Features_' +datasetPreparationType + '/'
folderDiverg= folderResults+Dataset+'/04_FeatureDivergence/'
createFolderIfNotExists(folderDiverg)

# ###########################################################################################
#SETUPS
GeneralParams.patients= patients
GeneralParams.plottingON=0
GeneralParams.PersGenApproach='personalized' #'personalized', 'generalized'
StandardMLParams.modelType = 'RandomForest'  # 'KNN', 'SVM', 'DecisionTree', 'RandomForest','BaggingClassifier','AdaBoost'

FeaturesParams.numStandardFeat=62 #56 (Sopic2018) + 6 AZC

numBins=100 #(30,100) bins used for the histogram calculation
FeatNames=np.array(['meanAmpl', 'LineLength', 'samp_1_d7_1', 'samp_1_d6_1', 'samp_2_d7_1', 'samp_2_d6_1', 'perm_d7_3', 'perm_d7_5', 'perm_d7_7', 'perm_d6_3', 'perm_d6_5', 'perm_d6_7',   'perm_d5_3', 'perm_d5_5',  'perm_d5_7', 'perm_d4_3', 'perm_d4_5', 'perm_d4_7', 'perm_d3_3', 'perm_d3_5', 'perm_d3_7',
           'shannon_en_sig', 'renyi_en_sig', 'tsallis_en_sig', 'shannon_en_d7', 'renyi_en_d7', 'tsallis_en_d7', 'shannon_en_d6', 'renyi_en_d6', 'tsallis_en_d6', 'shannon_en_d5', 'renyi_en_d5', 'tsallis_en_d5', 'shannon_en_d4', 'renyi_en_d4', 'tsallis_en_d4', 'shannon_en_d3', 'renyi_en_d3', 'tsallis_en_d3',
           'p_dc_rel', 'p_mov_rel', 'p_delta_rel', 'p_theta_rel', 'p_alfa_rel', 'p_middle_rel', 'p_beta_rel', 'p_gamma_rel', 'p_dc', 'p_mov', 'p_delta', 'p_theta', 'p_alfa', 'p_middle', 'p_beta', 'p_gamma', 'p_tot',
           'zeroCrossNoApprox', 'ZC_approx_16', 'ZC_approx_32', 'ZC_approx_64', 'ZC_approx_128', 'ZC_approx_256' ])


############################################################################################
# CALCULATING KL DIVERGENCE
numFeat=FeaturesParams.numStandardFeat* len(SigInfoParams.chToKeep)
JSdiverg=np.zeros((len(GeneralParams.patients), numFeat))
KLdiverg_NSS = np.zeros((len(GeneralParams.patients), numFeat))
for patIndx, pat in enumerate(GeneralParams.patients):
    if Dataset=='01_SWEC':
        filesIn=np.sort(glob.glob(folderOutFeatures + 'ID' + pat + '/ID' + pat + '*_Labels.csv.gz'))
    if Dataset=='02_CHB':
        filesIn = np.sort(glob.glob(folderOutFeatures +'chb' + pat+ '/chb' + pat + '*_Labels.csv.gz'))

    numFiles=len(filesIn)
    print('-- Patient:', pat, 'NumFiles:', numFiles)

    #load all files of that subject and concatenate feature values
    for fIndx, fileName in enumerate(filesIn):
        print('----- File:', fIndx)
        #load labels
        labels0 = readDataFromFile(fileName)
        #load feature values
        fileName2 = fileName[:-14] +  '_OtherFeat.csv.gz'
        OtherFeat0 = readDataFromFile(fileName2)
        fileName2 = fileName[:-14] +  '_ZCFeat.csv.gz'
        ZCFeat0 = readDataFromFile(fileName2)

        nch = np.int32((ZCFeat0.shape[1] + OtherFeat0.shape[1])/FeaturesParams.numStandardFeat)
        AllFeat0=mergeFeatFromTwoMatrixes(OtherFeat0, ZCFeat0, nch)
        if (fIndx==0):
            LabelsAll=labels0
            FeatAll=AllFeat0
        else:
            LabelsAll=np.vstack((LabelsAll, labels0))
            FeatAll = np.vstack((FeatAll, AllFeat0))

    #calculate histograms per seizure and non seizure
    for f in range(numFeat):
        (SeizHist, nonSeizHist) = calcHistogramValues_v2(FeatAll[:,f], LabelsAll,numBins)
        KLdiverg_NSS[patIndx, f] = kl_divergence(nonSeizHist[0],SeizHist[0])
        JSdiverg[patIndx,f] = js_divergence(nonSeizHist[0], SeizHist[0])

    # save predictions
    outputName = folderDiverg + 'AllSubj_KLdivergence_NSS.csv'
    np.savetxt(outputName, KLdiverg_NSS, delimiter=",")
    outputName = folderDiverg  + 'AllSubj_JSdivergence.csv'
    np.savetxt(outputName, JSdiverg, delimiter=",")

############################################################################################
############################################################################################
# PLOTTING
reader = csv.reader(open(folderDiverg + 'AllSubj_KLdivergence_NSS.csv', "r"))
KLdiverg_NSS = np.array(list(reader)).astype("float")
reader = csv.reader(open(folderDiverg  + 'AllSubj_JSdivergence.csv', "r"))
JSdiverg = np.array(list(reader)).astype("float")

#analysing per ch
KLdiverg_NSS_reshaped=np.reshape(KLdiverg_NSS, (len(GeneralParams.patients),-1,FeaturesParams.numStandardFeat ))
KLdiverg_NSS_meanForCh=np.nanmean(KLdiverg_NSS_reshaped,1)
KLdiverg_NSS_stdForCh=np.nanstd(KLdiverg_NSS_reshaped,1)
JSdiverg_reshaped=np.reshape(JSdiverg, (len(GeneralParams.patients),-1,FeaturesParams.numStandardFeat ))
JSdiverg_meanForCh=np.nanmean(JSdiverg_reshaped,1)
JSdiverg_stdForCh=np.nanstd(JSdiverg_reshaped,1)

####################################
#PLOTTING KL DIVERGENCE PER SUBJECT
fig1 = plt.figure(figsize=(16, 16), constrained_layout=False)
gs = GridSpec(6, 4, figure=fig1)
fig1.subplots_adjust(wspace=0.4, hspace=0.6)
xValues = np.arange(0, FeaturesParams.numStandardFeat, 1)
for p, pat in enumerate(GeneralParams.patients):
    ax1 = fig1.add_subplot(gs[int(np.floor(p / 4)), np.mod(p, 4)])
    ax1.errorbar(xValues, KLdiverg_NSS_meanForCh[p, :], yerr=KLdiverg_NSS_stdForCh[p, :], fmt='b', label='KL_NSS')
    ax1.errorbar(xValues, JSdiverg_meanForCh[p, :], yerr=JSdiverg_stdForCh[p, :], fmt='m', label='JS')
    ax1.legend()
    ax1.set_xlabel('Feature')
    ax1.set_ylabel('Divergence')
    ax1.set_title('Subj ' + pat)
    ax1.grid()
fig1.show()
fig1.savefig(folderDiverg + 'AllSubj_DifferentDivergenceMeasures_perSubj.png', bbox_inches='tight')
plt.close(fig1)


####################################
# CALCULATING AVERAGE KL VALUES FOR ALL SUBJECTS
KLdiverg_NSS_meanAllSubj=np.nanmean(KLdiverg_NSS_meanForCh,0)
JSdiverg_meanAllSubj=np.nanmean(JSdiverg_meanForCh,0)
KLdiverg_NSS_stdAllSubj=np.nanstd(KLdiverg_NSS_meanForCh,0)
JSdiverg_stdAllSubj=np.nanstd(JSdiverg_meanForCh,0)
# save predictions
outputName = folderDiverg + 'AllSubjAvrg_KLdivergence_NSS.csv'
np.savetxt(outputName, np.vstack((KLdiverg_NSS_meanAllSubj, KLdiverg_NSS_stdAllSubj)), delimiter=",")
outputName = folderDiverg + 'AllSubjAvrg_JSdivergence.csv'
np.savetxt(outputName, np.vstack((JSdiverg_meanAllSubj, JSdiverg_stdAllSubj)), delimiter=",")

#plotting
fig1 = plt.figure(figsize=(16, 16), constrained_layout=False)
gs = GridSpec(2, 1, figure=fig1)
fig1.subplots_adjust(wspace=0.4, hspace=0.2)
#fig1.suptitle('Feature ')
xValues = np.arange(0, FeaturesParams.numStandardFeat, 1)
ax1 = fig1.add_subplot(gs[0,0])
ax1.errorbar(xValues, KLdiverg_NSS_meanAllSubj, yerr=KLdiverg_NSS_stdAllSubj, fmt='b', label='KL_NSS')
#ax1.errorbar(xValues, JSdiverg_meanAllSubj, yerr=JSdiverg_stdAllSubj, fmt='k', label='JS')
ax1.legend()
ax1.set_xlabel('Feature')
ax1.set_ylabel('Divergence')
ax1.set_title('Kullback Leibler divergence')
ax1.grid()
ax1 = fig1.add_subplot(gs[1,0])
ax1.errorbar(xValues, JSdiverg_meanAllSubj, yerr=JSdiverg_stdAllSubj, fmt='m', label='JS')
ax1.legend()
ax1.set_xlabel('Feature')
ax1.set_ylabel('Divergence')
ax1.set_title('Jensen Shannon divergence')
ax1.grid()
fig1.show()
fig1.savefig(folderDiverg + 'AllSubj_DifferentDivergenceMeasures_avrgAllSubj.png', bbox_inches='tight')
plt.close(fig1)

#PLOTTING ONLY NSS WITH BOXPLOTS
fig1 = plt.figure(figsize=(16, 16), constrained_layout=False)
gs = GridSpec(2, 1, figure=fig1)
fig1.subplots_adjust(wspace=0.4, hspace=0.2)
#fig1.suptitle('Feature ')
xValues = np.arange(0, FeaturesParams.numStandardFeat, 1)
ax1 = fig1.add_subplot(gs[0,0])
ax1.boxplot(KLdiverg_NSS_meanForCh, medianprops=dict(color='red', linewidth=2),
            boxprops=dict(linewidth=2), capprops=dict(linewidth=2), whiskerprops=dict(linewidth=2), showfliers=False)
ax1.set_xlabel('Feature')
ax1.set_xticks(np.arange(1, FeaturesParams.numStandardFeat+1, 1))
ax1.set_xticklabels(FeatNames, fontsize=10, rotation=45, ha='right', rotation_mode='anchor')
ax1.set_ylabel('Divergence')
ax1.set_title('Kullback Leibler divergence')
ax1.grid()
fig1.show()
fig1.savefig(folderDiverg + 'AllSubj_KLdiverg_NSS_avrgAllSubj_BoxplotAllCh.png', bbox_inches='tight')
fig1.savefig(folderDiverg + 'AllSubj_KLdiverg_NSS_avrgAllSubj_BoxplotAllCh.svg', bbox_inches='tight')
plt.close(fig1)



############################################################################################
## CALCULATE THE BEST ONES FEATURES BASED ON KL DIVERGENCE AND SORTING THEM
featNames=np.array(['meanAmpl', 'LineLength', 'samp_1_d7_1', 'samp_1_d6_1', 'samp_2_d7_1', 'samp_2_d6_1', 'perm_d7_3', 'perm_d7_5', 'perm_d7_7', 'perm_d6_3', 'perm_d6_5', 'perm_d6_7',   'perm_d5_3', 'perm_d5_5',  'perm_d5_7', 'perm_d4_3', 'perm_d4_5', 'perm_d4_7', 'perm_d3_3', 'perm_d3_5', 'perm_d3_7',
           'shannon_en_sig', 'renyi_en_sig', 'tsallis_en_sig', 'shannon_en_d7', 'renyi_en_d7', 'tsallis_en_d7', 'shannon_en_d6', 'renyi_en_d6', 'tsallis_en_d6', 'shannon_en_d5', 'renyi_en_d5', 'tsallis_en_d5', 'shannon_en_d4', 'renyi_en_d4', 'tsallis_en_d4', 'shannon_en_d3', 'renyi_en_d3', 'tsallis_en_d3',
           'p_dc_rel', 'p_mov_rel', 'p_delta_rel', 'p_theta_rel', 'p_alfa_rel', 'p_middle_rel', 'p_beta_rel', 'p_gamma_rel', 'p_dc', 'p_mov', 'p_delta', 'p_theta', 'p_alfa', 'p_middle', 'p_beta', 'p_gamma', 'p_tot',
           'zeroCrossNoApprox', 'ZC_approx_16', 'ZC_approx_32', 'ZC_approx_64', 'ZC_approx_128', 'ZC_approx_256' ])
indexSorted=np.argsort(-JSdiverg_meanAllSubj)
ranking=np.arange(1,len(indexSorted)+1,1)
dataAppend=np.vstack((ranking, indexSorted, featNames[indexSorted],JSdiverg_meanAllSubj[indexSorted])).transpose()
FeaturesSorted_JS = pd.DataFrame(dataAppend, columns=['Ranking', 'FeatIndex', 'FeatName', 'JS_divergence'])
fileOutputName=folderDiverg + '/AllSubj_FeaturesSorted_JS.csv'
FeaturesSorted_JS.to_csv(fileOutputName)

indexSorted=np.argsort(-KLdiverg_NSS_meanAllSubj)
ranking=np.arange(1,len(indexSorted)+1,1)
dataAppend=np.vstack((ranking, indexSorted, featNames[indexSorted],KLdiverg_NSS_meanAllSubj[indexSorted])).transpose()
FeaturesSorted_JS = pd.DataFrame(dataAppend, columns=['Ranking', 'FeatIndex', 'FeatName', 'KL_NSS_divergence'])
fileOutputName=folderDiverg + '/AllSubj_FeaturesSorted_KL_NSS.csv'
FeaturesSorted_JS.to_csv(fileOutputName)


#--------------------------------------------------------------------------------------------------------------------
# PLOT FOR PAPER
numFeat =FeaturesParams.numStandardFeat
font_size=14
fig1, ax1 = plt.subplots(figsize=(14, 7))

fig1.canvas.set_window_title('KL divergenge')
fig1.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

bp = ax1.boxplot(KLdiverg_NSS_meanForCh, medianprops=dict(color='k', linewidth=1), boxprops=dict(linewidth=0.5, color='k'), 
            capprops=dict(linewidth=1), whiskerprops=dict(linewidth=1), showfliers=False, showcaps=False)

box_colors = ['lightcoral','#FFA07A', 'papayawhip', '#BDFCC9', 'mediumseagreen', 'olive', '#33A1C9'] #mint=BDFCC9, peacock = 33A1C9
num_colors = len(box_colors)
num_boxes = [2,4,15,18,8,9,6] #number of features per feature type

medians = np.empty(numFeat)
n_used_box=0
for l in range(num_colors):
    for i in range(num_boxes[l]):
        box = bp['boxes'][i+n_used_box]
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        box_coords = np.column_stack([boxX, boxY])
        # Alternate colors
        ax1.add_patch(Polygon(box_coords, facecolor=box_colors[l]))

    n_used_box = n_used_box + num_boxes[l]

colorLegends = ['Time domain', 'Sample Entropy', 'Permutation Entropy', 'Shan_Ren_Tsa Entropy', 'Relative Power', 'Bandpower', 'AZC']
ax1.legend(colorLegends,fontsize=font_size-2)
            
ax1.set_xlabel('Features', fontsize=font_size+1)
# ax1.set_xticks(np.arange(1, FeaturesParams.numStandardFeat+1, 1))
ax1.set_xticks([])
ax1.tick_params(axis='y', which='major', labelsize=font_size-1)
# ax1.set_axisbelow(True)
# ax1.set_xticklabels(FeatNames, fontsize=10, rotation=45, ha='right', rotation_mode='anchor')
ax1.set_ylabel('KL Scores', fontsize=font_size+1)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)
# ax1.set_title('Kullback Leibler divergence')
ax1.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)
fig1.show()
fig1.savefig(folderDiverg + '/'+ Dataset+ '_AllSubj_KLdiverg_NSS_avrgAllSubj_BoxplotAllCh.svg', bbox_inches='tight', dpi=200, format='svg')
plt.close(fig1)




