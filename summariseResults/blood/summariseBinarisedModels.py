
def cumSumSensitivity(data, bins):
	"function to calculate cumulative number of models at different thresholds"
	return np.asarray(np.cumsum(np.flip(pd.cut(data, bins = bins).value_counts(sort = False))))

def cumSumBivariate(data1, data2, bins):
    out = np.empty(bins.shape[0]-1)
    for lBound in bins:
        sum(data1 >= lBound & data2 >= lBound)

def gaps(granges):
    ## doesn't make sens unless ranges contains non-overlapping regions
    ## get chromosome size info
    chrSizes = pr.data.chromsizes()
    chrList = np.unique(granges.Chromosome)
    newChr = []
    newStart = []
    newEnd = []
    for chr in chrList:
        boolIndex = granges.Chromosome == chr
        newChr.extend([chr]*(sum(boolIndex)+1)) ## need an extra chr
        newStart.extend([0])
        newStart.extend((granges.End[boolIndex].values+1).tolist())
        newEnd.extend((granges.Start[boolIndex].values - 1).tolist())
        newEnd.extend(chrSizes.End[chrSizes.Chromosome == chr].tolist())
    return pr.PyRanges(chromosomes = newChr, starts = newStart, ends = newEnd)

def cumSumSensSpec(data1,data2,bins1,thres2):
    return pd.DataFrame(np.array(list(map(lambda x: cumSumSensitivity(data1[data2 > x], accuracyBins), thres2)))).T

def plotCumSum(axs, cumMatrix, accuracyBins, axRow, axCol, color, specThres, linestyle):
    for a,b in zip(np.arange(len(specThres)),specThres):
        axs[axRow,axCol].plot(np.flip(accuracyBins[:-1]), cumMatrix[a], linestyle=linestyle[a], label = "> " + str(b), color = color)
    return(axs)

def multiplot(cumMatBCells, cumMatCD4T, cumMatCD8T, cumMatGran, cumMatMono, axs,accuracyBins, specThres, color, linestyle):
    axs = plotCumSum(axs, cumMatBCells, accuracyBins, 0,0,color, specThres, linestyle)
    axs[0,0].set_title('B-cells')
    axs[0,0].set_ylabel('Number of models')  # Add a y-label to the axes.
    axs[1,2].legend(labels = list(map(lambda x: "> " + str(x), specThres)))
    axs[0,0].grid(True)
    axs = plotCumSum(axs, cumMatCD4T, accuracyBins, 0,1,color, specThres, linestyle)
    axs[0,1].set_title('CD4+ T-cells')
    axs[0,1].grid(True)
    axs = plotCumSum(axs, cumMatCD8T, accuracyBins, 0,2,color, specThres, linestyle)
    axs[0,2].set_title('CD8+ T-cells')
    axs[0,2].grid(True)
    axs = plotCumSum(axs, cumMatGran, accuracyBins, 1,0,color, specThres, linestyle)
    axs[1,0].set_title('Granulocytes')
    axs[1,0].set_xlabel('Sensitivity')  # Add an x-label to the axes.
    axs[1,0].set_ylabel('Number of models')  # Add a y-label to the axes.
    axs[1,0].grid(True)
    axs = plotCumSum(axs, cumMatMono, accuracyBins, 1,1,color, specThres, linestyle)
    axs[1,1].set_title('Monocytes')
    axs[1,1].set_xlabel('Sensitivity')  # Add an x-label to the axes.
    axs[1,1].grid(True)
    return(axs)

def identifyPredModels(pDat, thres, ctLabels):
    ## returns matrix ndicating which models have  sensitivity and specificity > user specified threshold
    iMat =  pd.DataFrame(list(map(lambda y: pDat[list(map(lambda x: x + y, ["Sensitivity", "Specificity"]))].min(axis = 1) > thres, ctLabels))).T
    iMat.columns = ctLabels
    return iMat
    


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyranges as pr
import seaborn as sns
from upsetplot import from_contents
from upsetplot import UpSet

os.chdir("/mnt/data1/Eilis/Projects/Asthma/ClassifyCellTypes/")

chr = 22

## set colors for plotting
colors = ["royalblue", "darkorange", "forestgreen", "darkred", "darkorchid"]

colNames = ("CHR", "MAPINFO", "nCpG", "windowSize", "SensitivityBcells", "SensitivityCD4T","SensitivityCD8T","SensitivityGran","SensitivityMono", "SpecificityBcells", "SpecificityCD4T","SpecificityCD8T","SpecificityGran","SpecificityMono")

svm = pd.read_csv("Results/SVMBinarizedChr" + str(chr) + "_100obsCT.csv", header = 0, names = colNames)
nBayes = pd.read_csv("Results/NaiveBayesBernoulliBinarizedChr" + str(chr) + "_100obsCT.csv", header = 0, names = colNames)
knn = pd.read_csv("Results/KNNBinarizedChr" + str(chr) + "_100obsCT.csv", header = 0, names = colNames)
randFor = pd.read_csv("Results/RandomForestBinarizedChr" + str(chr) + "_100obsCT.csv", header = 0, names = colNames)

## calculate overall accuracy
sensCols = [col for col in colNames if col.startswith('Sensitivity')]

svm['Accuracy'] = svm[sensCols].mean(1)
knn['Accuracy'] = knn[sensCols].mean(1)
nBayes['Accuracy'] = nBayes[sensCols].mean(1)
randFor['Accuracy'] = randFor[sensCols].mean(1)

## merge into a single data.frame to determine best algorithm for each model

mergeBcells = pd.DataFrame({'svm': svm["SensitivityBcells"], 'knn': knn["SensitivityBcells"], 'naiveBayes': nBayes["SensitivityBcells"], 'randomForest': randFor["SensitivityBcells"]})
mergeCD4T = pd.DataFrame({'svm': svm["SensitivityCD4T"], 'knn': knn["SensitivityCD4T"], 'naiveBayes': nBayes["SensitivityCD4T"], 'randomForest': randFor["SensitivityCD4T"]})
mergeCD8T = pd.DataFrame({'svm': svm["SensitivityCD8T"], 'knn': knn["SensitivityCD8T"], 'naiveBayes': nBayes["SensitivityCD8T"], 'randomForest': randFor["SensitivityCD8T"]})
mergeMono = pd.DataFrame({'svm': svm["SensitivityMono"], 'knn': knn["SensitivityMono"], 'naiveBayes': nBayes["SensitivityMono"], 'randomForest': randFor["SensitivityMono"]})
mergeGran = pd.DataFrame({'svm': svm["SensitivityGran"], 'knn': knn["SensitivityGran"], 'naiveBayes': nBayes["SensitivityGran"], 'randomForest': randFor["SensitivityGran"]})
mergeOverall = pd.DataFrame({'svm': svm["Accuracy"], 'knn': knn["Accuracy"], 'naiveBayes': nBayes["Accuracy"], 'randomForest': randFor["Accuracy"]})


## calculate the best accuracy score across models
mergeBcells['best'] = mergeBcells.max(1)
mergeCD4T['best'] = mergeCD4T.max(1)
mergeCD8T['best'] = mergeCD8T.max(1)
mergeMono['best'] = mergeMono.max(1)
mergeGran['best'] = mergeGran.max(1)
mergeOverall['best'] = mergeOverall.max(1)


## boxplot of accuracy statistics
fig1, axs = plt.subplots(2,3, sharex = 'all', sharey = 'all')
bplot00 = axs[0,0].boxplot((mergeOverall['knn'],mergeOverall['naiveBayes'],mergeOverall['randomForest'],mergeOverall['svm'], mergeOverall['best']), labels = ("KNN", "Naive Bayes", "Random Forest", "SVM", "Best"), patch_artist=True)
axs[0,0].grid(True)
axs[0,0].set_title("Overall")

bplot01 = axs[0,1].boxplot((mergeBcells['knn'],mergeBcells['naiveBayes'],mergeBcells['randomForest'],mergeBcells['svm'], mergeBcells['best']), labels = ("KNN", "Naive Bayes", "Random Forest", "SVM", "Best"), patch_artist=True)
axs[0,1].grid(True)
axs[0,1].set_title("B-cells")

bplot02 = axs[0,2].boxplot((mergeCD4T['knn'],mergeCD4T['naiveBayes'],mergeCD4T['randomForest'],mergeCD4T['svm'], mergeCD4T['best']), labels = ("KNN", "Naive Bayes", "Random Forest", "SVM", "Best"), patch_artist=True)
axs[0,2].grid(True)
axs[0,2].set_title("CD4+ T-cells")

bplot10 = axs[1,0].boxplot((mergeCD8T['knn'],mergeCD8T['naiveBayes'],mergeCD8T['randomForest'],mergeCD8T['svm'], mergeCD8T['best']), labels = ("KNN", "Naive Bayes", "Random Forest", "SVM", "Best"), patch_artist=True)
axs[1,0].grid(True)
axs[1,0].set_title("CD8+ T-cells")

bplot11 = axs[1,1].boxplot((mergeGran['knn'],mergeGran['naiveBayes'],mergeGran['randomForest'],mergeGran['svm'], mergeGran['best']), labels = ("KNN", "Naive Bayes", "Random Forest", "SVM", "Best"), patch_artist=True)
axs[1,1].grid(True)
axs[1,1].set_title("Granulocytes")

bplot12 = axs[1,2].boxplot((mergeMono['knn'],mergeMono['naiveBayes'],mergeMono['randomForest'],mergeMono['svm'], mergeMono['best']), labels = ("KNN", "Naive Bayes", "Random Forest", "SVM", "Best"), patch_artist=True)
axs[1,2].grid(True)
axs[1,2].set_title("Monocytes")

plt.setp(axs, xlabel='', ylabel='Sensitivity')

# fill with colors
for bplot in (bplot00, bplot01, bplot02, bplot10, bplot11, bplot12):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

fig1.set_size_inches(15, 10)
fig1.savefig("Results/Pilot/BoxplotSensitivityByCellTypeBinarisedModels.png", dpi=150)


## boxplot of specificity statistics

fig2, axs = plt.subplots(2,3, sharex = 'all', sharey = 'all')


bplot01 = axs[0,1].boxplot((knn['SpecificityBcells'], nBayes['SpecificityBcells'], randFor['SpecificityBcells'],svm['SpecificityBcells']), labels = ("KNN", "Naive Bayes", "Random Forest", "SVM"), patch_artist=True)
axs[0,1].grid(True)
axs[0,1].set_title("B-cells")

bplot02 = axs[0,2].boxplot((knn['SpecificityCD4T'], nBayes['SpecificityCD4T'], randFor['SpecificityCD4T'],svm['SpecificityCD4T']), labels = ("KNN", "Naive Bayes", "Random Forest", "SVM"), patch_artist=True)
axs[0,2].grid(True)
axs[0,2].set_title("CD4+ T-cells")

bplot10 = axs[1,0].boxplot((knn['SpecificityCD8T'], nBayes['SpecificityCD8T'], randFor['SpecificityCD8T'],svm['SpecificityCD8T']), labels = ("KNN", "Naive Bayes", "Random Forest", "SVM"), patch_artist=True)
axs[1,0].grid(True)
axs[1,0].set_title("CD8+ T-cells")

bplot11 = axs[1,1].boxplot((knn['SpecificityGran'], nBayes['SpecificityGran'], randFor['SpecificityGran'],svm['SpecificityGran']), labels = ("KNN", "Naive Bayes", "Random Forest", "SVM"), patch_artist=True)
axs[1,1].grid(True)
axs[1,1].set_title("Granulocytes")

bplot12 = axs[1,2].boxplot((knn['SpecificityMono'], nBayes['SpecificityMono'], randFor['SpecificityMono'],svm['SpecificityMono']), labels = ("KNN", "Naive Bayes", "Random Forest", "SVM"), patch_artist=True)
axs[1,2].grid(True)
axs[1,2].set_title("Monocytes")

plt.setp(axs, xlabel='', ylabel='Specificity')

# fill with colors
for bplot in (bplot01, bplot02, bplot10, bplot11, bplot12):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

fig2.set_size_inches(15, 10)
fig2.savefig("Results/Pilot/BoxplotSpecificityByCellTypeBinarisedModels.png", dpi=150)

## good models need to be sensitivity and specific
## count cumulative sum of number of models with accuracy > x.
## do by cell type and model

accuracyBins = np.arange(0,1.01,0.05)
specThres  = np.arange(0,1,0.2)

cumKNNBcells = cumSumSensSpec(knn['SensitivityBcells'], knn['SpecificityBcells'], accuracyBins, specThres)
cumNBayesBcells = cumSumSensSpec(nBayes['SensitivityBcells'], nBayes['SpecificityBcells'], accuracyBins, specThres)
cumRandForBcells = cumSumSensSpec(randFor['SensitivityBcells'], randFor['SpecificityBcells'], accuracyBins, specThres)
cumSVMBcells = cumSumSensSpec(svm['SensitivityBcells'], svm['SpecificityBcells'], accuracyBins, specThres)

cumKNNCD4T = cumSumSensSpec(knn['SensitivityCD4T'], knn['SpecificityCD4T'], accuracyBins, specThres)
cumNBayesCD4T = cumSumSensSpec(nBayes['SensitivityCD4T'], nBayes['SpecificityCD4T'], accuracyBins, specThres)
cumRandForCD4T = cumSumSensSpec(randFor['SensitivityCD4T'], randFor['SpecificityCD4T'], accuracyBins, specThres)
cumSVMCD4T = cumSumSensSpec(svm['SensitivityCD4T'], svm['SpecificityCD4T'], accuracyBins, specThres)

cumKNNCD8T = cumSumSensSpec(knn['SensitivityCD8T'], knn['SpecificityCD8T'], accuracyBins, specThres)
cumNBayesCD8T = cumSumSensSpec(nBayes['SensitivityCD8T'], nBayes['SpecificityCD8T'], accuracyBins, specThres)
cumRandForCD8T = cumSumSensSpec(randFor['SensitivityCD8T'], randFor['SpecificityCD8T'], accuracyBins, specThres)
cumSVMCD8T = cumSumSensSpec(svm['SensitivityCD8T'], svm['SpecificityCD8T'], accuracyBins, specThres)

cumKNNGran = cumSumSensSpec(knn['SensitivityGran'], knn['SpecificityGran'], accuracyBins, specThres)
cumNBayesGran = cumSumSensSpec(nBayes['SensitivityGran'], nBayes['SpecificityGran'], accuracyBins, specThres)
cumRandForGran = cumSumSensSpec(randFor['SensitivityGran'], randFor['SpecificityGran'], accuracyBins, specThres)
cumSVMGran = cumSumSensSpec(svm['SensitivityGran'], svm['SpecificityGran'], accuracyBins, specThres)

cumKNNMono = cumSumSensSpec(knn['SensitivityMono'], knn['SpecificityMono'], accuracyBins, specThres)
cumNBayesMono = cumSumSensSpec(nBayes['SensitivityMono'], nBayes['SpecificityMono'], accuracyBins, specThres)
cumRandForMono = cumSumSensSpec(randFor['SensitivityMono'], randFor['SpecificityMono'], accuracyBins, specThres)
cumSVMMono = cumSumSensSpec(svm['SensitivityMono'], svm['SpecificityMono'], accuracyBins, specThres)


linestyle = ["-", "--", "-.", ":", (0, (3, 1, 1, 1, 1, 1))]

fig3, axs = plt.subplots(2,3, sharex = 'all', sharey = 'all')
multiplot(cumKNNBcells, cumKNNCD4T, cumKNNCD8T, cumKNNGran, cumKNNMono, axs, accuracyBins, specThres, colors[0], linestyle)
fig3.set_size_inches(15, 10)
fig3.savefig("Results/Pilot/LineGraphCumulativeSensSpecByCellTypeKNNBinarisedModels.png", dpi=150)

fig4, axs = plt.subplots(2,3, sharex = 'all', sharey = 'all')
multiplot(cumNBayesBcells, cumNBayesCD4T, cumNBayesCD8T, cumNBayesGran, cumNBayesMono, axs, accuracyBins, specThres, colors[1], linestyle)
fig4.set_size_inches(15, 10)
fig4.savefig("Results/Pilot/LineGraphCumulativeSensSpecByCellTypeNBayesBinarisedModels.png", dpi=150)

fig5, axs = plt.subplots(2,3, sharex = 'all', sharey = 'all')
multiplot(cumRandForBcells, cumRandForCD4T, cumRandForCD8T, cumRandForGran, cumRandForMono, axs, accuracyBins, specThres, colors[1], linestyle)
fig5.set_size_inches(15, 10)
fig5.savefig("Results/Pilot/LineGraphCumulativeSensSpecByCellTypeRandForBinarisedModels.png", dpi=150)

fig6, axs = plt.subplots(2,3, sharex = 'all', sharey = 'all')
multiplot(cumSVMBcells, cumSVMCD4T, cumSVMCD8T, cumSVMGran, cumSVMMono, axs, accuracyBins, specThres, colors[1], linestyle)
fig6.set_size_inches(15, 10)
fig6.savefig("Results/Pilot/LineGraphCumulativeSensSpecByCellTypeSVMBinarisedModels.png", dpi=150)


## any evidence that models can predict multiple cell types simultaneously?
thres = 0.8
ctLabels = ["Bcells", "CD4T", "CD8T", "Gran", "Mono"]

iMatKNN = identifyPredModels(knn, thres, ctLabels)
nCTKNN = iMatKNN.sum(axis = 1).value_counts()

iMatNB = identifyPredModels(nBayes, thres, ctLabels)
nCTNB = iMatNB.sum(axis = 1).value_counts()

iMatRF = identifyPredModels(randFor, thres, ctLabels)
nCTRF = iMatRF.sum(axis = 1).value_counts()

iMatSVM = identifyPredModels(svm, thres, ctLabels)
nCTSVM = iMatSVM.sum(axis = 1).value_counts()

fig7, axs = plt.subplots()
x = np.arange(5)  # the label locations
width = 0.2  # the width of the bars
axs.bar((x - (2*width))[:len(nCTKNN)], nCTKNN, width, label='KNN')
axs.bar((x - (width))[:len(nCTNB)], nCTNB, width, label='Naive Bayes')
axs.bar(x[:len(nCTRF)], nCTRF, width, label='Random Forest')
axs.bar((x + (width))[:len(nCTSVM)], nCTSVM, width, label='SVM')


# Add some text for labels, title and custom x-axis tick labels, etc.
axs.set_ylabel('Number of models')
axs.set_title('Number of cell types predicted')
axs.set_xticks(x)
axs.legend()

fig7.savefig("Results/Pilot/BarplotCountPredictiveModels.png", dpi=150)

## create zoomed in version for models predictive in at least one cell type
plt.xlim([0.5, 2.5])
plt.ylim([0, 1500])
fig7.savefig("Results/Pilot/BarplotCountPredictiveModelsZoom.png", dpi=150)


## characterize good models 

## identify good models
ctPredKNN = knn[iMatKNN.sum(axis = 1) > 0]
ctPredNB = nBayes[iMatNB.sum(axis = 1) > 0]
ctPredRF = randFor[iMatRF.sum(axis = 1) > 0]
ctPredSVM = svm[iMatSVM.sum(axis = 1) > 0]

## which cell types are these predictive of?



## boxplot of accuracy statistics
fig8, axs = plt.subplots(2,3, sharex = 'all', sharey = 'all')
bplot00 = axs[0,0].boxplot((ctPredKNN['Accuracy'],ctPredNB['Accuracy'], ctPredRF['Accuracy'], ctPredSVM['Accuracy']), labels = ("KNN", "Naive Bayes", "Random Forest", "SVM"), patch_artist=True)
axs[0,0].grid(True)
axs[0,0].set_ylabel("Accuracy")
axs[0,0].set_title("Overall")

bplot01 = axs[0,1].boxplot((ctPredKNN['SensitivityBcells'],ctPredNB['SensitivityBcells'], ctPredRF['SensitivityBcells'], ctPredSVM['SensitivityBcells']), labels = ("KNN", "Naive Bayes", "Random Forest", "SVM"), patch_artist=True)
axs[0,1].grid(True)
axs[0,1].set_ylabel("Sensitivity")
axs[0,1].set_title("B-cells")

bplot02 = axs[0,2].boxplot((ctPredKNN['SensitivityCD4T'],ctPredNB['SensitivityCD4T'], ctPredRF['SensitivityCD4T'], ctPredSVM['SensitivityCD4T']), labels = ("KNN", "Naive Bayes", "Random Forest", "SVM"), patch_artist=True)
axs[0,2].grid(True)
axs[0,2].set_ylabel("Sensitivity")
axs[0,2].set_title("CD4 T-cells")

bplot10 = axs[1,0].boxplot((ctPredKNN['SensitivityCD8T'],ctPredNB['SensitivityCD8T'], ctPredRF['SensitivityCD8T'], ctPredSVM['SensitivityCD8T']), labels = ("KNN", "Naive Bayes", "Random Forest", "SVM"), patch_artist=True)
axs[1,0].grid(True)
axs[1,0].set_ylabel("Sensitivity")
axs[1,0].set_title("CD8 T-cells")

bplot11 = axs[1,1].boxplot((ctPredKNN['SensitivityGran'],ctPredNB['SensitivityGran'], ctPredRF['SensitivityGran'], ctPredSVM['SensitivityGran']), labels = ("KNN", "Naive Bayes", "Random Forest", "SVM"), patch_artist=True)
axs[1,1].grid(True)
axs[1,1].set_ylabel("Sensitivity")
axs[1,1].set_title("Granulocytes")

bplot12 = axs[1,2].boxplot((ctPredKNN['SensitivityMono'],ctPredNB['SensitivityMono'], ctPredRF['SensitivityMono'], ctPredSVM['SensitivityMono']), labels = ("KNN", "Naive Bayes", "Random Forest", "SVM"), patch_artist=True)
axs[1,2].grid(True)
axs[1,2].set_ylabel("Sensitivity")
axs[1,2].set_title("Monocytes")

# fill with colors
for bplot in (bplot00, bplot01, bplot02, bplot10, bplot11, bplot12):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

## identify which cell types

fig8, axs = plt.subplots(2, 2, figsize=(10, 10), sharey=False)
axs[0,0].bar(ctLabels, iMatKNN[iMatKNN.sum(axis = 1) > 0].sum(), color = colors[0])
axs[0,0].set_title("KNN")
axs[0,0].set_ylabel("Number of Models")

axs[0,1].bar(ctLabels, iMatNB[iMatNB.sum(axis = 1) > 0].sum(), color = colors[1])
axs[0,1].set_title("Naive Bayes")
axs[0,1].set_ylabel("Number of Models")

axs[1,0].bar(ctLabels, iMatRF[iMatRF.sum(axis = 1) > 0].sum(), color = colors[2])
axs[1,0].set_title("Random Forest")
axs[1,0].set_ylabel("Number of Models")

axs[1,1].bar(ctLabels, iMatSVM[iMatSVM.sum(axis = 1) > 0].sum(), color = colors[3])
axs[1,1].set_title("SVM")
axs[1,1].set_ylabel("Number of Models")


fig8.set_size_inches(15, 15)
fig8.savefig("Results/Pilot/BarplotWhichCTPredictiveModels.png", dpi=150)


## parameters of models
fig9, axs = plt.subplots(1,2)

bplot00 = axs[0].boxplot((ctPredKNN['nCpG'], ctPredNB['nCpG'], ctPredRF['nCpG'], ctPredSVM['nCpG']), labels = ("KNN", "Naive Bayes", "Random Forest", "SVM"), patch_artist=True)
axs[0].grid(True)
axs[0].set_ylabel('Number of CpGs')

bplot01 = axs[1].boxplot((ctPredKNN['windowSize'], ctPredNB['windowSize'], ctPredRF['windowSize'], ctPredSVM['windowSize']), labels = ("KNN", "Naive Bayes", "Random Forest", "SVM"), patch_artist=True)
axs[1].grid(True)
axs[1].set_ylabel('Span of CpGs')

# fill with colors
for bplot in (bplot00, bplot01):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

fig9.set_size_inches(15, 10)
fig9.savefig("Results/Pilot/BoxplotPredictiveModelsProperties.png", dpi=150)

## which models selected by multiple
modelIDKNN = ctPredKNN['MAPINFO'].astype(int).astype(str) + "_" + ctPredKNN['nCpG'].astype(int).astype(str)
modelIDNB = ctPredNB['MAPINFO'].astype(int).astype(str) + "_" + ctPredNB['nCpG'].astype(int).astype(str)
modelIDRF = ctPredRF['MAPINFO'].astype(int).astype(str) + "_" + ctPredRF['nCpG'].astype(int).astype(str)
modelIDSVM = ctPredSVM['MAPINFO'].astype(int).astype(str) + "_" + ctPredSVM['nCpG'].astype(int).astype(str)

models = from_contents({'KNN': modelIDKNN, 'Naive Bayes': modelIDNB, 'Random Forest': modelIDRF, 'SVM': modelIDSVM})

plot(models, subset_size='count')
plt.savefig("Results/Pilot/UpsetPlotPredictiveModelsZoom.png", dpi=150)

## for models that are predictive with multiple ML algorthms
## take KNN models and plot grn accuracy againt other models
plt.scatter(ctPredKNN['SensitivityGran'], nBayes[iMatKNN.sum(axis = 1) > 0]['SensitivityGran'])
ctPredNB = nBayes[iMatNB.sum(axis = 1) > 0]
ctPredRF = randFor[iMatRF.sum(axis = 1) > 0]
ctPredSVM = svm[iMatSVM.sum(axis = 1) > 0]

## how many unique regions

