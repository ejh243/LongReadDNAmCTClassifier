
import utils
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# accuracy threshold
thres = 0.8

## process command line information
resultsPath = sys.argv[1]
nCT  = sys.argv[2]

os.chdir(resultsPath)


## create output folder for plots
if not os.path.exists("Plots"):
    os.makedirs("Plots")

## list files in results folder
allFiles = os.listdir()
allFiles = list(filter(lambda x:'BinaryClassifier' in x, allFiles))

allDat = {}

## load results
for modelType in ["KNN", "NBayes", "RandFor", "SVM"]:
    subFiles = list(filter(lambda x:modelType in x, allFiles))
    print(str(len(subFiles)) + " files found for model type " + modelType)
    allDat[modelType] = pd.concat([utils.loadResults(x,"binary", nCT) for x in subFiles]).sort_values(by=["Chr", "Position", "nCpG"])
    ## count
    print(str(len(allDat[modelType])) + " models loaded for model type " + modelType)
    ## add Density column
    allDat[modelType]["Density"] = allDat[modelType]['WindowSize']/allDat[modelType]['nCpG']
    ## calc overall accuracy
    sensCols = [col for col in allDat[modelType].columns if col.endswith('sensitivity')]
    allDat[modelType]["Accuracy"] = allDat[modelType][sensCols].mean(1)

## summarize the parameters of the models
allDat[modelType]['nCpG'].describe()
allDat[modelType]['WindowSize'].describe()
allDat[modelType]['Density'].describe()

minCpG = allDat[modelType].nCpG.min()
maxWindow = allDat[modelType].WindowSize.max()
maxDensity = allDat[modelType].Density.max()


## violinplot of accuracy statistics
fig2, ax1 = plt.subplots()
ax1.violinplot((allDat["KNN"]['Accuracy'],allDat["NBayes"]['Accuracy'],allDat["RandFor"]['Accuracy'],allDat["SVM"]['Accuracy']),showextrema=True, showmedians=True)
plt.xticks((1,2,3,4), ("KNN", "Naive Bayes", "Random Forest", "SVM"))
ax1.set(xlim = [0.5,4.5], ylim = [0,1], xlabel='Algorithm', ylabel='Mean accuracy across CV')
ax1.grid(True)
fig2.savefig("Plots/ViolinplotAccuracyBinaryClassifiers.png", dpi=150)

## boxplot sensitivity

fig3, axs = plt.subplots(2,2, sharex = 'all', sharey = 'all')
axs[0,0].violinplot((allDat["KNN"][sensCols]),showextrema=True, showmedians=True)
axs[0,1].violinplot((allDat["NBayes"][sensCols]),showextrema=True, showmedians=True)
axs[1,0].violinplot((allDat["RandFor"][sensCols]),showextrema=True, showmedians=True)
axs[1,1].violinplot((allDat["SVM"][sensCols]),showextrema=True, showmedians=True)

axs[0,0].set_title("KNN")
axs[0,1].set_title("NBayes")
axs[1,0].set_title("RandFor")
axs[1,1].set_title("SVM")

axs[0,0].grid(True)
axs[0,1].grid(True)
axs[1,0].grid(True)
axs[1,1].grid(True)

fig3.text(0.5, 0.04, 'Cell type', ha='center')
fig3.text(0.04, 0.5, 'Sensitivity', va='center', rotation='vertical')

fig3.savefig("Plots/ViolinplotSensitivityBinaryClassifiers.png", dpi=150)


## boxplot specificity
specCols = [col for col in allDat[modelType].columns if col.endswith('specificity')]
fig4, axs = plt.subplots(2,2, sharex = 'all', sharey = 'all')
axs[0,0].violinplot((allDat["KNN"][specCols]),showextrema=True, showmedians=True)
axs[0,0].set_title("KNN")
axs[0,1].violinplot((allDat["NBayes"][specCols]),showextrema=True, showmedians=True)
axs[0,1].set_title("NBayes")
axs[1,0].violinplot((allDat["RandFor"][specCols]),showextrema=True, showmedians=True)
axs[1,0].set_title("RandFor")
axs[1,1].violinplot((allDat["SVM"][specCols]),showextrema=True, showmedians=True)
axs[1,1].set_title("SVM")

axs[0,0].grid(True)
axs[0,1].grid(True)
axs[1,0].grid(True)
axs[1,1].grid(True)

fig4.text(0.5, 0.04, 'Cell type', ha='center')
fig4.text(0.04, 0.5, 'Specificity', va='center', rotation='vertical')

fig4.savefig("Plots/ViolinplotSpecificityBinaryClassifiers.png", dpi=150)

## compare ML algorithms
## good models need to be sensitivity and specific
## count cumulative sum of number of models with accuracy > x.
## do by cell type and model

accuracyBins = np.arange(0,1.01,0.05)

cumSumTotals = {}
for modelType in ["KNN", "NBayes", "RandFor", "SVM"]:
    cumSumTotals[modelType] = pd.DataFrame([utils.cumSumSensSpec(allDat[modelType], x,accuracyBins) for x in [["CT" + str(i+1) + "_sensitivity","CT" + str(i+1) + "_specificity"] for i in range(0,int(nCT))]]).transpose()


fig5, axs = plt.subplots(2,2, sharex = 'all', sharey = 'all')
for col in cumSumTotals['KNN'].columns:
    axs[0,0].plot(np.flip(np.arange(0,1,0.05)), cumSumTotals['KNN'][col], label='CT '+str(col+1))
    axs[0,1].plot(np.flip(np.arange(0,1,0.05)), cumSumTotals['NBayes'][col], label='CT '+str(col+1))
    axs[1,0].plot(np.flip(np.arange(0,1,0.05)), cumSumTotals['RandFor'][col], label='CT '+str(col+1))
    axs[1,1].plot(np.flip(np.arange(0,1,0.05)), cumSumTotals['SVM'][col], label='CT '+str(col+1))

y_vals = axs[0,0].get_yticks()
axs[0,0].yaxis.set_major_locator(mticker.FixedLocator(y_vals))
axs[0,0].set_yticklabels(['{:.0f}'.format(x / 1000) for x in y_vals])
axs[1,0].yaxis.set_major_locator(mticker.FixedLocator(y_vals))
axs[1,0].set_yticklabels(['{:.0f}'.format(x / 1000) for x in y_vals])

axs[0,0].set_title("KNN")
axs[0,1].set_title("NBayes")
axs[1,0].set_title("RandFor")
axs[1,1].set_title("SVM")

axs[0,0].legend()
axs[0,0].grid(True)
axs[0,1].grid(True)
axs[1,0].grid(True)
axs[1,1].grid(True)

fig5.text(0.5, 0.04, 'Sensivity & Specificity', ha='center')
fig5.text(0.04, 0.5, 'Number of models (x1000)', va='center', rotation='vertical')
fig5.savefig("Plots/LineGraphCumulativeSensitivitySpecificityBinaryClassifiers.png", dpi=150)



## summarise as a function of model characteristics
fig6, axs = plt.subplots(2,4, sharex = 'all', sharey = 'all')
groupedCpG = {}
modelIndex = 0
for modelType in ["KNN", "NBayes", "RandFor", "SVM"]:
    groupedCpG[modelType] = allDat[modelType].groupby(allDat[modelType].nCpG).mean() 
    for ctNum in np.arange(int(nCT))+1:
        axs[0,modelIndex].plot(np.arange(minCpG,max(allDat[modelType]['nCpG'])+1,1), 
        np.asarray(groupedCpG[modelType]['CT' + str(ctNum) + '_sensitivity']), label='CT  '+ str(ctNum))
        axs[1,modelIndex].plot(np.arange(minCpG,max(allDat[modelType]['nCpG'])+1,1), 
        np.asarray(groupedCpG[modelType]['CT' + str(ctNum) + '_specificity']), label='CT ' + str(ctNum))
        axs[0,modelIndex].grid(True)
        axs[1,modelIndex].grid(True)
         
    modelIndex+=1 

axs[0,0].set_ylabel('Mean sensitivity')  
axs[1,0].set_ylabel('Mean specificity') 

axs[0,0].set_title("KNN")
axs[0,1].set_title("NBayes")
axs[0,2].set_title("RandFor")
axs[0,3].set_title("SVM")

#axs[0,0].legend()

fig6.text(0.5, 0.04, 'Number of CpGs', ha='center')
fig6.set_size_inches(20, 10)
fig6.savefig("Plots/LineGraphMeanSensitivitySpecificitynCpGBinaryClassifiers.png", dpi=150)



## plot against window size
spanBins = np.arange(0,20001,100)
windowBins = pd.cut(allDat["KNN"]['WindowSize'], bins = spanBins)

fig7, axs = plt.subplots(2,4, sharex = 'all', sharey = 'all')
groupedSpan = {}
modelIndex = 0
for modelType in ["KNN", "NBayes", "RandFor", "SVM"]:
    groupedSpan[modelType] = allDat[modelType].groupby(windowBins).mean() 
    for ctNum in np.arange(int(nCT))+1:
        axs[0,modelIndex].plot(spanBins[:-1], 
        np.asarray(groupedSpan[modelType]['CT' + str(ctNum) + '_sensitivity']), label='CT  '+ str(ctNum))
        axs[1,modelIndex].plot(spanBins[:-1], 
        np.asarray(groupedSpan[modelType]['CT' + str(ctNum) + '_specificity']), label='CT ' + str(ctNum))
        axs[0,modelIndex].grid(True)
        axs[1,modelIndex].grid(True)
         
    modelIndex+=1 

axs[0,0].set_ylabel('Mean sensitivity')  
axs[1,0].set_ylabel('Mean specificity') 

axs[0,0].set_title("KNN")
axs[0,1].set_title("NBayes")
axs[0,2].set_title("RandFor")
axs[0,3].set_title("SVM")

#axs[0,0].legend()

fig7.text(0.5, 0.04, 'Span of CpGs', ha='center')
fig7.set_size_inches(20, 10)
fig7.savefig("Plots/LineGraphMeanSensitivitySpecificityWindowSpanBinaryClassifiers.png", dpi=150)
 

## plot against density
densityBreaks = np.arange(0,maxDensity+50,50)
densityBins = pd.cut(allDat["KNN"]['Density'], bins = densityBreaks)


fig8, axs = plt.subplots(2,4, sharex = 'all', sharey = 'all')
groupedDensity = {}
modelIndex = 0
for modelType in ["KNN", "NBayes", "RandFor", "SVM"]:
    groupedDensity[modelType] = allDat[modelType].groupby(densityBins).mean() 
    for ctNum in np.arange(int(nCT))+1:
        axs[0,modelIndex].plot(densityBreaks[:-1], 
        np.asarray(groupedDensity[modelType]['CT' + str(ctNum) + '_sensitivity']), label='CT  '+ str(ctNum))
        axs[1,modelIndex].plot(densityBreaks[:-1], 
        np.asarray(groupedDensity[modelType]['CT' + str(ctNum) + '_specificity']), label='CT ' + str(ctNum))
        axs[0,modelIndex].grid(True)
        axs[1,modelIndex].grid(True)
         
    modelIndex+=1 

axs[0,0].set_ylabel('Mean sensitivity')  
axs[1,0].set_ylabel('Mean specificity') 

axs[0,0].set_title("KNN")
axs[0,1].set_title("NBayes")
axs[0,2].set_title("RandFor")
axs[0,3].set_title("SVM")

#axs[0,0].legend()

fig8.text(0.5, 0.04, 'Density of CpGs', ha='center')
fig8.set_size_inches(20, 10)
fig8.savefig("Plots/LineGraphMeanSensitivitySpecificityDensityBinaryClassifiers.png", dpi=150)
 