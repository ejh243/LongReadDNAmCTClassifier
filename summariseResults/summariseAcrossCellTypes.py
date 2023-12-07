## For one algorithm plot all binary models together

import utils
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# accuracy threshold
thres = 0.9

## process command line information
resultsPath  = sys.argv[1]
cellTypes = sys.argv[2:]

nCT = len(cellTypes)/2

os.chdir(resultsPath)

## create output folder for plots
if not os.path.exists("Plots"):
    os.makedirs("Plots")

## list files in results folder

allDat = {}
modelOpts = ["KNN", "NBayes", "RandFor", "SVM"]


## load results
for ct,nLevels in zip(cellTypes[::2], cellTypes[1::2]):
    modelOptsFilt = []
    allFiles = os.listdir(ct)
    allFiles = list(filter(lambda x:'BinaryClassifier' in x, allFiles))
    tmpDat = {}
    for modelType in modelOpts:
        subFiles = list(filter(lambda x:modelType in x, allFiles))
        print(str(len(subFiles)) + " files found for model type " + modelType)
        if(len(subFiles) == 22):
            modelOptsFilt = modelOptsFilt + [modelType]
            tmpDat[modelType] = pd.concat([utils.loadResults(os.path.join(ct,x),"binary", nLevels) for x in subFiles]).sort_values(by=["Chr", "Position", "nCpG"])
            ## count
            print(str(len(tmpDat[modelType])) + " models loaded for model type " + modelType)  
            ## add Density column
            tmpDat[modelType]["Density"] = tmpDat[modelType]['WindowSize']/tmpDat[modelType]['nCpG']
            ## calc overall accuracy
            sensCols = [col for col in tmpDat[modelType].columns if col.endswith('sensitivity')]
            tmpDat[modelType]["Accuracy"] = tmpDat[modelType][sensCols].mean(1)
            
    ## merge into a single data.frame to determine best algorithm for each model
    mergeDf = pd.concat([tmpDat[x]["Accuracy"] for x in modelOptsFilt], axis = 1)
    mergeDf.columns = modelOptsFilt
    ## identify for each model the best ML algorithm
    maxMean = mergeDf.max(1)
    allDat[ct] = maxMean

for ct in cellTypes[::2]:
    print(str("Summary for cell type " + ct))
    allDat[ct].describe()

minCpG = tmpDat[modelType].nCpG.min()
maxWindow = tmpDat[modelType].WindowSize.max()
maxDensity = tmpDat[modelType].Density.max()


## violinplot of accuracy statistics
fig2, ax1 = plt.subplots()
ax1.violinplot(pd.concat([allDat[x] for x in cellTypes[::2]], axis = 1, names = allDat.keys()),showextrema=True, showmedians=True)
plt.xticks(np.array(range(len(allDat)))+1, allDat.keys())
ax1.set(xlim = [0.5,nCT+0.5], ylim = [0,1], xlabel='Cell type', ylabel='Mean accuracy across CV')
ax1.grid(True)
fig2.savefig("Plots/ViolinplotAccuracyBinaryClassifiersAcrossCellTypes.png", dpi=150)


## count the number of CTs that can be predicted from each set of CpGs
cellTypes[::2][cellTypes[1::2] == 2]
countCT = pd.DataFrame(allDat).min(axis=1) > thres
)
fig3, axs = plt.subplots()
x = np.arange(nCT+1)  # the label locations
width = 0.8  # the width of the bars
axs.bar(x, countCT.sum(axis = 1).value_counts(sort=False).values, width)

# Add some text for labels, title and custom x-axis tick labels, etc.
axs.set_ylabel('Number of models')
axs.set_xlabel('Number of cell types')
axs.set_xticks(x)
fig3.savefig("Plots/BarchartNumberofCelltypesPredictedBinary" + modelType + "Classifiers.png", dpi=150)

## count cumulative sum of number of models with accuracy > x.

accuracyBins = np.arange(0,1.01,0.02)
	
allDat = pd.DataFrame.from_dict(allDat)
    
cumSumTotals = allDat.apply(utils.cumSumAccuracy, axis=0, bins = accuracyBins)


fig3, ax1 = plt.subplots()
for each in cellTypes[::2]:
    ax1.plot(np.flip(np.arange(0,1,0.02)), cumSumTotals[each], label = each)


ax1.set_xlabel('Mean accuracy across CV')  # Add an x-label to the axes.
ax1.set_ylabel('Number of classifiers(x1000)')  # Add a y-label to the axes.
y_vals = ax1.get_yticks()
ax1.yaxis.set_major_locator(mticker.FixedLocator(y_vals))
ax1.set_yticklabels(['{:.0f}'.format(x / 1000) for x in y_vals])
ax1.legend()
ax1.grid(True)
fig3.savefig("Plots/LineGraphCumulativeAccuracyBinaryClassifiersAcrossCellTypes.png", dpi=150)


## summarise as a function of model characteristics
fig5, (ax1,ax2, ax3) = plt.subplots(1,3)

groupedCpG = allDat.groupby(tmpDat[modelOptsFilt[0]].nCpG).mean()
minCpG = tmpDat[modelOptsFilt[0]].nCpG.min()
maxCpG = max(tmpDat[modelOptsFilt[0]].nCpG)+1
for each in cellTypes[::2]:
    ax1.plot(np.arange(minCpG,maxCpG,1), np.asarray(groupedCpG[each]), label = each)
ax1.legend()
ax1.set_ylabel('Mean accuracy')  
ax1.set_xlabel('Number of CpGs')  
ax1.grid(True)

## plot against window size
maxWindow = tmpDat[modelOptsFilt[0]].WindowSize.max()
spanBins = np.arange(0,maxWindow+100,100)
windowBins = pd.cut(tmpDat[modelOptsFilt[0]]['WindowSize'], bins = spanBins)

groupedSpan = allDat.groupby(windowBins).mean()
for each in cellTypes[::2]:
    ax2.plot(np.arange(100,maxWindow+100,100), np.asarray(groupedSpan[each]), label = each)

ax2.set_xlabel('Span of CpGs')  
ax2.grid(True)

## plot against density
tmpDat[modelOptsFilt[0]]["Density"] = tmpDat[modelOptsFilt[0]]['WindowSize']/tmpDat[modelOptsFilt[0]]['nCpG']
maxDensity = tmpDat[modelOptsFilt[0]].Density.max()
densityBreaks = np.arange(0,maxDensity+50,50)
densityBins = pd.cut(tmpDat[modelOptsFilt[0]]['Density'], bins = densityBreaks)

groupedDensity = allDat.groupby(densityBins).mean()
for each in cellTypes[::2]:
    ax3.plot(densityBreaks[1:], np.asarray(groupedDensity[each]), label = each)

ax3.set_xlabel('Density of CpGs')  
ax3.grid(True)
fig5.set_size_inches(12, 4)
fig5.savefig("Plots/LineGraphAccuracyAgainstModelPropertiesAcrossCellTypesBinaryClassifiers.png", dpi=150)


# summarise model properties as a function of accuracy
accuracyBreaks = np.arange(0.5,1.01,0.01)
fig6, (ax1,ax2, ax3) = plt.subplots(1,3)

for each in cellTypes[::2]:
    accuracyBins = pd.cut(allDat[each], bins = accuracyBreaks)
    groupedAccuracy = tmpDat[modelOptsFilt[0]].groupby(accuracyBins).mean()
    ax1.plot(accuracyBreaks[:-1]+0.005, np.asarray(groupedAccuracy['nCpG']), label = each)
    ax2.plot(accuracyBreaks[:-1]+0.005, np.asarray(groupedAccuracy['WindowSize']), label = each)
    ax3.plot(accuracyBreaks[:-1]+0.005, np.asarray(groupedAccuracy['Density']), label = each)
    
    groupedAccuracy.to_csv("GroupedAccuracySummaryStatistics" + each + ".csv")


ax1.set_ylabel('Number of CpGs')  
ax1.set_xlabel('Mean accuracy')  
ax1.grid(True)
ax2.set_ylabel('Span of CpGs (bp)')  
ax2.set_xlabel('Mean accuracy')  
ax2.grid(True)

ax3.set_ylabel('Density of CpGs (bp)')  
ax3.set_xlabel('Mean accuracy')  
ax3.grid(True)
ax3.legend()

plt.subplots_adjust(left=0.05,
                    bottom=0.12, 
                    right=0.95, 
                    top=0.95, 
                    wspace=0.3, 
                    hspace=0.4)
fig6.set_size_inches(12, 4)
fig6.savefig("Plots/LineGraphModelPropertiesAgainstAccuracyAcrossCellTypesBinaryClassifiers.png", dpi=150)


## cumulative number of specific and sensitive models
accuracyBins = np.arange(0,1.01,0.05)

cumSumTotals = pd.DataFrame([utils.cumSumSensSpec(allDat[x], ["CT1_sensitivity","CT1_specificity"],accuracyBins) for x in cellTypes[::2]]).transpose()

fig5, axs = plt.subplots()
for col in cumSumTotals.columns:
    axs.plot(np.flip(np.arange(0,1,0.05)), cumSumTotals[col], label=cellTypes[::2][col])


axs.legend()
axs.grid(True)
axs.set_xlabel('Mean accuracy across CV')  # Add an x-label to the axes.
axs.set_ylabel('Number of models (x1000)')  # Add a y-label to the axes.
y_vals = axs.get_yticks()
axs.yaxis.set_major_locator(mticker.FixedLocator(y_vals))
axs.set_yticklabels(['{:.0f}'.format(x / 1000) for x in y_vals])
fig5.savefig("Plots/LineGraphCumulativeSensitivitySpecificityBinary" + modelType + "Classifiers.png", dpi=150)

## function of model properties
fig6, axs = plt.subplots(1,3, sharex = 'all', sharey = 'all')
groupedCpG = {}
modelIndex = 0
for ct in cellTypes[::2]:
    groupedCpG[ct] = allDat[ct].groupby(allDat[ct].nCpG).mean() 
    axs[0].plot(np.arange(minCpG,max(allDat[ct]['nCpG'])+1,1), 
        np.asarray(groupedCpG[ct]['CT1_sensitivity']), label=ct)
    axs[1].plot(np.arange(minCpG,max(allDat[ct]['nCpG'])+1,1), 
        np.asarray(groupedCpG[ct]['CT1_specificity']), label=ct)
    axs[2].plot(np.arange(minCpG,max(allDat[ct]['nCpG'])+1,1), 
        np.asarray(groupedCpG[ct]['Accuracy']), label=ct)       

        
axs[0].grid(True)
axs[1].grid(True)
axs[2].grid(True)

axs[0].set_ylabel('Mean sensitivity')  
axs[1].set_ylabel('Mean specificity') 
axs[2].set_ylabel('Mean Accuracy') 

axs[0].legend()

fig6.text(0.5, 0.04, 'Number of CpGs', ha='center')
fig6.set_size_inches(20, 8)
fig6.savefig("Plots/LineGraphMeanSensitivitySpecificitynCpGBinary" + modelType + "Classifiers.png", dpi=150)



## plot against window size
spanBins = np.arange(0,20001,100)
windowBins = pd.cut(allDat[cellTypes[0]]['WindowSize'], bins = spanBins)

fig7, axs = plt.subplots(1,3, sharex = 'all', sharey = 'all')
groupedSpan = {}
modelIndex = 0
for ct in cellTypes:
    groupedSpan[ct] = allDat[ct].groupby(windowBins).mean() 
    axs[0].plot(spanBins[:-1], 
        np.asarray(groupedSpan[ct]['CT1_sensitivity']), label=ct)
    axs[1].plot(spanBins[:-1], 
        np.asarray(groupedSpan[ct]['CT1_specificity']), label=ct)
    axs[2].plot(spanBins[:-1], 
        np.asarray(groupedSpan[ct]['Accuracy']), label=ct)

axs[0].grid(True)
axs[1].grid(True)
axs[2].grid(True)

axs[0].set_ylabel('Mean sensitivity')  
axs[1].set_ylabel('Mean specificity') 
axs[2].set_ylabel('Mean Accuracy') 

axs[0].legend()

fig7.text(0.5, 0.04, 'Span of CpGs', ha='center')
fig7.set_size_inches(20, 8)
fig7.savefig("Plots/LineGraphMeanSensitivitySpecificityWindowSpanBinary" + modelType + "Classifiers.png", dpi=150)
 

## plot against density
densityBreaks = np.arange(0,maxDensity+50,50)
densityBins = pd.cut(allDat[cellTypes[0]]['Density'], bins = densityBreaks)


fig8, axs = plt.subplots(1,3, sharex = 'all', sharey = 'all')
groupedDensity = {}
modelIndex = 0
for ct in cellTypes:
    groupedDensity[ct] = allDat[ct].groupby(densityBins).mean() 
    axs[0].plot(densityBreaks[:-1], 
        np.asarray(groupedDensity[ct]['CT1_sensitivity']), label=ct)
    axs[1].plot(densityBreaks[:-1], 
        np.asarray(groupedDensity[ct]['CT1_specificity']), label=ct)
    axs[2].plot(densityBreaks[:-1], 
        np.asarray(groupedDensity[ct]['Accuracy']), label=ct)


axs[0].grid(True)
axs[1].grid(True)
axs[2].grid(True)

axs[0].set_ylabel('Mean sensitivity')  
axs[1].set_ylabel('Mean specificity') 
axs[2].set_ylabel('Mean Accuracy') 

axs[0].legend()

fig8.text(0.5, 0.04, 'Density of CpGs', ha='center')
fig8.set_size_inches(20, 8)
fig8.savefig("Plots/LineGraphMeanSensitivitySpecificityDensityBinary" + modelType + "Classifiers.png", dpi=150)
 