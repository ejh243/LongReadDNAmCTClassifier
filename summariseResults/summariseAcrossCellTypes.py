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
modelType = sys.argv[1]
resultsPath  = sys.argv[2]
cellTypes = sys.argv[3:]

nCT = len(cellTypes)

os.chdir(resultsPath)

## create output folder for plots
if not os.path.exists("Plots"):
    os.makedirs("Plots")

## list files in results folder

allDat = {}

## load results
for ct in cellTypes:
    allFiles = os.listdir(ct)
    allFiles = list(filter(lambda x:'BinaryClassifier' in x, allFiles))
    subFiles = list(filter(lambda x:modelType in x, allFiles))
    print(str(len(subFiles)) + " files found for model type " + modelType + " and cell type " + ct)
    if len(subFiles) > 0:
        allDat[ct] = pd.concat([utils.loadResults(os.path.join(ct, x),"binary", 2) for x in subFiles]).sort_values(by=["Chr", "Position", "nCpG"])
        ## count
        print(str(len(allDat[ct])) + " models loaded for model type " + modelType + " and cell type " + ct)
        ## add Density column
        allDat[ct]["Density"] = allDat[ct]['WindowSize']/allDat[ct]['nCpG']
        ## calc overall accuracy
        sensCols = [col for col in allDat[ct].columns if col.endswith('sensitivity')]
        allDat[ct]["Accuracy"] = allDat[ct][sensCols].mean(1)

minCpG = allDat[cellTypes[0]].nCpG.min()
maxWindow = allDat[cellTypes[0]].WindowSize.max()
maxDensity = allDat[cellTypes[0]].Density.max()


## violinplot of accuracy statistics
fig2, ax1 = plt.subplots()
ax1.violinplot(pd.concat([allDat[x]['Accuracy'] for x in cellTypes], axis = 1, names = allDat.keys()),showextrema=True, showmedians=True)
plt.xticks(np.array(range(len(allDat)))+1, allDat.keys())
ax1.set(xlim = [0.5,nCT+0.5], ylim = [0,1], xlabel='Cell type', ylabel='Mean accuracy across CV')
ax1.grid(True)
fig2.savefig("Plots/ViolinplotAccuracyBinary" + modelType + "Classifiers.png", dpi=150)

fig2, ax1 = plt.subplots()
ax1.violinplot(pd.concat([allDat[x]['CT1_sensitivity'] for x in cellTypes], axis = 1, names = allDat.keys()),showextrema=True, showmedians=True)
plt.xticks(np.array(range(len(allDat)))+1, allDat.keys())
ax1.set(xlim = [0.5,nCT+0.5], ylim = [0,1], xlabel='Cell type', ylabel='Sensitivity across CV')
ax1.grid(True)
fig2.savefig("Plots/ViolinplotSensitivityBinary" + modelType + "Classifiers.png", dpi=150)

fig2, ax1 = plt.subplots()
ax1.violinplot(pd.concat([allDat[x]['CT1_specificity'] for x in cellTypes], axis = 1, names = allDat.keys()),showextrema=True, showmedians=True)
plt.xticks(np.array(range(len(allDat)))+1, allDat.keys())
ax1.set(xlim = [0.5,nCT+0.5], ylim = [0,1], xlabel='Cell type', ylabel='Specificity across CV')
ax1.grid(True)
fig2.savefig("Plots/ViolinplotSpecificityBinary" + modelType + "Classifiers.png", dpi=150)

## count the number of CTs that can be predicted from each set of CpGs
countCT = pd.concat([allDat[x][["CT1_sensitivity","CT1_specificity"]].min(axis = 1) > thres for x in cellTypes], axis = 1)
fig3, axs = plt.subplots()
x = np.arange(nCT+1)  # the label locations
width = 0.8  # the width of the bars
axs.bar(x, countCT.sum(axis = 1).value_counts(sort=False).values, width)

# Add some text for labels, title and custom x-axis tick labels, etc.
axs.set_ylabel('Number of models')
axs.set_xlabel('Number of cell types')
axs.set_xticks(x)
fig3.savefig("Plots/BarchartNumberofCelltypesPredictedBinary" + modelType + "Classifiers.png", dpi=150)


## cumulative number of specific and sensitive models
accuracyBins = np.arange(0,1.01,0.05)

cumSumTotals = pd.DataFrame([utils.cumSumSensSpec(allDat[x], ["CT1_sensitivity","CT1_specificity"],accuracyBins) for x in cellTypes]).transpose()

fig5, axs = plt.subplots()
for col in cumSumTotals.columns:
    axs.plot(np.flip(np.arange(0,1,0.05)), cumSumTotals[col], label=cellTypes[col])


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
for ct in cellTypes:
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
 