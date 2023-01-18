
import utils
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# accuracy threshold
thres = 0.8
# model names
modelOpts = ["KNN", "NBayes", "RandFor", "SVM"]
# axis labels
axisLabels = ["KNN", "Naive Bayes", "Random Forest", "SVM"]

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
for modelType in modelOpts:
    subFiles = list(filter(lambda x:modelType in x, allFiles))
    print(str(len(subFiles)) + " files found for model type " + modelType)
    if len(subFiles) > 0:
        allDat[modelType] = pd.concat([utils.loadResults(x,"binary", nCT) for x in subFiles]).sort_values(by=["Chr", "Position", "nCpG"])
        ## count
        print(str(len(allDat[modelType])) + " models loaded for model type " + modelType)
        ## add Density column
        allDat[modelType]["Density"] = allDat[modelType]['WindowSize']/allDat[modelType]['nCpG']
        ## calc overall accuracy
        sensCols = [col for col in allDat[modelType].columns if col.endswith('sensitivity')]
        allDat[modelType]["Accuracy"] = allDat[modelType][sensCols].mean(1)

## filter options to models available
modelOpts = [element for element in modelOpts if element in allDat.keys()]

## summarize the parameters of the models
allDat[modelOpts[0]]['nCpG'].describe()
allDat[modelOpts[0]]['WindowSize'].describe()
allDat[modelOpts[0]]['Density'].describe()

minCpG = allDat[modelOpts[0]].nCpG.min()
maxWindow = allDat[modelOpts[0]].WindowSize.max()
maxDensity = allDat[modelOpts[0]].Density.max()


## assumes the same for all models
fig1, (ax1, ax2, ax3) = plt.subplots(nrows=1,ncols=3)
n1, bins1, patches1 = ax1.hist(allDat[modelType]['nCpG'], int(max(allDat[modelType]['nCpG']))-2, density=False, facecolor='g', alpha=0.75)
## histogram of number of CpGs
ax1.set(xlim = [2, max(allDat[modelType]['nCpG'])], ylim = [0, max(n1)], xlabel='Number of CpGs',
ylabel='Number of models (x1000)')
ax1.grid(True)
y_vals = ax1.get_yticks().tolist()
ax1.yaxis.set_major_locator(mticker.FixedLocator(y_vals))
ax1.set_yticklabels(['{:.0f}'.format(x / 1000) for x in y_vals])

## histogram of the window size
n2, bins2, patches2 = ax2.hist(allDat[modelType]['WindowSize'], 50, density=False, facecolor='g', alpha=0.75)
ax2.set(xlabel='Span of CpGs (bp)', ylabel='')
ax2.grid(True)
y_vals = ax2.get_yticks()
ax2.yaxis.set_major_locator(mticker.FixedLocator(y_vals))
ax2.set_yticklabels(['{:.0f}'.format(x / 1000) for x in y_vals])

## histogram of probe density
n3, bins3, patches3 = ax3.hist(allDat[modelType]['Density'], 50, density=False, facecolor='g', alpha=0.75)
ax3.set(xlabel='Density of CpGs (bp)', ylabel='')
ax3.grid(True)
y_vals = ax3.get_yticks()
ax3.yaxis.set_major_locator(mticker.FixedLocator(y_vals))
ax3.set_yticklabels(['{:.0f}'.format(x / 1000) for x in y_vals])

# Save files in png format
fig1.set_size_inches(12, 4)
fig1.savefig("Plots/HistogramModelCharacteristics.png", dpi=150)


## violinplot of accuracy statistics
fig2, ax1 = plt.subplots()

ax1.violinplot(pd.concat([allDat[x]['Accuracy'] for x in modelOpts], axis = 1, names = allDat.keys()),showextrema=True, showmedians=True)
plt.xticks(np.array(range(len(allDat)))+1, allDat.keys())
ax1.set(xlim = [0.5,4.5], ylim = [0,1], xlabel='Algorithm', ylabel='Mean accuracy across CV')
ax1.grid(True)
fig2.savefig("Plots/ViolinplotAccuracyBinaryClassifiers.png", dpi=150)

## boxplot sensitivity

fig3, axs = plt.subplots(2,2, sharex = 'all', sharey = 'all')
if "KNN" in allDat.keys():
    axs[0,0].violinplot((allDat["KNN"][sensCols]),showextrema=True, showmedians=True)
    axs[0,0].set_title("KNN")
    axs[0,0].grid(True)

if "NBayes" in allDat.keys():
    axs[0,1].violinplot((allDat["NBayes"][sensCols]),showextrema=True, showmedians=True)
    axs[0,1].set_title("Naive Bayes")
    axs[0,1].grid(True)

if "RandFor" in allDat.keys():
    axs[1,0].violinplot((allDat["RandFor"][sensCols]),showextrema=True, showmedians=True)
    axs[1,0].set_title("Random Forest")
    axs[1,0].grid(True)

if "SVM" in allDat.keys():
    axs[1,1].violinplot((allDat["SVM"][sensCols]),showextrema=True, showmedians=True)
    axs[1,1].set_title("SVM")
    axs[1,1].grid(True)

fig3.text(0.5, 0.04, 'Cell type', ha='center')
fig3.text(0.04, 0.5, 'Sensitivity', va='center', rotation='vertical')
fig3.savefig("Plots/ViolinplotSensitivityBinaryClassifiers.png", dpi=150)


## boxplot specificity
specCols = [col for col in allDat[modelOpts[0]].columns if col.endswith('specificity')]
fig4, axs = plt.subplots(2,2, sharex = 'all', sharey = 'all')

if "KNN" in allDat.keys():
    axs[0,0].violinplot((allDat["KNN"][specCols]),showextrema=True, showmedians=True)
    axs[0,0].set_title("KNN")
    axs[0,0].grid(True)

if "NBayes" in allDat.keys():
    axs[0,1].violinplot((allDat["NBayes"][specCols]),showextrema=True, showmedians=True)
    axs[0,1].set_title("Naive Bayes")
    axs[0,1].grid(True)

if "RandFor" in allDat.keys():
    axs[1,0].violinplot((allDat["RandFor"][specCols]),showextrema=True, showmedians=True)
    axs[1,0].set_title("Random Forest")
    axs[1,0].grid(True)

if "SVM" in allDat.keys():
    axs[1,1].violinplot((allDat["SVM"][specCols]),showextrema=True, showmedians=True)
    axs[1,1].set_title("SVM")
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
for modelType in modelOpts:
    cumSumTotals[modelType] = pd.DataFrame([utils.cumSumSensSpec(allDat[modelType], x,accuracyBins) for x in [["CT" + str(i+1) + "_sensitivity","CT" + str(i+1) + "_specificity"] for i in range(0,int(nCT))]]).transpose()


fig5, axs = plt.subplots(2,2, sharex = 'all', sharey = 'all')
for col in cumSumTotals['KNN'].columns:
    if "KNN" in allDat.keys():
        axs[0,0].plot(np.flip(np.arange(0,1,0.05)), cumSumTotals['KNN'][col], label='CT '+str(col+1))
    if "NBayes" in allDat.keys():
        axs[0,1].plot(np.flip(np.arange(0,1,0.05)), cumSumTotals['NBayes'][col], label='CT '+str(col+1))
    if "RandFor" in allDat.keys():
        axs[1,0].plot(np.flip(np.arange(0,1,0.05)), cumSumTotals['RandFor'][col], label='CT '+str(col+1))
    if "SVM" in allDat.keys():
        axs[1,1].plot(np.flip(np.arange(0,1,0.05)), cumSumTotals['SVM'][col], label='CT '+str(col+1))

y_vals = axs[0,0].get_yticks()
axs[0,0].yaxis.set_major_locator(mticker.FixedLocator(y_vals))
axs[0,0].set_yticklabels(['{:.0f}'.format(x / 1000) for x in y_vals])
axs[1,0].yaxis.set_major_locator(mticker.FixedLocator(y_vals))
axs[1,0].set_yticklabels(['{:.0f}'.format(x / 1000) for x in y_vals])

axs[0,0].set_title("KNN")
axs[0,1].set_title("Naive Bayes")
axs[1,0].set_title("Random Forest")
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
for modelType in modelOpts:
    groupedCpG[modelType] = allDat[modelType].groupby(allDat[modelType].nCpG).mean() 
    for ctNum in np.arange(int(nCT))+1:
        axs[0,modelIndex].plot(np.arange(minCpG,max(allDat[modelType]['nCpG'])+1,1), 
        np.asarray(groupedCpG[modelType]['CT' + str(ctNum) + '_sensitivity']), label='CT  '+ str(ctNum))
        axs[1,modelIndex].plot(np.arange(minCpG,max(allDat[modelType]['nCpG'])+1,1), 
        np.asarray(groupedCpG[modelType]['CT' + str(ctNum) + '_specificity']), label='CT ' + str(ctNum))
        axs[0,modelIndex].grid(True)
        axs[1,modelIndex].grid(True)
        axs[0,modelIndex].set_title(modelType)
    modelIndex+=1 

axs[0,0].set_ylabel('Mean sensitivity')  
axs[1,0].set_ylabel('Mean specificity') 

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
for modelType in modelOpts:
    groupedSpan[modelType] = allDat[modelType].groupby(windowBins).mean() 
    for ctNum in np.arange(int(nCT))+1:
        axs[0,modelIndex].plot(spanBins[:-1], 
        np.asarray(groupedSpan[modelType]['CT' + str(ctNum) + '_sensitivity']), label='CT  '+ str(ctNum))
        axs[1,modelIndex].plot(spanBins[:-1], 
        np.asarray(groupedSpan[modelType]['CT' + str(ctNum) + '_specificity']), label='CT ' + str(ctNum))
        axs[0,modelIndex].grid(True)
        axs[1,modelIndex].grid(True)
        axs[0,modelIndex].set_title(modelType) 
    modelIndex+=1 

axs[0,0].set_ylabel('Mean sensitivity')  
axs[1,0].set_ylabel('Mean specificity') 


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
for modelType in modelOpts:
    groupedDensity[modelType] = allDat[modelType].groupby(densityBins).mean() 
    for ctNum in np.arange(int(nCT))+1:
        axs[0,modelIndex].plot(densityBreaks[:-1], 
        np.asarray(groupedDensity[modelType]['CT' + str(ctNum) + '_sensitivity']), label='CT  '+ str(ctNum))
        axs[1,modelIndex].plot(densityBreaks[:-1], 
        np.asarray(groupedDensity[modelType]['CT' + str(ctNum) + '_specificity']), label='CT ' + str(ctNum))
        axs[0,modelIndex].grid(True)
        axs[1,modelIndex].grid(True)
        axs[0,modelIndex].set_title(modelType)
        
    modelIndex+=1 

axs[0,0].set_ylabel('Mean sensitivity')  
axs[1,0].set_ylabel('Mean specificity') 

fig8.text(0.5, 0.04, 'Density of CpGs', ha='center')
fig8.set_size_inches(20, 10)
fig8.savefig("Plots/LineGraphMeanSensitivitySpecificityDensityBinaryClassifiers.png", dpi=150)
 
 
 ## as a function of accuracy
 # summarise model properties as a function of accuracy
maxAcc = [allDat[x].Accuracy.max() for x in modelOpts]

accuracyBreaks = np.arange(0.5,1.01,0.01)
fig6, (ax1,ax2, ax3) = plt.subplots(1,3)

groupedAccuracy = {}
for modelType in modelOpts:
    accuracyBins = pd.cut(allDat[modelType]['Accuracy'], bins = accuracyBreaks)
    groupedAccuracy[modelType] = allDat[modelType].groupby(accuracyBins).mean()
    ax1.plot(accuracyBreaks[:-1]+0.005, np.asarray(groupedAccuracy[modelType]['nCpG']), label = modelType)
    ax2.plot(accuracyBreaks[:-1]+0.005, np.asarray(groupedAccuracy[modelType]['WindowSize']), label = modelType)
    ax3.plot(accuracyBreaks[:-1]+0.005, np.asarray(groupedAccuracy[modelType]['Density']), label = modelType)
   

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
fig6.savefig("Plots/LineGraphModelPropertiesAgainstAccuracyBinaryClassifiers.png", dpi=150)
