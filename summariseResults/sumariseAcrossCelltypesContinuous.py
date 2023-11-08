
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

nCT = len(cellTypes)

os.chdir(resultsPath)

## create output folder for plots
if not os.path.exists("Plots"):
    os.makedirs("Plots")

## list files in results folder

allDat = {}
modelOpts = ["KNN", "NBayes", "RandFor", "SVM"]

## load results
for ct in cellTypes:
    modelOptsFilt = []
    allFiles = os.listdir(ct)
    allFiles = list(filter(lambda x:'ContinuousClassifier' in x, allFiles))
    ## load results
    tmpDat = {}
    for modelType in modelOpts:
        subFiles = list(filter(lambda x:modelType in x, allFiles))
        print(str(len(subFiles)) + " files found for model type " + modelType)
        if(len(subFiles) == 22):
            modelOptsFilt = modelOptsFilt + [modelType]
            tmpDat[modelType] = pd.concat([utils.loadResults(os.path.join(ct,x),"continuous") for x in subFiles]).sort_values(by=["Chr", "Position", "nCpG"])
            ## count
            print(str(len(tmpDat[modelType])) + " models loaded for model type " + modelType)          
    
    ## merge into a single data.frame to determine best algorithm for each model
    mergeDf = pd.concat([tmpDat[x]["MeanAccuracy"] for x in modelOptsFilt], axis = 1)
    mergeDf.columns = modelOptsFilt
    ## identify for each model the best ML algorithm
    maxMean = mergeDf.max(1)
    allDat[ct] = maxMean

for ct in cellTypes:
    print(str("Summary for cell type " + ct))
    allDat[ct].describe()


## violinplot of accuracy statistics
fig1, ax1 = plt.subplots()
vplots = ax1.violinplot(pd.concat([allDat[x] for x in cellTypes], axis = 1, names = cellTypes),showextrema=True, showmedians=True)
# Set the color of the violin patches
colNum=0
for pc in vplots['bodies']:
    pc.set_facecolor("C" + str(colNum))
    colNum+=1

# Make all the violin statistics marks black:
for partname in ('cbars','cmins','cmaxes','cmedians'):
    vp = vplots[partname]
    vp.set_edgecolor("black")
    vp.set_linewidth(1)


plt.xticks(list(range(1, nCT+1)), cellTypes)
ax1.set(xlim = [0.5,nCT+0.5], ylim = [0.4,1], xlabel='Cell type', ylabel='Mean accuracy across CV')
ax1.grid(True)
fig1.savefig("Plots/ViolinplotAccuracyContinuousClassifiersAcrossCellTypes.png", dpi=150)


## count cumulative sum of number of models with accuracy > x.

accuracyBins = np.arange(0,1.01,0.02)
	
allDat = pd.DataFrame.from_dict(allDat)
    
cumSumTotals = allDat.apply(utils.cumSumAccuracy, axis=0, bins = accuracyBins)


fig3, ax1 = plt.subplots()
for each in cellTypes:
    ax1.plot(np.flip(np.arange(0,1,0.02)), cumSumTotals[each], label = each)


ax1.set_xlabel('Mean accuracy across CV')  # Add an x-label to the axes.
ax1.set_ylabel('Number of classifiers(x1000)')  # Add a y-label to the axes.
y_vals = ax1.get_yticks()
ax1.yaxis.set_major_locator(mticker.FixedLocator(y_vals))
ax1.set_yticklabels(['{:.0f}'.format(x / 1000) for x in y_vals])
ax1.legend()
ax1.grid(True)
fig3.savefig("Plots/LineGraphCumulativeAccuracyContinuousClassifiersAcrossCellTypes.png", dpi=150)


## summarise as a function of model characteristics
fig5, (ax1,ax2, ax3) = plt.subplots(1,3)

groupedCpG = allDat.groupby(tmpDat[modelOptsFilt[0]].nCpG).mean()
minCpG = tmpDat[modelOptsFilt[0]].nCpG.min()
maxCpG = max(tmpDat[modelOptsFilt[0]].nCpG)+1
for each in cellTypes:
    ax1.plot(np.arange(minCpG,maxCpG,1), np.asarray(groupedCpG[each]), label = each)

ax1.set_ylabel('Mean accuracy')  
ax1.set_xlabel('Number of CpGs')  
ax1.grid(True)

## plot against window size
maxWindow = tmpDat[modelOptsFilt[0]].WindowSize.max()
spanBins = np.arange(0,maxWindow+100,100)
windowBins = pd.cut(tmpDat[modelOptsFilt[0]]['WindowSize'], bins = spanBins)

groupedSpan = allDat.groupby(windowBins).mean()
for each in cellTypes:
    ax2.plot(np.arange(100,maxWindow+100,100), np.asarray(groupedSpan[each]), label = each)

ax2.set_xlabel('Span of CpGs')  
ax2.grid(True)

## plot against density
tmpDat[modelOptsFilt[0]]["Density"] = tmpDat[modelOptsFilt[0]]['WindowSize']/tmpDat[modelOptsFilt[0]]['nCpG']
maxDensity = tmpDat[modelOptsFilt[0]].Density.max()
densityBreaks = np.arange(0,maxDensity+50,50)
densityBins = pd.cut(tmpDat[modelOptsFilt[0]]['Density'], bins = densityBreaks)

groupedDensity = allDat.groupby(densityBins).mean()
for each in cellTypes:
    ax3.plot(densityBreaks[1:], np.asarray(groupedDensity[each]), label = each)

ax3.set_xlabel('Density of CpGs')  
ax3.legend()
ax3.grid(True)
fig5.set_size_inches(12, 4)
fig5.savefig("Plots/LineGraphAccuracyAgainstModelPropertiesAcrossCellTypes.png", dpi=150)


# summarise model properties as a function of accuracy
accuracyBreaks = np.arange(0.5,1.01,0.01)
fig6, (ax1,ax2, ax3) = plt.subplots(1,3)

for each in cellTypes:
    accuracyBins = pd.cut(allDat[each], bins = accuracyBreaks)
    groupedAccuracy = tmpDat[modelOptsFilt[0]].groupby(accuracyBins).mean()
    ax1.plot(accuracyBreaks[:-1]+0.005, np.asarray(groupedAccuracy['nCpG']), label = each)
    ax2.plot(accuracyBreaks[:-1]+0.005, np.asarray(groupedAccuracy['WindowSize']), label = each)
    ax3.plot(accuracyBreaks[:-1]+0.005, np.asarray(groupedAccuracy['Density']), label = each)


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
fig6.savefig("Plots/LineGraphModelPropertiesAgainstAccuracyAcrossCellTypes.png", dpi=150)

## compare regions
allRegions = {}
for ct in cellTypes:
    results = pd.read_csv(os.path.join(ct,"SummaryModelPerformanceByAccuracyContinuousClassifiers.csv"), header = 0, names = ("Algorithm", "Threshold", "nModels", "MeanModelSize", "SDModelSize", "nRegions", "MeanRegionSize", "SDRegionSize", "TotalRegionLength", "MeanInterRegionSize", "SDInterRegionSize", "TotalInterRegionSize", "ProportionBasesInRegion", "MeannModelsRegion", "SDnModelsRegion", "MaxnModelsRegion", "nSingleModelRegions", "MeanMinOverlapRegion", "SDMinOverlapRegion"))
    allRegions[ct] = results[results.Algorithm == "BestAccuracy"].reset_index()


for sumStat in ["nRegions", "MeanRegionSize", "ProportionBasesInRegion", "MeanInterRegionSize",  "MeannModelsRegion"]:
    fig2, ax1 = plt.subplots()
    ax1.plot(pd.concat([allRegions[x][sumStat] for x in cellTypes], axis = 1, names = cellTypes))
    ax1.set_xticklabels([str(round(float(label), 2)) for label in allRegions[ct]["Threshold"]])
    ax1.set_ylabel('Number of Regions')  
    ax1.set_xlabel('Accuracy Threshold')
    ax1.grid(True)
    ax1.legend(labels = cellTypes)
    fig2.savefig("Plots/LineGraphAccuracy" + sumStat + "ContinuousClassifiersAcrossCellTypes.png", dpi=150)
