
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

## load results
for ct in cellTypes:
    allFiles = os.listdir(ct)
    allFiles = list(filter(lambda x:'ContinuousClassifier' in x, allFiles))
    ## load results
    tmpDat = {}
    for modelType in ["KNN", "NBayes", "RandFor", "SVM"]:
        subFiles = list(filter(lambda x:modelType in x, allFiles))
        print(str(len(subFiles)) + " files found for model type " + modelType)
        tmpDat[modelType] = pd.concat([utils.loadResults(os.path.join(ct,x),"continuous") for x in subFiles]).sort_values(by=["Chr", "Position", "nCpG"])
        ## count
        print(str(len(tmpDat[modelType])) + " models loaded for model type " + modelType)
    ## merge into a single data.frame to determine best algorithm for each model
    mergeDf = pd.DataFrame({'svm': tmpDat["SVM"]["MeanAccuracy"], 'knn': tmpDat["KNN"]["MeanAccuracy"], 'naiveBayes': tmpDat["NBayes"]["MeanAccuracy"], 'randomForest': tmpDat["RandFor"]["MeanAccuracy"]})
    del tmpDat
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

## for binary classifiers how many cell types can it predict,
## any common combinations
(pd.DataFrame(allDat) > thres).sum(1).value_counts(sort=False)



fig4, axs = plt.subplots()
x = np.arange(nCT)  # the label locations
width = 1  # the width of the bars
tab = pd.DataFrame(allDat).idxmax(1).value_counts(sort=False)
axs.bar(x, tab.values, width)
# Add some text for labels, title and custom x-axis tick labels, etc.
axs.set_ylabel('Number of classifiers')
axs.set_xlabel('Best binary predictor')
axs.set_xticks(x, tab.index)


pd.DataFrame(allDat).max(1) - pd.DataFrame(allDat).min(1)

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
