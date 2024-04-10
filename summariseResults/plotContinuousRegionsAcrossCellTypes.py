
import utils
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pandas.api.types import CategoricalDtype

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

allDat = {}

## load results
for ct in cellTypes:
    allFiles = os.listdir(ct)
    allFiles = list(filter(lambda x:'SummaryModelPerformanceByAccuracyContinuousClassifiersChr' in x, allFiles))
    if(len(allFiles) == 22):
        print("22 files found for aggregating")
        results = pd.concat([pd.read_csv(os.path.join(ct,x), header = 0, names = ("Algorithm", "Threshold", "nModels", "MeanModelSize", "SDModelSize", "nRegions", "MeanRegionSize", "SDRegionSize", "TotalRegionLength", "MeanInterRegionSize", "SDInterRegionSize", "TotalInterRegionSize", "ProportionBasesInRegion", "MeannModelsRegion", "SDnModelsRegion", "MaxnModelsRegion", "nSingleModelRegions", "MeanMinOverlapRegion", "SDMinOverlapRegion")) for x in allFiles])
        # filter to best results
        results = results[results.Algorithm == "BestAccuracy"]
        results["CellType"] = ct
        allDat[ct] = results

  
results = pd.concat(allDat)

aggResults = pd.concat([
# Columns that need to be summed together
results[["CellType", "Threshold", "nModels", "nRegions", "TotalRegionLength", "TotalInterRegionSize", "nSingleModelRegions"]].groupby(["CellType", "Threshold"]).sum(), 
# Columns that need max taken
results[["CellType", "Threshold", "MaxnModelsRegion"]].groupby(["CellType", "Threshold"]).max()], axis = 1)

# Columns that need recalculating
aggResults["MeanRegionSize"] = aggResults.TotalRegionLength/aggResults.nRegions
aggResults["MeanInterRegionSize"] = aggResults.TotalInterRegionSize/(aggResults.nRegions+22)
aggResults["MeannModelsRegion"] = aggResults.nModels/aggResults.nRegions


aggResults["MeanModelSize"] = results[["CellType", "Threshold", "nModels", "MeanModelSize"]].groupby(["CellType", "Threshold"]).apply(lambda x: "NaN" if sum(x.nModels) == 0 else np.average(x.MeanModelSize, weights = x.nModels, axis = 0))

aggResults["MeanMinOverlapRegion"] = results[["CellType", "Threshold", "nRegions", "MeanMinOverlapRegion"]].groupby(["CellType", "Threshold"]).apply(lambda x: "NaN" if sum(x.nRegions) == 0 else np.average(x.MeanMinOverlapRegion, weights = x.nRegions, axis = 0))

aggResults["ProportionBasesInRegion"] = aggResults.TotalRegionLength/ (aggResults.TotalInterRegionSize + aggResults.TotalRegionLength)

aggResults = aggResults.reset_index()

cat_type = CategoricalDtype(categories=cellTypes, ordered=True)
aggResults["CellType"] = aggResults["CellType"].astype(cat_type)

fig1,ax1 = plt.subplots()
ax1.plot(aggResults.pivot(columns = "CellType", values = "nRegions", index = "Threshold"))
ax1.set_ylabel('Number of Regions')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
ax1.legend(labels = aggResults.CellType.unique())
fig1.savefig("Plots/LineGraphAccuracynRegionsContinuousClassifiersAcrossCellTypes.png", dpi=150)


fig2,ax1 = plt.subplots()
ax1.plot(aggResults.pivot(columns = "CellType", values = "MeanModelSize", index = "Threshold"))
ax1.set_ylabel('Mean span of CpGs (bp)')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
ax1.legend(aggResults.CellType.unique())
fig2.savefig("Plots/LineGraphAccuracyMeanModelSizeContinuousClassifiersAcrossCellTypes.png", dpi=150)



fig3,ax1 = plt.subplots()
ax1.plot(aggResults.pivot(columns = "CellType", values = "ProportionBasesInRegion", index = "Threshold"))
ax1.set_ylabel('Proportion bases')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
ax1.legend(aggResults.CellType.unique())
fig3.savefig("Plots/LineGraphAccuracyProportionBasesContinuousClassifiersAcrossCellTypes.png", dpi=150)



fig4,ax1 = plt.subplots()
ax1.plot(aggResults.pivot(columns = "CellType", values = "MeanRegionSize", index = "Threshold"))
ax1.set_ylabel('Mean region size')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
ax1.legend(aggResults.CellType.unique())
fig4.savefig("Plots/LineGraphAccuracyMeanRegionSizeContinuousClassifiersAcrossCellTypes.png", dpi=150)


fig5,ax1 = plt.subplots()
ax1.plot(aggResults.pivot(columns = "CellType", values = "MeanInterRegionSize", index = "Threshold"))
ax1.set_ylabel('Mean gap between regions')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
ax1.legend(aggResults.CellType.unique())
fig5.savefig("Plots/LineGraphAccuracyMeanInterRegionSizeContinuousClassifiersAcrossCellTypes.png", dpi=150)


fig6,ax1 = plt.subplots()
ax1.plot(aggResults.pivot(columns = "CellType", values = "MeannModelsRegion", index = "Threshold"))
ax1.set_ylabel('Mean number of models per region')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
ax1.legend(aggResults.CellType.unique())
fig6.savefig("Plots/LineGraphAccuracyMeannModelsRegionContinuousClassifiersAcrossCellTypes.png", dpi=150)




fig7,ax1 = plt.subplots()
ax1.plot(aggResults.pivot(columns = "CellType", values = "nSingleModelRegions", index = "Threshold"))
ax1.set_ylabel('Number of single model regions')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
ax1.legend(aggResults.CellType.unique())
fig7.savefig("Plots/LineGraphAccuracynSingleModelRegionsContinuousClassifiersAcrossCellTypes.png", dpi=150)


fig8,ax1 = plt.subplots()
ax1.plot(aggResults.pivot(columns = "CellType", values = "MeanMinOverlapRegion", index = "Threshold"))
ax1.set_ylabel('Mean minimum overlap required')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
ax1.legend(aggResults.CellType.unique())
fig8.savefig("Plots/LineGraphAccuracyMeanMinOverlapRegionContinuousClassifiersAcrossCellTypes.png", dpi=150)

# mulitpanel figurexvar 
xvar = aggResults.Threshold.unique()
fig, axs = plt.subplots(2,3)
axs[0,0].plot(aggResults.pivot(columns = "CellType", values = "nRegions", index = "Threshold"))
axs[0,0].set_ylabel('Number of Regions')  
axs[0,0].set_xlabel('Accuracy Threshold')
axs[0,0].grid(True)
#plt.title('A', loc='left')

mu = np.array(aggResults.pivot(columns = "CellType", values = "MeannModelsRegion", index = "Threshold"))
#sigma = np.array(aggResults.pivot(columns = "Algorithm", values = "SDnModelsRegion", index = "Threshold")["BestAccuracy"])
#ax1.fill_between(xvar, mu-sigma, mu+sigma, alpha = 0.5)
axs[0,1].plot(xvar, mu)
axs[0,1].set_ylabel('Mean number of models')  
axs[0,1].set_xlabel('Accuracy Threshold')
axs[0,1].grid(True)
#plt.title('B', loc='left')

mu = np.array(aggResults.pivot(columns = "CellType", values = "MeanRegionSize", index = "Threshold"))/100000
#sigma = np.array(aggResults.pivot(columns = "Algorithm", values = "SDRegionSize", index = "Threshold")["BestAccuracy"])
#ax1.fill_between(xvar, mu-sigma, mu+sigma, alpha = 0.5)
axs[0,2].plot(xvar, mu)
axs[0,2].set_ylabel('Mean region size (kb)')  
axs[0,2].set_xlabel('Accuracy Threshold')
axs[0,2].grid(True)
#axs[0,2].title('C', loc='left')

axs[1,0].plot(aggResults.pivot(columns = "CellType", values = "ProportionBasesInRegion", index = "Threshold"))
axs[1,0].set_ylabel('Proportion bases')  
axs[1,0].set_xlabel('Accuracy Threshold')
axs[1,0].grid(True)
#axs[1,0].title('D', loc='left')

mu = np.array(aggResults.pivot(columns = "CellType", values = "MeanInterRegionSize", index = "Threshold"))/100000
mu[-1] = float("nan")
#sigma = np.array(aggResults.pivot(columns = "Algorithm", values = "SDInterRegionSize", index = "Threshold")["BestAccuracy"])
#ax1.fill_between(xvar, mu-sigma, mu+sigma, alpha = 0.5)
axs[1,1].plot(xvar, mu)
axs[1,1].set_ylabel('Mean gap between regions (kb)')  
axs[1,1].set_xlabel('Accuracy Threshold')
axs[1,1].grid(True)
#plt.title('E', loc='left')
axs[1,1].legend(aggResults.CellType.unique())

axs[1,2].axis('off')

handles, labels = axs[0,0].get_legend_handles_labels()
#axs[1,2].legend(handles, labels)


plt.subplots_adjust(left=0.1,
                    bottom=0.12, 
                    right=0.95, 
                    top=0.95, 
                    wspace=0.3, 
                    hspace=0.4)
                    
fig.legend(bbox_to_anchor=(1.3, 0.6))
fig.set_size_inches(16, 8)

fig.savefig("Plots/MultiPanelPlotLineGraphAccuracyRegionPropertiesContinuousClassifiersAcrossCellTypes.png", dpi=150)
