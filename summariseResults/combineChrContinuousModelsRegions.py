# Take chromosome level regional summarise and aggregate to genome-wide statistics

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})


## process command line information
resultsPath = sys.argv[1]

os.chdir(resultsPath)

## create output folder for plots
if not os.path.exists("Plots"):
    os.makedirs("Plots")

allFiles = os.listdir()
allFiles = list(filter(lambda x:'SummaryModelPerformanceByAccuracyContinuousClassifiersChr' in x, allFiles))

if(len(allFiles) == 22):
    print("22 files found for aggregating")
    results = pd.concat([pd.read_csv(x, header = 0, names = ("Algorithm", "Threshold", "nModels", "MeanModelSize", "SDModelSize", "nRegions", "MeanRegionSize", "SDRegionSize", "TotalRegionLength", "MeanInterRegionSize", "SDInterRegionSize", "TotalInterRegionSize", "ProportionBasesInRegion", "MeannModelsRegion", "SDnModelsRegion", "MaxnModelsRegion", "nSingleModelRegions", "MeanMinOverlapRegion", "SDMinOverlapRegion")) for x in allFiles])
else: 
    sys.exit("No files found for aggregating")


aggResults = pd.concat([
# Columns that need to be summed together
results[["Algorithm", "Threshold", "nModels", "nRegions", "TotalRegionLength", "TotalInterRegionSize", "nSingleModelRegions"]].groupby(["Algorithm", "Threshold"]).sum(), 
# Columns that need max taken
results[["Algorithm", "Threshold", "MaxnModelsRegion"]].groupby(["Algorithm", "Threshold"]).max()], axis = 1)



# Columns that need recalculating
aggResults["MeanRegionSize"] = aggResults.TotalRegionLength/aggResults.nRegions
aggResults["MeanInterRegionSize"] = aggResults.TotalInterRegionSize/(aggResults.nRegions+22)
aggResults["MeannModelsRegion"] = aggResults.nModels/aggResults.nRegions


grouped = results[["Algorithm", "Threshold", "MeanRegionSize", "SDRegionSize", "nRegions"]].groupby(["Algorithm", "Threshold"])
allSD = []
for name, group in grouped:
    mu = (group.MeanRegionSize * group.nRegions).sum()/ group.nRegions.sum()
    sigma = ((((group.nRegions - 1) * group.SDRegionSize ** 2 + group.nRegions * (group.MeanRegionSize - mu)**2).sum())/(group.nRegions.sum()-1))**0.5
    allSD.append(sigma)

aggResults["SDRegionSize"] = allSD   
 
grouped = results[["Algorithm", "Threshold", "MeanInterRegionSize", "SDInterRegionSize", "nRegions"]].groupby(["Algorithm", "Threshold"])
allSD = []
for name, group in grouped:
    mu = (group.MeanInterRegionSize * group.nRegions).sum()/ group.nRegions.sum()
    sigma = ((((group.nRegions - 1) * group.SDInterRegionSize ** 2 + group.nRegions * (group.MeanInterRegionSize - mu)**2).sum())/(group.nRegions.sum()-1))**0.5
    allSD.append(sigma)

aggResults["SDInterRegionSize"] = allSD   

grouped = results[["Algorithm", "Threshold", "MeannModelsRegion", "SDnModelsRegion", "nRegions"]].groupby(["Algorithm", "Threshold"])
allSD = []
for name, group in grouped:
    mu = (group.MeannModelsRegion * group.nRegions).sum()/ group.nRegions.sum()
    sigma = ((((group.nRegions - 1) * group.SDnModelsRegion ** 2 + group.nRegions * (group.MeannModelsRegion - mu)**2).sum())/(group.nRegions.sum()-1))**0.5
    allSD.append(sigma)

aggResults["SDnModelsRegion"] = allSD   
    

aggResults["MeanModelSize"] = results[["Algorithm", "Threshold", "nModels", "MeanModelSize"]].groupby(["Algorithm", "Threshold"]).apply(lambda x: "NaN" if sum(x.nModels) == 0 else np.average(x.MeanModelSize, weights = x.nModels, axis = 0))

aggResults["MeanMinOverlapRegion"] = results[["Algorithm", "Threshold", "nRegions", "MeanMinOverlapRegion"]].groupby(["Algorithm", "Threshold"]).apply(lambda x: "NaN" if sum(x.nRegions) == 0 else np.average(x.MeanMinOverlapRegion, weights = x.nRegions, axis = 0))

aggResults["ProportionBasesInRegion"] = aggResults.TotalRegionLength/ (aggResults.TotalInterRegionSize + aggResults.TotalRegionLength)

aggResults = aggResults.reset_index()

fig1,ax1 = plt.subplots()
ax1.plot(aggResults.pivot(columns = "Algorithm", values = "nRegions", index = "Threshold"))
ax1.set_ylabel('Number of Regions')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
ax1.legend(labels = aggResults.Algorithm.unique())
fig1.savefig("Plots/LineGraphAccuracynRegionsContinuousClassifiers.png", dpi=150)


fig1,ax1 = plt.subplots()
ax1.plot(aggResults.pivot(columns = "Algorithm", values = "nRegions", index = "Threshold")[["BestAccuracy"]])
ax1.set_ylabel('Number of Regions')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
fig1.savefig("Plots/LineGraphAccuracynRegionsContinuousClassifiersBestOnly.png", dpi=150)



fig2,ax1 = plt.subplots()
ax1.plot(aggResults.pivot(columns = "Algorithm", values = "MeanModelSize", index = "Threshold"))
ax1.set_ylabel('Mean span of CpGs (bp)')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
ax1.legend(aggResults.Algorithm.unique())
fig2.savefig("Plots/LineGraphAccuracyMeanModelSizeContinuousClassifiers.png", dpi=150)

xvar = np.array(results["Threshold"].unique())
mu = np.array(aggResults.pivot(columns = "Algorithm", values = "MeanModelSize", index = "Threshold")["BestAccuracy"])
#sigma = np.array(aggResults.pivot(columns = "Algorithm", values = "SDModelSize", index = "Threshold")["BestAccuracy"])
fig2,ax1 = plt.subplots()
#ax1.fill_between(xvar, mu-sigma, mu+sigma, alpha = 0.5)
ax1.plot(xvar, mu)
ax1.set_ylabel('Mean span of CpGs (bp)')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
fig2.savefig("Plots/LineGraphAccuracyMeanModelSizeContinuousClassifiersBestOnly.png", dpi=150)

fig3,ax1 = plt.subplots()
ax1.plot(aggResults.pivot(columns = "Algorithm", values = "ProportionBasesInRegion", index = "Threshold"))
ax1.set_ylabel('Proportion bases')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
ax1.legend(aggResults.Algorithm.unique())
fig3.savefig("Plots/LineGraphAccuracyProportionBasesContinuousClassifiers.png", dpi=150)

fig3,ax1 = plt.subplots()
ax1.plot(aggResults.pivot(columns = "Algorithm", values = "ProportionBasesInRegion", index = "Threshold")["BestAccuracy"])
ax1.set_ylabel('Proportion bases')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
fig3.savefig("Plots/LineGraphAccuracyProportionBasesContinuousClassifiersBestOnly.png", dpi=150)


fig4,ax1 = plt.subplots()
ax1.plot(aggResults.pivot(columns = "Algorithm", values = "MeanRegionSize", index = "Threshold"))
ax1.set_ylabel('Mean region size')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
ax1.legend(aggResults.Algorithm.unique())
fig4.savefig("Plots/LineGraphAccuracyMeanRegionSizeContinuousClassifiers.png", dpi=150)

mu = np.array(aggResults.pivot(columns = "Algorithm", values = "MeanRegionSize", index = "Threshold")["BestAccuracy"])
#sigma = np.array(aggResults.pivot(columns = "Algorithm", values = "SDRegionSize", index = "Threshold")["BestAccuracy"])

fig4,ax1 = plt.subplots()
#ax1.fill_between(xvar, mu-sigma, mu+sigma, alpha = 0.5)
ax1.plot(xvar, mu)
ax1.set_ylabel('Mean region size')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
ax1.legend(aggResults.Algorithm.unique())
fig4.savefig("Plots/LineGraphAccuracyMeanRegionSizeContinuousClassifiersBestOnly.png", dpi=150)


fig5,ax1 = plt.subplots()
ax1.plot(aggResults.pivot(columns = "Algorithm", values = "MeanInterRegionSize", index = "Threshold"))
ax1.set_ylabel('Mean gap between regions')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
ax1.legend(aggResults.Algorithm.unique())
fig5.savefig("Plots/LineGraphAccuracyMeanInterRegionSizeContinuousClassifiers.png", dpi=150)


mu = np.array(aggResults.pivot(columns = "Algorithm", values = "MeanInterRegionSize", index = "Threshold")["BestAccuracy"])
#sigma = np.array(aggResults.pivot(columns = "Algorithm", values = "SDInterRegionSize", index = "Threshold")["BestAccuracy"])
fig5,ax1 = plt.subplots()
#ax1.fill_between(xvar, mu-sigma, mu+sigma, alpha = 0.5)
ax1.plot(xvar, mu)
ax1.set_ylabel('Mean gap between regions')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
fig5.savefig("Plots/LineGraphAccuracyMeanInterRegionSizeContinuousClassifiersBestOnly.png", dpi=150)

fig6,ax1 = plt.subplots()
ax1.plot(aggResults.pivot(columns = "Algorithm", values = "MeannModelsRegion", index = "Threshold"))
ax1.set_ylabel('Mean number of models per region')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
ax1.legend(aggResults.Algorithm.unique())
fig6.savefig("Plots/LineGraphAccuracyMeannModelsRegionContinuousClassifiers.png", dpi=150)

mu = np.array(aggResults.pivot(columns = "Algorithm", values = "MeannModelsRegion", index = "Threshold")["BestAccuracy"])
#sigma = np.array(aggResults.pivot(columns = "Algorithm", values = "SDnModelsRegion", index = "Threshold")["BestAccuracy"])
fig6,ax1 = plt.subplots()
#ax1.fill_between(xvar, mu-sigma, mu+sigma, alpha = 0.5)
ax1.plot(xvar, mu)
ax1.set_ylabel('Mean number of models per region')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
fig6.savefig("Plots/LineGraphAccuracyMeannModelsRegionContinuousClassifiersBestOnly.png", dpi=150)


fig7,ax1 = plt.subplots()
ax1.plot(aggResults.pivot(columns = "Algorithm", values = "nSingleModelRegions", index = "Threshold"))
ax1.set_ylabel('Number of single model regions')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
ax1.legend(aggResults.Algorithm.unique())
fig7.savefig("Plots/LineGraphAccuracynSingleModelRegionsContinuousClassifiers.png", dpi=150)

fig7,ax1 = plt.subplots()
ax1.plot(aggResults.pivot(columns = "Algorithm", values = "nSingleModelRegions", index = "Threshold")["BestAccuracy"])
ax1.set_ylabel('Number of single model regions')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
fig7.savefig("Plots/LineGraphAccuracynSingleModelRegionsContinuousClassifiersBestOnly.png", dpi=150)


fig8,ax1 = plt.subplots()
ax1.plot(aggResults.pivot(columns = "Algorithm", values = "MeanMinOverlapRegion", index = "Threshold"))
ax1.set_ylabel('Mean minimum overlap required')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
ax1.legend(aggResults.Algorithm.unique())
fig8.savefig("Plots/LineGraphAccuracyMeanMinOverlapRegionContinuousClassifiers.png", dpi=150)

mu = np.array(aggResults.pivot(columns = "Algorithm", values = "MeanMinOverlapRegion", index = "Threshold")["BestAccuracy"])
#sigma = np.array(aggResults.pivot(columns = "Algorithm", values = "SDMinOverlapRegion", index = "Threshold")["BestAccuracy"])
fig8,ax1 = plt.subplots()
#ax1.fill_between(xvar, mu-sigma, mu+sigma, alpha = 0.5)
ax1.plot(xvar, mu)
ax1.set_ylabel('Mean minimum overlap required')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
fig8.savefig("Plots/LineGraphAccuracyMeanMinOverlapRegionContinuousClassifiersBestOnly.png", dpi=150)

## manuscript figure

fig, axs = plt.subplots(2,2)
axs[0,0].plot(aggResults.pivot(columns = "Algorithm", values = "nRegions", index = "Threshold")[["BestAccuracy"]])
axs[0,0].set_ylabel('Number of Regions')  
axs[0,0].set_xlabel('Accuracy Threshold')
axs[0,0].grid(True)
plt.title('A', loc='left')

mu = np.array(aggResults.pivot(columns = "Algorithm", values = "MeannModelsRegion", index = "Threshold")["BestAccuracy"])
#sigma = np.array(aggResults.pivot(columns = "Algorithm", values = "SDnModelsRegion", index = "Threshold")["BestAccuracy"])
#axs[0,1].fill_between(xvar, mu-sigma, mu+sigma, alpha = 0.5)
axs[0,1].plot(xvar, mu)
axs[0,1].set_ylabel('Mean number of models')  
axs[0,1].set_xlabel('Accuracy Threshold')
axs[0,1].grid(True)
plt.title('B', loc='left')

mu = np.array(aggResults.pivot(columns = "Algorithm", values = "MeanRegionSize", index = "Threshold")["BestAccuracy"])/1000
#sigma = np.array(aggResults.pivot(columns = "Algorithm", values = "SDRegionSize", index = "Threshold")["BestAccuracy"])/1000
#axs[1,0].fill_between(xvar, mu-sigma, mu+sigma, alpha = 0.5)
axs[1,0].plot(xvar, mu)
axs[1,0].set_ylabel('Mean region size (kb)')  
axs[1,0].set_xlabel('Accuracy Threshold')
axs[1,0].grid(True)
axs[1,0].title('C', loc='left')

mu = np.array(aggResults.pivot(columns = "Algorithm", values = "MeanInterRegionSize", index = "Threshold")["BestAccuracy"])/1000
mu[-1] = float("nan")
#sigma = np.array(aggResults.pivot(columns = "Algorithm", values = "SDInterRegionSize", index = "Threshold")["BestAccuracy"])/1000
#axs[1,1].fill_between(xvar, mu-sigma, mu+sigma, alpha = 0.5)
axs[1,1].plot(xvar, mu)
axs[1,1].set_ylabel('Mean gap between regions (kb)')  
axs[1,1].set_xlabel('Accuracy Threshold')
axs[1,1].grid(True)
plt.title('D', loc='left')

plt.subplots_adjust(left=0.2,
                    bottom=0.12, 
                    right=0.95, 
                    top=0.95, 
                    wspace=0.3, 
                    hspace=0.4)
fig.set_size_inches(8, 8)
fig.savefig("Plots/MultiPanelPlotLineGraphAccuracyRegionPropertiesContinuousClassifiersBestOnly.png", dpi=150)
