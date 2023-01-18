


import utils
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


## process command line information
resultsPath = sys.argv[1]
dataType = sys.argv[2]

os.chdir(resultsPath)

results = pd.read_csv("SummaryModelPerformanceByAccuracy" + dataType + "Classifiers.csv", header = 0, names = ("Algorithm", "CT", "Threshold", "nModels", "MeanModelSize", "SDModelSize", "nRegions", "MeanRegionSize", "SDRegionSize", "TotalRegionLength", "MeanInterRegionSize", "SDInterRegionSize", "TotalInterRegionSize", "ProportionBasesInRegion", "MeannModelsRegion", "SDnModelsRegion", "MaxnModelsRegion", "nSingleModelRegions", "MeanMinOverlapRegion", "SDMinOverlapRegion"))

fig1,ax1 = plt.subplots()
ax1.plot(results.pivot(columns = "Algorithm", values = "nRegions", index = "Threshold"))
ax1.set_ylabel('Number of Regions')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
ax1.legend(labels = results["Algorithm"].unique())
fig1.savefig("Plots/LineGraphAccuracynRegions" + dataType + "Classifiers.png", dpi=150)


fig2,ax1 = plt.subplots()
ax1.plot(results.pivot(columns = "Algorithm", values = "MeanModelSize", index = "Threshold"))
ax1.set_ylabel('Mean span of CpGs (bp)')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
ax1.legend(labels = results["Algorithm"].unique())
fig2.savefig("Plots/LineGraphAccuracyMeanModelSize" + dataType + "Classifiers.png", dpi=150)

fig3,ax1 = plt.subplots()
ax1.plot(results.pivot(columns = "Algorithm", values = "ProportionBasesInRegion", index = "Threshold"))
ax1.set_ylabel('Proportion bases')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
ax1.legend(labels = results["Algorithm"].unique())
fig3.savefig("Plots/LineGraphAccuracyProportionBases" + dataType + "Classifiers.png", dpi=150)

fig4,ax1 = plt.subplots()
ax1.plot(results.pivot(columns = "Algorithm", values = "MeanRegionSize", index = "Threshold"))
ax1.set_ylabel('Mean region size')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
ax1.legend(labels = results["Algorithm"].unique())
fig4.savefig("Plots/LineGraphAccuracyMeanRegionSize" + dataType + "Classifiers.png", dpi=150)

fig5,ax1 = plt.subplots()
ax1.plot(results.pivot(columns = "Algorithm", values = "MeanInterRegionSize", index = "Threshold"))
ax1.set_ylabel('Mean gap between regions')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
ax1.legend(labels = results["Algorithm"].unique())
fig5.savefig("Plots/LineGraphAccuracyMeanInterRegionSize" + dataType + "Classifiers.png", dpi=150)

fig6,ax1 = plt.subplots()
ax1.plot(results.pivot(columns = "Algorithm", values = "MeannModelsRegion", index = "Threshold"))
ax1.set_ylabel('Mean number of models per region')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
ax1.legend(labels = results["Algorithm"].unique())
fig6.savefig("Plots/LineGraphAccuracyMeannModelsRegion" + dataType + "Classifiers.png", dpi=150)

fig7,ax1 = plt.subplots()
ax1.plot(results.pivot(columns = "Algorithm", values = "nSingleModelRegions", index = "Threshold"))
ax1.set_ylabel('Number of single model regions')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
ax1.legend(labels = results["Algorithm"].unique())
fig7.savefig("Plots/LineGraphAccuracynSingleModelRegions" + dataType + "Classifiers.png", dpi=150)


fig8,ax1 = plt.subplots()
ax1.plot(results.pivot(columns = "Algorithm", values = "MeanMinOverlapRegion", index = "Threshold"))
ax1.set_ylabel('Mean minimum overlap required')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
ax1.legend(labels = results["Algorithm"].unique())
fig8.savefig("Plots/LineGraphAccuracyMeanMinOverlapRegion" + dataType + "Classifiers.png", dpi=150)
