


import utils
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


## process command line information
resultsPath = sys.argv[1]

os.chdir(resultsPath)

results = pd.read_csv("SummaryModelPerformanceByAccuracyContinuousClassifiers.csv", header = 0, names = ("Algorithm", "Threshold", "nModels", "MeanModelSize", "SDModelSize", "nRegions", "MeanRegionSize", "SDRegionSize", "TotalRegionLength", "MeanInterRegionSize", "SDInterRegionSize", "TotalInterRegionSize", "ProportionBasesInRegion", "MeannModelsRegion", "SDnModelsRegion", "MaxnModelsRegion", "nSingleModelRegions", "MeanMinOverlapRegion", "SDMinOverlapRegion"))

fig1,ax1 = plt.subplots()
ax1.plot(results.pivot(columns = "Algorithm", values = "nRegions", index = "Threshold"))
ax1.set_ylabel('Number of Regions')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
ax1.legend(labels = ['Best', 'KNN', 'NaiveBayes', "RandomForest", "SVM"])
fig1.savefig("Plots/LineGraphAccuracynRegionsContinuousClassifiers.png", dpi=150)


fig2,ax1 = plt.subplots()
ax1.plot(results.pivot(columns = "Algorithm", values = "MeanModelSize", index = "Threshold"))
ax1.set_ylabel('Mean span of CpGs (bp)')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
ax1.legend(labels = ['Best', 'KNN', 'NaiveBayes', "RandomForest", "SVM"])
fig2.savefig("Plots/LineGraphAccuracyMeanModelSizeContinuousClassifiers.png", dpi=150)

fig3,ax1 = plt.subplots()
ax1.plot(results.pivot(columns = "Algorithm", values = "ProportionBasesInRegion", index = "Threshold"))
ax1.set_ylabel('Proportion bases')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
ax1.legend(labels = ['Best', 'KNN', 'NaiveBayes', "RandomForest", "SVM"])
fig3.savefig("Plots/LineGraphAccuracyProportionBasesContinuousClassifiers.png", dpi=150)

fig4,ax1 = plt.subplots()
ax1.plot(results.pivot(columns = "Algorithm", values = "MeanRegionSize", index = "Threshold"))
ax1.set_ylabel('Mean region size (bp)')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
ax1.legend(labels = ['Best', 'KNN', 'NaiveBayes', "RandomForest", "SVM"])
fig4.savefig("Plots/LineGraphAccuracyMeanRegionSizeContinuousClassifiers.png", dpi=150)

fig5,ax1 = plt.subplots()
ax1.plot(results.pivot(columns = "Algorithm", values = "MeanInterRegionSize", index = "Threshold"))
ax1.set_ylabel('Mean gap between regions (bp)')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
ax1.legend(labels = ['Best', 'KNN', 'NaiveBayes', "RandomForest", "SVM"])
fig5.savefig("Plots/LineGraphAccuracyMeanInterRegionSizeContinuousClassifiers.png", dpi=150)

fig6,ax1 = plt.subplots()
ax1.plot(results.pivot(columns = "Algorithm", values = "MeannModelsRegion", index = "Threshold"))
ax1.set_ylabel('Mean number of models per region')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
ax1.legend(labels = ['Best', 'KNN', 'NaiveBayes', "RandomForest", "SVM"])
fig6.savefig("Plots/LineGraphAccuracyMeannModelsRegionContinuousClassifiers.png", dpi=150)

fig7,ax1 = plt.subplots()
ax1.plot(results.pivot(columns = "Algorithm", values = "nSingleModelRegions", index = "Threshold"))
ax1.set_ylabel('Number of single model regions')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
ax1.legend(labels = ['Best', 'KNN', 'NaiveBayes', "RandomForest", "SVM"])
fig7.savefig("Plots/LineGraphAccuracynSingleModelRegionsContinuousClassifiers.png", dpi=150)


fig8,ax1 = plt.subplots()
ax1.plot(results.pivot(columns = "Algorithm", values = "MeanMinOverlapRegion", index = "Threshold"))
ax1.set_ylabel('Mean minimum overlap required')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
ax1.legend(labels = ['Best', 'KNN', 'NaiveBayes', "RandomForest", "SVM"])
fig8.savefig("Plots/LineGraphAccuracyMeanMinOverlapRegionContinuousClassifiers.png", dpi=150)
