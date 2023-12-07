


import utils
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 15})


## process command line information
resultsPath = sys.argv[1]
dataType = sys.argv[2]

os.chdir(resultsPath)

results = pd.read_csv("SummaryModelPerformanceByAccuracy" + dataType + "Classifiers.csv", header = 0, names = ("Algorithm", "Threshold", "nModels", "MeanModelSize", "SDModelSize", "nRegions", "MeanRegionSize", "SDRegionSize", "TotalRegionLength", "MeanInterRegionSize", "SDInterRegionSize", "TotalInterRegionSize", "ProportionBasesInRegion", "MeannModelsRegion", "SDnModelsRegion", "MaxnModelsRegion", "nSingleModelRegions", "MeanMinOverlapRegion", "SDMinOverlapRegion"))

fig1,ax1 = plt.subplots()
ax1.plot(results.pivot(columns = "Algorithm", values = "nRegions", index = "Threshold")[["KNN", "NaiveBayes", "RandomForest", "SVM", "BestAccuracy"]])
ax1.set_ylabel('Number of Regions')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
ax1.legend(labels = ["KNN", "Naive Bayes", "Random Forest", "SVM", "Best"])
# This should be called after all axes have been added
fig1.tight_layout()
fig1.savefig("Plots/LineGraphAccuracynRegions" + dataType + "Classifiers.png", dpi=150)

fig1,ax1 = plt.subplots()
ax1.plot(results.pivot(columns = "Algorithm", values = "nRegions", index = "Threshold")[["BestAccuracy"]])
ax1.set_ylabel('Number of Regions')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
# This should be called after all axes have been added
fig1.tight_layout()
fig1.savefig("Plots/LineGraphAccuracynRegions" + dataType + "ClassifiersBestOnly.png", dpi=150)



fig2,ax1 = plt.subplots()
ax1.plot(results.pivot(columns = "Algorithm", values = "MeanModelSize", index = "Threshold")[["KNN", "NaiveBayes", "RandomForest", "SVM", "BestAccuracy"]])
ax1.set_ylabel('Mean span of CpGs (bp)')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
ax1.legend(labels = ["KNN", "Naive Bayes", "Random Forest", "SVM", "Best"])
fig2.savefig("Plots/LineGraphAccuracyMeanModelSize" + dataType + "Classifiers.png", dpi=150)

xvar = np.array(results["Threshold"].unique())
mu = np.array(results.pivot(columns = "Algorithm", values = "MeanModelSize", index = "Threshold")["BestAccuracy"])
sigma = np.array(results.pivot(columns = "Algorithm", values = "SDModelSize", index = "Threshold")["BestAccuracy"])
fig2,ax1 = plt.subplots()
ax1.fill_between(xvar, mu-sigma, mu+sigma, alpha = 0.5)
ax1.plot(xvar, mu)
ax1.set_ylabel('Mean span of CpGs (bp)')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
fig2.savefig("Plots/LineGraphAccuracyMeanModelSize" + dataType + "ClassifiersBestOnly.png", dpi=150)

fig3,ax1 = plt.subplots()
ax1.plot(results.pivot(columns = "Algorithm", values = "ProportionBasesInRegion", index = "Threshold")[["KNN", "NaiveBayes", "RandomForest", "SVM", "BestAccuracy"]])
ax1.set_ylabel('Proportion bases')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
ax1.legend(labels = ["KNN", "Naive Bayes", "Random Forest", "SVM", "Best"])
fig3.savefig("Plots/LineGraphAccuracyProportionBases" + dataType + "Classifiers.png", dpi=150)

fig3,ax1 = plt.subplots()
ax1.plot(results.pivot(columns = "Algorithm", values = "ProportionBasesInRegion", index = "Threshold")["BestAccuracy"])
ax1.set_ylabel('Proportion bases')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
fig3.savefig("Plots/LineGraphAccuracyProportionBases" + dataType + "ClassifiersBestOnly.png", dpi=150)


fig4,ax1 = plt.subplots()
ax1.plot(results.pivot(columns = "Algorithm", values = "MeanRegionSize", index = "Threshold")[["KNN", "NaiveBayes", "RandomForest", "SVM", "BestAccuracy"]])
ax1.set_ylabel('Mean region size')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
ax1.legend(labels = ["KNN", "Naive Bayes", "Random Forest", "SVM", "Best"])
fig4.savefig("Plots/LineGraphAccuracyMeanRegionSize" + dataType + "Classifiers.png", dpi=150)

mu = np.array(results.pivot(columns = "Algorithm", values = "MeanRegionSize", index = "Threshold")["BestAccuracy"])
sigma = np.array(results.pivot(columns = "Algorithm", values = "SDRegionSize", index = "Threshold")["BestAccuracy"])

fig4,ax1 = plt.subplots()
ax1.fill_between(xvar, mu-sigma, mu+sigma, alpha = 0.5)
ax1.plot(xvar, mu)
ax1.set_ylabel('Mean region size')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
ax1.legend(labels = ["KNN", "Naive Bayes", "Random Forest", "SVM", "Best"])
fig4.savefig("Plots/LineGraphAccuracyMeanRegionSize" + dataType + "ClassifiersBestOnly.png", dpi=150)


fig5,ax1 = plt.subplots()
ax1.plot(results.pivot(columns = "Algorithm", values = "MeanInterRegionSize", index = "Threshold")[["KNN", "NaiveBayes", "RandomForest", "SVM", "BestAccuracy"]])
ax1.set_ylabel('Mean gap between regions')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
ax1.legend(labels = ["KNN", "Naive Bayes", "Random Forest", "SVM", "Best"])
fig5.savefig("Plots/LineGraphAccuracyMeanInterRegionSize" + dataType + "Classifiers.png", dpi=150)


mu = np.array(results.pivot(columns = "Algorithm", values = "MeanInterRegionSize", index = "Threshold")["BestAccuracy"])
sigma = np.array(results.pivot(columns = "Algorithm", values = "SDInterRegionSize", index = "Threshold")["BestAccuracy"])
fig5,ax1 = plt.subplots()
ax1.fill_between(xvar, mu-sigma, mu+sigma, alpha = 0.5)
ax1.plot(xvar, mu)
ax1.set_ylabel('Mean gap between regions')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
fig5.savefig("Plots/LineGraphAccuracyMeanInterRegionSize" + dataType + "ClassifiersBestOnly.png", dpi=150)

fig6,ax1 = plt.subplots()
ax1.plot(results.pivot(columns = "Algorithm", values = "MeannModelsRegion", index = "Threshold")[["KNN", "NaiveBayes", "RandomForest", "SVM", "BestAccuracy"]])
ax1.set_ylabel('Mean number of models per region')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
ax1.legend(labels = ["KNN", "Naive Bayes", "Random Forest", "SVM", "Best"])
fig6.savefig("Plots/LineGraphAccuracyMeannModelsRegion" + dataType + "Classifiers.png", dpi=150)

mu = np.array(results.pivot(columns = "Algorithm", values = "MeannModelsRegion", index = "Threshold")["BestAccuracy"])
sigma = np.array(results.pivot(columns = "Algorithm", values = "SDnModelsRegion", index = "Threshold")["BestAccuracy"])
fig6,ax1 = plt.subplots()
ax1.fill_between(xvar, mu-sigma, mu+sigma, alpha = 0.5)
ax1.plot(xvar, mu)
ax1.set_ylabel('Mean number of models per region')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
fig6.savefig("Plots/LineGraphAccuracyMeannModelsRegion" + dataType + "ClassifiersBestOnly.png", dpi=150)


fig7,ax1 = plt.subplots()
ax1.plot(results.pivot(columns = "Algorithm", values = "nSingleModelRegions", index = "Threshold")[["KNN", "NaiveBayes", "RandomForest", "SVM", "BestAccuracy"]])
ax1.set_ylabel('Number of single model regions')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
ax1.legend(labels = ["KNN", "Naive Bayes", "Random Forest", "SVM", "Best"])
fig7.savefig("Plots/LineGraphAccuracynSingleModelRegions" + dataType + "Classifiers.png", dpi=150)

fig7,ax1 = plt.subplots()
ax1.plot(results.pivot(columns = "Algorithm", values = "nSingleModelRegions", index = "Threshold")["BestAccuracy"])
ax1.set_ylabel('Number of single model regions')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
fig7.savefig("Plots/LineGraphAccuracynSingleModelRegions" + dataType + "ClassifiersBestOnly.png", dpi=150)


fig8,ax1 = plt.subplots()
ax1.plot(results.pivot(columns = "Algorithm", values = "MeanMinOverlapRegion", index = "Threshold")[["KNN", "NaiveBayes", "RandomForest", "SVM", "BestAccuracy"]])
ax1.set_ylabel('Mean minimum overlap required')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
ax1.legend(labels = ["KNN", "Naive Bayes", "Random Forest", "SVM", "Best"])
fig8.savefig("Plots/LineGraphAccuracyMeanMinOverlapRegion" + dataType + "Classifiers.png", dpi=150)

mu = np.array(results.pivot(columns = "Algorithm", values = "MeanMinOverlapRegion", index = "Threshold")["BestAccuracy"])
sigma = np.array(results.pivot(columns = "Algorithm", values = "SDMinOverlapRegion", index = "Threshold")["BestAccuracy"])
fig8,ax1 = plt.subplots()
ax1.fill_between(xvar, mu-sigma, mu+sigma, alpha = 0.5)
ax1.plot(xvar, mu)
ax1.set_ylabel('Mean minimum overlap required')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
fig8.savefig("Plots/LineGraphAccuracyMeanMinOverlapRegion" + dataType + "ClassifiersBestOnly.png", dpi=150)
