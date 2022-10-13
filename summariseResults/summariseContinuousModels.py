## create plots of cell type predictions using continuous DNAm levels

import utils
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# accuracy threshold
thres = 0.8

## process command line information
resultsPath = sys.argv[1]

os.chdir(resultsPath)


## create output folder for plots
if not os.path.exists("Plots"):
    os.makedirs("Plots")

## list files in results folder
allFiles = os.listdir()
allFiles = list(filter(lambda x:'ContinuousClassifier' in x, allFiles))

allDat = {}


## load results
for modelType in ["KNN", "NBayes", "RandFor", "SVM"]:
    subFiles = list(filter(lambda x:modelType in x, allFiles))
    print(str(len(subFiles)) + " files found for model type " + modelType)
    allDat[modelType] = pd.concat([utils.loadResults(x,"continuous") for x in subFiles]).sort_values(by=["Chr", "Position", "nCpG"])
    ## count
    print(str(len(allDat[modelType])) + " models loaded for model type " + modelType)
    ## add Density column
    allDat[modelType]["Density"] = allDat[modelType]['WindowSize']/allDat[modelType]['nCpG']


## summarize the parameters of the models
allDat[modelType]['nCpG'].describe()
allDat[modelType]['WindowSize'].describe()
allDat[modelType]['Density'].describe()

minCpG = allDat[modelType].nCpG.min()

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


## boxplot of accuracy statistics
fig2, ax1 = plt.subplots()
ax1.violinplot((allDat["KNN"]['MeanAccuracy'],allDat["NBayes"]['MeanAccuracy'],allDat["RandFor"]['MeanAccuracy'],allDat["SVM"]['MeanAccuracy']),showextrema=True, showmedians=True)
plt.xticks((1,2,3,4), ("KNN", "Naive Bayes", "Random Forest", "SVM"))
ax1.set(xlim = [0.5,4.5], ylim = [0,1], xlabel='Algorithm', ylabel='Mean accuracy across CV')
ax1.grid(True)
fig2.savefig("Plots/ViolinplotAccuracyContinuousClassifiers.png", dpi=150)


## compare ML algorithms

## merge into a single data.frame to determine best algorithm for each model
mergeDf = pd.DataFrame({'svm': allDat["SVM"]["MeanAccuracy"], 'knn': allDat["KNN"]["MeanAccuracy"], 'naiveBayes': allDat["NBayes"]["MeanAccuracy"], 'randomForest': allDat["RandFor"]["MeanAccuracy"]})
## identify for each model the best ML algorithm
maxMean = mergeDf.max(1)
mergeDf.idxmax(1).value_counts()
mergeDf['best'] = maxMean


## count cumulative sum of number of models with accuracy > x.

accuracyBins = np.arange(0,1.01,0.05)
	
cumSumTotals = mergeDf.apply(utils.cumSumAccuracy, axis=0, bins = accuracyBins)

fig3, ax1 = plt.subplots()
ax1.plot(np.flip(np.arange(0,1,0.05)), cumSumTotals['knn'], label = "KNN")
ax1.plot(np.flip(np.arange(0,1,0.05)), cumSumTotals['naiveBayes'], label = "Naive Bayes")
ax1.plot(np.flip(np.arange(0,1,0.05)), cumSumTotals['randomForest'], label = "Random Forest")
ax1.plot(np.flip(np.arange(0,1,0.05)), cumSumTotals['svm'], label = "SVM")
ax1.plot(np.flip(np.arange(0,1,0.05)), cumSumTotals['best'], label = "Best")
ax1.set_xlabel('Mean accuracy across CV')  # Add an x-label to the axes.
ax1.set_ylabel('Number of models (x1000)')  # Add a y-label to the axes.
y_vals = ax1.get_yticks()
ax1.yaxis.set_major_locator(mticker.FixedLocator(y_vals))
ax1.set_yticklabels(['{:.0f}'.format(x / 1000) for x in y_vals])
ax1.legend()
ax1.grid(True)
fig3.savefig("Plots/LineGraphCumulativeAccuracyContinuousClassifiers.png", dpi=150)

## for each model how many algorithms produce accuracte predictions?
fig4, axs = plt.subplots()
x = np.arange(5)  # the label locations
width = 1  # the width of the bars
axs.bar(x, (mergeDf.drop('best', axis = 1) > thres).sum(1).value_counts(sort=False).values, width)

# Add some text for labels, title and custom x-axis tick labels, etc.
axs.set_ylabel('Number of models')
axs.set_xlabel('Number of algorithms')
axs.set_xticks(x)
fig4.savefig("Plots/BarchartNumberofPredictiveAlgorithmsContinuousClassifiers.png", dpi=150)


## summarise as a function of model characteristics
fig5, (ax1,ax2, ax3) = plt.subplots(1,3)
groupedCpG = allDat["KNN"].groupby(allDat["KNN"].nCpG).mean()
ax1.plot(np.arange(minCpG,max(allDat["KNN"]['nCpG'])+1,1), np.asarray(groupedCpG['MeanAccuracy']), label = "KNN")

groupedCpG = allDat["NBayes"].groupby(allDat["NBayes"].nCpG).mean()
ax1.plot(np.arange(minCpG,max(allDat["NBayes"]['nCpG'])+1,1), np.asarray(groupedCpG['MeanAccuracy']), label = "Naive Bayes")
groupedCpG = allDat["RandFor"].groupby(allDat["RandFor"].nCpG).mean()
ax1.plot(np.arange(minCpG,max(allDat["RandFor"]['nCpG'])+1,1), np.asarray(groupedCpG['MeanAccuracy']), label = "Random Forest")
groupedCpG = allDat["SVM"].groupby(allDat["SVM"].nCpG).mean()
ax1.plot(np.arange(minCpG,max(allDat["SVM"]['nCpG'])+1,1), np.asarray(groupedCpG['MeanAccuracy']), label = "SVM")
ax1.set_ylabel('Mean accuracy')  
ax1.set_xlabel('Number of CpGs')  
ax1.grid(True)

## plot against window size
spanBins = np.arange(0,10001,100)
windowBins = pd.cut(allDat["KNN"]['WindowSize'], bins = spanBins)

groupedSpan = allDat["KNN"].groupby(windowBins).mean()
ax2.plot(np.arange(100,10001,100), np.asarray(groupedSpan['MeanAccuracy']), label = "KNN")
groupedSpan = allDat["NBayes"].groupby(windowBins).mean()
ax2.plot(np.arange(100,10001,100), np.asarray(groupedSpan['MeanAccuracy']), label = "Naive Bayes")
groupedSpan = allDat["RandFor"].groupby(windowBins).mean()
ax2.plot(np.arange(100,10001,100), np.asarray(groupedSpan['MeanAccuracy']), label = "Random Forest")
groupedSpan = allDat["SVM"].groupby(windowBins).mean()
ax2.plot(np.arange(100,10001,100), np.asarray(groupedSpan['MeanAccuracy']), label = "SVM")
 
ax2.set_xlabel('Span of CpGs')  
ax2.grid(True)

## plot against density
densityBreaks = np.arange(0,5001,50)
densityBins = pd.cut(allDat["KNN"]['Density'], bins = densityBreaks)

groupedDensity = allDat["KNN"].groupby(densityBins).mean()
ax3.plot(densityBreaks[1:], np.asarray(groupedDensity['MeanAccuracy']), label = "KNN")
groupedDensity = allDat["NBayes"].groupby(densityBins).mean()
ax3.plot(densityBreaks[1:], np.asarray(groupedDensity['MeanAccuracy']), label = "Naive Bayes")
groupedDensity = allDat["RandFor"].groupby(densityBins).mean()
ax3.plot(densityBreaks[1:], np.asarray(groupedDensity['MeanAccuracy']), label = "Random Forest")
groupedDensity = allDat["SVM"].groupby(densityBins).mean()
ax3.plot(densityBreaks[1:], np.asarray(groupedDensity['MeanAccuracy']), label = "SVM")

ax3.set_xlabel('Density of CpGs')  
ax3.legend()
ax3.grid(True)
fig5.set_size_inches(12, 4)
fig5.savefig("Plots/LineGraphAccuracyAgainstModelProperties.png", dpi=150)


# summarise model properties as a function of accuracy
accuracyBreaks = np.arange(0.5,1,0.01)
fig6, (ax1,ax2, ax3) = plt.subplots(1,3)

accuracyBins = pd.cut(allDat["KNN"]['MeanAccuracy'], bins = accuracyBreaks)
knnByAccuracy = allDat["KNN"].groupby(accuracyBins).mean()
accuracyBins = pd.cut(allDat["NBayes"]['MeanAccuracy'], bins = accuracyBreaks)
nbayesByAccuracy = allDat["NBayes"].groupby(accuracyBins).mean()
accuracyBins = pd.cut(allDat["RandFor"]['MeanAccuracy'], bins = accuracyBreaks)
randforByAccuracy = allDat["RandFor"].groupby(accuracyBins).mean()
accuracyBins = pd.cut(allDat["SVM"]['MeanAccuracy'], bins = accuracyBreaks)
svmByAccuracy = allDat["SVM"].groupby(accuracyBins).mean()

ax1.plot(accuracyBreaks[:-1]+0.05, np.asarray(knnByAccuracy['nCpG']), label = "KNN")
ax1.plot(accuracyBreaks[:-1]+0.05, np.asarray(nbayesByAccuracy['nCpG']), label = "Naive Bayes")
ax1.plot(accuracyBreaks[:-1]+0.05, np.asarray(randforByAccuracy['nCpG']), label = "Random Forest")
ax1.plot(accuracyBreaks[:-1]+0.05, np.asarray(svmByAccuracy['nCpG']), label = "SVM")
ax1.set_ylabel('Number of CpGs')  
ax1.set_xlabel('Mean accuracy')  
ax1.grid(True)

ax2.plot(accuracyBreaks[:-1]+0.05, np.asarray(knnByAccuracy['WindowSize']), label = "KNN")
ax2.plot(accuracyBreaks[:-1]+0.05, np.asarray(nbayesByAccuracy['WindowSize']), label = "Naive Bayes")
ax2.plot(accuracyBreaks[:-1]+0.05, np.asarray(randforByAccuracy['WindowSize']), label = "Random Forest")
ax2.plot(accuracyBreaks[:-1]+0.05, np.asarray(svmByAccuracy['WindowSize']), label = "SVM")
ax2.set_ylabel('Span of CpGs (bp)')  
ax2.set_xlabel('Mean accuracy')  
ax2.grid(True)

ax3.plot(accuracyBreaks[:-1]+0.05, np.asarray(knnByAccuracy['Density']), label = "KNN")
ax3.plot(accuracyBreaks[:-1]+0.05, np.asarray(nbayesByAccuracy['Density']), label = "Naive Bayes")
ax3.plot(accuracyBreaks[:-1]+0.05, np.asarray(randforByAccuracy['Density']), label = "Random Forest")
ax3.plot(accuracyBreaks[:-1]+0.05, np.asarray(svmByAccuracy['Density']), label = "SVM")
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
fig6.savefig("Plots/LineGraphModelPropertiesAgainstAccuracy.png", dpi=150)
