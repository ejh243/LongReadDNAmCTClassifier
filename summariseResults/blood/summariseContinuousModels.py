

def cumSumAccuracy(data, bins):
	return np.asarray(np.cumsum(np.flip(pd.cut(data, bins = bins).value_counts(sort = False))))


def gaps(granges):
    ## doesn't make sens unless ranges contains non-overlapping regions
    ## get chromosome size info
    chrSizes = pr.data.chromsizes()
    chrList = np.unique(granges.Chromosome)
    newChr = []
    newStart = []
    newEnd = []
    for chr in chrList:
        boolIndex = granges.Chromosome == chr
        newChr.extend([chr]*(sum(boolIndex)+1)) ## need an extra chr
        newStart.extend([0])
        newStart.extend((granges.End[boolIndex].values+1).tolist())
        newEnd.extend((granges.Start[boolIndex].values - 1).tolist())
        newEnd.extend(chrSizes.End[chrSizes.Chromosome == chr].tolist())
    return pr.PyRanges(chromosomes = newChr, starts = newStart, ends = newEnd)


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyranges as pr
import seaborn as sns
from upsetplot import from_contents
from upsetplot import UpSet

os.chdir("/mnt/data1/Eilis/Projects/Asthma/ClassifyCellTypes/")

chr = 22

## set colors for plotting
colors = ["royalblue", "darkorange", "forestgreen", "darkred", "darkorchid"]


colNames = ("Chr", "Position", "nCpG", "WindowSize", "MeanAccuracy", "SDAccuracy")


svm = pd.read_csv("Results/Continuous/ContinuousClassifierSVM_5CT_Chr" + str(chr) + ".csv", header = 0, names = colNames)
nBayes = pd.read_csv("Results/Continuous/ContinuousClassifierNBayes_5CT_Chr" + str(chr) + ".csv", header = 0, names = colNames)
knn = pd.read_csv("Results/Continuous/ContinuousClassifierKNN_5CT_Chr" + str(chr) + ".csv", header = 0, names = colNames)
randFor = pd.read_csv("Results/Continuous/ContinuousClassifierRandFor_5CT_Chr" + str(chr) + ".csv", header = 0, names = colNames)


## summarize the parameters of the models
fig1, (ax1, ax2, ax3) = plt.subplots(nrows=1,ncols=3)
n1, bins1, patches1 = ax1.hist(svm['nCpG'], int(max(svm['nCpG']))-2, density=False, facecolor='g', alpha=0.75)
## histogram of number of CpGs
ax1.set(xlim = [2, max(svm['nCpG'])], ylim = [0, max(n1)], xlabel='Number of CpGs',
ylabel='Number of models')
ax1.grid(True)

## histogram of the window size
n2, bins2, patches2 = ax2.hist(svm['WindowSize'], 50, density=False, facecolor='g', alpha=0.75)
ax2.set(xlabel='Span of CpGs (bp)', ylabel='')
ax2.grid(True)

## histogram of probe density
n3, bins3, patches3 = ax3.hist(svm['WindowSize']/svm['nCpG'], 50, density=False, facecolor='g', alpha=0.75)
ax3.set(xlabel='Density of CpGs (bp)', ylabel='')
ax3.grid(True)

# Save files in png format
fig1.set_size_inches(12, 4)
fig1.savefig("Results/Pilot/HistogramModelCharacteristics.png", dpi=150)



## boxplot of accuracy statistics
fig2, ax1 = plt.subplots()
ax1.violinplot((knn['MeanAccuracy'],nBayes['MeanAccuracy'],randFor['MeanAccuracy'],svm['MeanAccuracy']),showextrema=True, showmedians=True)
plt.xticks((1,2,3,4), ("KNN", "Naive Bayes", "Random Forest", "SVM"))
ax1.set(xlim = [0.5,4.5], ylim = [0,1], xlabel='Algorithm', ylabel='Mean accuracy across CV')
ax1.grid(True)
fig2.savefig("Results/Pilot/BoxplotAccuracyContinuousClassifiers.png", dpi=150)


## compare ML algorithms

## merge into a single data.frame to determine best algorithm for each model
mergeDf = pd.DataFrame({'svm': svm["MeanAccuracy"], 'knn': knn["MeanAccuracy"], 'naiveBayes': nBayes["MeanAccuracy"], 'randomForest': randFor["MeanAccuracy"]})
## identify for each model the best ML algorithm
maxMean = mergeDf.max(1)
mergeDf.idxmax(1).value_counts()
mergeDf['best'] = maxMean


## count cumulative sum of number of models with accuracy > x.

accuracyBins = np.arange(0,1.01,0.05)
	
cumSumTotals = mergeDf.apply(cumSumAccuracy, axis=0, bins = accuracyBins)

fig3, ax1 = plt.subplots()
ax1.plot(np.flip(np.arange(0,1,0.05)), cumSumTotals['knn'], label = "KNN")
ax1.plot(np.flip(np.arange(0,1,0.05)), cumSumTotals['naiveBayes'], label = "Naive Bayes")
ax1.plot(np.flip(np.arange(0,1,0.05)), cumSumTotals['randomForest'], label = "Random Forest")
ax1.plot(np.flip(np.arange(0,1,0.05)), cumSumTotals['svm'], label = "SVM")
ax1.plot(np.flip(np.arange(0,1,0.05)), cumSumTotals['best'], label = "Best")
ax1.set_xlabel('Mean accuracy across CV')  # Add an x-label to the axes.
ax1.set_ylabel('Number of models')  # Add a y-label to the axes.
ax1.legend()
ax1.grid(True)
fig3.savefig("Results/Pilot/LineGraphCumulativeAccuracyContinuousClassifiers.png", dpi=150)

## for each model how many algorithms produce accuracte predictions?
thres = 0.8
fig4, axs = plt.subplots()
x = np.arange(5)  # the label locations
width = 1  # the width of the bars
axs.bar(x, (mergeDf.drop('best', axis = 1) > thres).sum(1).value_counts(sort=False).values, width)

# Add some text for labels, title and custom x-axis tick labels, etc.
axs.set_ylabel('Number of models')
axs.set_xlabel('Number of algorithms')
axs.set_xticks(x)
fig4.savefig("Results/Pilot/BarchartNumberofPredictiveAlgorithmsContinuousClassifiers.png", dpi=150)


fig5, (ax1,ax2, ax3) = plt.subplots(1,3)
groupedCpG = knn.groupby(knn.nCpG).mean()
ax1.plot(np.arange(2,max(knn['nCpG'])+1,1), np.asarray(groupedCpG['MeanAccuracy']), label = "KNN")
groupedCpG = nBayes.groupby(nBayes.nCpG).mean()
ax1.plot(np.arange(2,max(nBayes['nCpG'])+1,1), np.asarray(groupedCpG['MeanAccuracy']), label = "Naive Bayes")
groupedCpG = randFor.groupby(randFor.nCpG).mean()
ax1.plot(np.arange(2,max(randFor['nCpG'])+1,1), np.asarray(groupedCpG['MeanAccuracy']), label = "Random Forest")
groupedCpG = svm.groupby(svm.nCpG).mean()
ax1.plot(np.arange(2,max(svm['nCpG'])+1,1), np.asarray(groupedCpG['MeanAccuracy']), label = "SVM")
ax1.set_ylabel('Mean accuracy')  
ax1.set_xlabel('Number of CpGs')  
ax1.grid(True)

## plot against window size
spanBins = np.arange(0,10001,100)
windowBins = pd.cut(knn['WindowSize'], bins = spanBins)

groupedSpan = knn.groupby(windowBins).mean()
ax2.plot(np.arange(100,10001,100), np.asarray(groupedSpan['MeanAccuracy']), label = "KNN")
groupedSpan = nBayes.groupby(windowBins).mean()
ax2.plot(np.arange(100,10001,100), np.asarray(groupedSpan['MeanAccuracy']), label = "Naive Bayes")
groupedSpan = randFor.groupby(windowBins).mean()
ax2.plot(np.arange(100,10001,100), np.asarray(groupedSpan['MeanAccuracy']), label = "Random Forest")
groupedSpan = svm.groupby(windowBins).mean()
ax2.plot(np.arange(100,10001,100), np.asarray(groupedSpan['MeanAccuracy']), label = "SVM")
 
ax2.set_xlabel('Span of CpGs')  
ax2.grid(True)

## plot against density
densityBreaks = np.arange(0,5001,50)
densityBins = pd.cut(knn['WindowSize']/knn['nCpG'], bins = densityBreaks)

groupedDensity = knn.groupby(densityBins).mean()
ax3.plot(densityBreaks[1:], np.asarray(groupedDensity['MeanAccuracy']), label = "KNN")
groupedDensity = nBayes.groupby(densityBins).mean()
ax3.plot(densityBreaks[1:], np.asarray(groupedDensity['MeanAccuracy']), label = "Naive Bayes")
groupedDensity = randFor.groupby(densityBins).mean()
ax3.plot(densityBreaks[1:], np.asarray(groupedDensity['MeanAccuracy']), label = "Random Forest")
groupedDensity = svm.groupby(densityBins).mean()
ax3.plot(densityBreaks[1:], np.asarray(groupedDensity['MeanAccuracy']), label = "SVM")

ax3.set_xlabel('Density of CpGs')  
ax3.legend()
ax3.grid(True)
fig5.set_size_inches(12, 4)
fig5.savefig("Results/Pilot/LineGraphAccuracyAgainstModelProperties.png", dpi=150)


## create set of genomic regions

granges = pr.PyRanges(chromosomes = "chr" + str(chr), starts = svm["Position"], ends = svm["Position"]+svm["WindowSize"]) 
setattr(granges, "SVM", svm['MeanAccuracy'])
setattr(granges, "KNN", knn['MeanAccuracy'])
setattr(granges, "NaiveBayes", nBayes['MeanAccuracy'])
setattr(granges, "RandomForest", randFor['MeanAccuracy'])
setattr(granges, "BestAccuracy", mergeDf['best'])
setattr(granges, "nCpG", svm["nCpG"])

## how many genomic regions with reasonable predictive power

allRows = []

## identify models with sufficinet accuracy
col = ("SVM", "KNN", "NaiveBayes", "RandomForest", "BestAccuracy")
for threshold in np.arange(0.5,1.00, 0.01):
    for ml in col:
        output = []
        output.append(ml)
        output.append(threshold)
        boolIndex = getattr(granges,ml) > threshold
        ## count number of models
        output.append(sum(boolIndex))
        
        ## summarise length of models
        output.append(np.mean(granges[boolIndex].End - granges[boolIndex].Start))
        output.append(np.std(granges[boolIndex].End - granges[boolIndex].Start))
        
        ## count number of genomic regions - merge into non-overlapping set
        regions = granges[boolIndex].merge()
        output.append(len(regions))
        ## summarise length of regions
        output.append(np.mean(regions.End - regions.Start))
        output.append(np.std(regions.End - regions.Start))
        output.append(sum(regions.End - regions.Start))
        
        ## summarise gaps between regions
        interRegions = gaps(regions)
        output.append(np.mean(interRegions.End - interRegions.Start))
        output.append(np.std(interRegions.End - interRegions.Start))
        output.append(sum(interRegions.End - interRegions.Start))
        
        ## calculate proportion of bases with model
        output.append(sum(regions.End - regions.Start)/(sum(regions.End - regions.Start)+sum(interRegions.End - interRegions.Start)))
        
        ## how many models within each region
        modelOverlaps = regions.coverage(granges[boolIndex])
        output.append(np.mean(modelOverlaps.NumberOverlaps))
        output.append(np.std(modelOverlaps.NumberOverlaps))
        output.append(np.max(modelOverlaps.NumberOverlaps))
        
        ## how many regions with a single model?
        output.append(sum(modelOverlaps.NumberOverlaps == 1))
        
        ## within each region what is the minimum overlap?
        
        minWindow = []
        for i in range(len(regions)):
            ## if only 1 model then minimum is the size of the region
            if modelOverlaps.NumberOverlaps[i] == 1:
                minWindow.append(modelOverlaps.End[i]-modelOverlaps.Start[i])
            else:
                ## find all models within this region
                start = modelOverlaps.Start[i]
                end = modelOverlaps.End[i]
                ## find smallest starting region 
                startIndex = (granges.Start == start) & (boolIndex)
                startModel = granges[startIndex]
                startSmall = min(startModel.End - startModel.Start)
                ## find the smallest end region 
                endIndex = (granges.End <= end) & (boolIndex)
                endModel = granges[endIndex]
                endSmall = min(endModel.End - endModel.Start)
                ## take the maximum of these two values
                minWindow.append(max([startSmall, endSmall]))
        
        output.append(np.mean(minWindow))
        output.append(np.std(minWindow))
        allRows.append(output)
 


results = pd.DataFrame(allRows, columns = ("Algorithm", "Threshold", "nModels", "MeanModelSize", "SDModelSize", "nRegions", "MeanRegionSize", "SDRegionSize", "TotalRegionLength", "MeanInterRegionSize", "SDInterRegionSize", "TotalInterRegionSize", "ProportionBasesInRegion", "MeannModelsRegion", "SDnModelsRegion", "MaxnModelsRegion", "nSingleModelRegions", "MeanMinOverlapRegion", "SDMinOverlapRegion"))

results.to_csv("Results/Pilot/SummaryModelPerformanceByAccuracyContinuousClassifiers.csv")

fig7,ax1 = plt.subplots()
ax1.plot(results.pivot(columns = "Algorithm", values = "nRegions", index = "Threshold"))
ax1.set_ylabel('Number of Regions')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
fig7.savefig("Results/Pilot/LineGraphAccuracynRegionsContinuousClassifiers.png", dpi=150)


fig7,ax1 = plt.subplots()
ax1.plot(results.pivot(columns = "Algorithm", values = "MeanModelSize", index = "Threshold"))
ax1.set_ylabel('Mean span of CpGs (bp)')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
fig7.savefig("Results/Pilot/LineGraphAccuracyMeanModelSizeContinuousClassifiers.png", dpi=150)

fig7,ax1 = plt.subplots()
ax1.plot(results.pivot(columns = "Algorithm", values = "ProportionBasesInRegion", index = "Threshold"))
ax1.set_ylabel('Proportion bases')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
fig7.savefig("Results/Pilot/LineGraphAccuracyProportionBasesContinuousClassifiers.png", dpi=150)

fig7,ax1 = plt.subplots()
ax1.plot(results.pivot(columns = "Algorithm", values = "MeanRegionSize", index = "Threshold"))
ax1.set_ylabel('Mean region size (bp)')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
fig7.savefig("Results/Pilot/LineGraphAccuracyMeanRegionSizeContinuousClassifiers.png", dpi=150)

fig7,ax1 = plt.subplots()
ax1.plot(results.pivot(columns = "Algorithm", values = "MeanInterRegionSize", index = "Threshold"))
ax1.set_ylabel('Mean gap between regions (bp)')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
fig7.savefig("Results/Pilot/LineGraphAccuracyMeanInterRegionSizeContinuousClassifiers.png", dpi=150)

fig7,ax1 = plt.subplots()
ax1.plot(results.pivot(columns = "Algorithm", values = "MeannModelsRegion", index = "Threshold"))
ax1.set_ylabel('Mean number of models per region')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
fig7.savefig("Results/Pilot/LineGraphAccuracyMeannModelsRegionContinuousClassifiers.png", dpi=150)

fig7,ax1 = plt.subplots()
ax1.plot(results.pivot(columns = "Algorithm", values = "nSingleModelRegions", index = "Threshold"))
ax1.set_ylabel('Number of single model regions')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
fig7.savefig("Results/Pilot/LineGraphAccuracynSingleModelRegionsContinuousClassifiers.png", dpi=150)


fig7,ax1 = plt.subplots()
ax1.plot(results.pivot(columns = "Algorithm", values = "MeanMinOverlapRegion", index = "Threshold")
ax1.set_ylabel('Mean minimum overlap required')  
ax1.set_xlabel('Accuracy Threshold')
ax1.grid(True)
fig7.savefig("Results/Pilot/LineGraphAccuracyMeanMinOverlapRegionContinuousClassifiers.png", dpi=150)
