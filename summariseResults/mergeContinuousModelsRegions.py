import sys
import utils
import os
import pandas as pd
import numpy as np
import pyranges as pr


## process command line information
resultsPath = sys.argv[1]

os.chdir(resultsPath)

## list files in results folder
allFiles = os.listdir()
allFiles = list(filter(lambda x:'ContinuousClassifier' in x, allFiles))

allDat = {}


## load results
for modelType in ["KNN", "NBayes", "RandFor", "SVM"]:
    subFiles = list(filter(lambda x:modelType in x, allFiles))
    print(str(len(subFiles)) + " files found for model type " + modelType)
    allDat[modelType] = pd.concat(list(map(utils.loadResults, subFiles))).sort_values(by=["Chr", "Position", "nCpG"])
    ## count
    print(str(len(allDat[modelType])) + " models loaded for model type " + modelType)
    ## add Density column
    allDat[modelType]["Density"] = allDat[modelType]['WindowSize']/allDat[modelType]['nCpG']


## merge into a single data.frame to determine best algorithm for each model
mergeDf = pd.DataFrame({'svm': allDat["SVM"]["MeanAccuracy"], 'knn': allDat["KNN"]["MeanAccuracy"], 'naiveBayes': allDat["NBayes"]["MeanAccuracy"], 'randomForest': allDat["RandFor"]["MeanAccuracy"]})
## identify for each model the best ML algorithm
maxMean = mergeDf.max(1)
mergeDf.idxmax(1).value_counts()
mergeDf['best'] = maxMean

## create set of genomic regions

granges = pr.PyRanges(chromosomes = allDat["SVM"]["Chr"].astype("int"), starts = allDat["SVM"]["Position"], ends = allDat["SVM"]["Position"]+allDat["SVM"]["WindowSize"]) 
setattr(granges, "SVM", allDat["SVM"]['MeanAccuracy'])
setattr(granges, "KNN", allDat["KNN"]['MeanAccuracy'])
setattr(granges, "NaiveBayes", allDat["NBayes"]['MeanAccuracy'])
setattr(granges, "RandomForest", allDat["RandFor"]['MeanAccuracy'])
setattr(granges, "BestAccuracy", mergeDf['best'])
setattr(granges, "nCpG", allDat["SVM"]["nCpG"])

## how many genomic regions with reasonable predictive power

allRows = []

## identify models with sufficinet accuracy
col = ("SVM", "KNN", "NaiveBayes", "RandomForest", "BestAccuracy")
for threshold in np.arange(0.7,1.00, 0.02):
    print("Aggregating models with accuracy > " + str(threshold))
    for ml in col:
        
        output = []
        output.append(ml)
        output.append(threshold)
        boolIndex = getattr(granges,ml) > threshold
        ## count number of models
        #print("Aggregating " + str(sum(boolIndex)) + "" + col + " models.")
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
        interRegions = utils.gaps(regions)
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
            if modelOverlaps.NumberOverlaps.to_numpy()[i] == 1:
                minWindow.append(modelOverlaps.End.to_numpy()[i]-modelOverlaps.Start.to_numpy()[i])
            else:
                ## find all models within this region
                start = modelOverlaps.Start.to_numpy()[i]
                end = modelOverlaps.End.to_numpy()[i]
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

results.to_csv("SummaryModelPerformanceByAccuracyContinuousClassifiers.csv")
