import sys
import utils
import os
import pandas as pd
import numpy as np
import pyranges as pr


## process command line information
resultsPath = sys.argv[1]
chr = sys.argv[2]
nCT = sys.argv[3]

os.chdir(resultsPath)

## list files in results folder
allFiles = os.listdir()
allFiles = list(filter(lambda x:'BinaryClassifier' in x, allFiles))
chrFiles = list(filter(lambda x:'_Chr'+chr+'_' in x, allFiles))

allDat = {}
modelOpts = ["KNN", "NBayes", "RandFor", "SVM"]
modelMissing = []
## load results
for modelType in modelOpts:
    subFiles = list(filter(lambda x:modelType in x, chrFiles))
    print(str(len(subFiles)) + " files found for model type " + modelType)
    if len(subFiles) == 1:
       allDat[modelType] = utils.loadResults(subFiles[0],"binary", nCT).sort_values(by=["Chr", "Position", "nCpG"])
       ## count
       print(str(len(allDat[modelType])) + " models loaded for model type " + modelType)
       ## add Density column
       allDat[modelType]["Density"] = allDat[modelType]['WindowSize']/allDat[modelType]['nCpG']
       ## calc overall accuracy
       sensCols = [col for col in allDat[modelType].columns if col.endswith('sensitivity')]
       allDat[modelType]["MeanAccuracy"] = allDat[modelType][sensCols].mean(1)    
    else:
        modelMissing.append(modelType)

if modelMissing is not None and len(modelMissing) > 0:
    modelOpts = [modelType for modelType in modelOpts if modelType not in modelMissing]


## merge into a single data.frame to determine best algorithm for each model
mergeDf = pd.concat([allDat[x]["MeanAccuracy"] for x in modelOpts], axis = 1)
mergeDf.columns = modelOpts
## identify for each model the best ML algorithm
maxMean = mergeDf.max(1)
mergeDf.idxmax(1).value_counts()
mergeDf['best'] = maxMean

## create set of genomic regions

granges = pr.PyRanges(chromosomes = allDat[modelOpts[0]]["Chr"].astype("int"), starts = allDat[modelOpts[0]]["Position"], ends = allDat[modelOpts[0]]["Position"]+allDat[modelOpts[0]]["WindowSize"])
for each in modelOpts:
    setattr(granges, each, allDat[each]['MeanAccuracy'])

setattr(granges, "BestAccuracy", mergeDf['best'])
setattr(granges, "nCpG", allDat[modelOpts[0]]["nCpG"])

## how many genomic regions with reasonable predictive power

allRows = []

## identify models with sufficinet accuracy
col = modelOpts + ["BestAccuracy"]
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
        if sum(boolIndex) > 0:
        
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
        else:
            output = output +  ["NA"] * 16
        allRows.append(output)
    results = pd.DataFrame(allRows, columns = ("Algorithm", "Threshold", "nModels", "MeanModelSize", "SDModelSize", "nRegions", "MeanRegionSize", "SDRegionSize", "TotalRegionLength", "MeanInterRegionSize", "SDInterRegionSize", "TotalInterRegionSize", "ProportionBasesInRegion", "MeannModelsRegion", "SDnModelsRegion", "MaxnModelsRegion", "nSingleModelRegions", "MeanMinOverlapRegion", "SDMinOverlapRegion"))
    results.to_csv("SummaryModelPerformanceByAccuracyBinaryClassifiersChr" + str(chr) + ".csv")
