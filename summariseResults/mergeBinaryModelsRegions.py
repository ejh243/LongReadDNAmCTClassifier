import sys
import utils
import os
import pandas as pd
import numpy as np
import pyranges as pr


# model names
modelOpts = ["KNN", "NBayes", "RandFor", "SVM"]

## process command line information
resultsPath = sys.argv[1]
nCT  = sys.argv[2]

os.chdir(resultsPath)

## list files in results folder
allFiles = os.listdir()
allFiles = list(filter(lambda x:'BinaryClassifier' in x, allFiles))


## load results
modelMissing = []
for modelType in modelOpts:
    subFiles = list(filter(lambda x:modelType in x, allFiles))
    print(str(len(subFiles)) + " files found for model type " + modelType)
    if len(subFiles) == 22:
        allDat = pd.concat([utils.loadResults(x,"binary", nCT) for x in subFiles]).sort_values(by=["Chr", "Position", "nCpG"])
        ## count
        print(str(len(allDat)) + " models loaded for model type " + modelType)
        ## add Density column
        allDat["Density"] = allDat['WindowSize']/allDat['nCpG']
        ## calc overall accuracy Only holds if cell types equal
        sensCols = [col for col in allDat.columns if col.endswith('sensitivity')]
        allDat["Accuracy"] = allDat[sensCols].mean(1)
        
        # check if pyranges object exists
        var_exists = 'granges' in locals() or 'var' in globals()
        if not var_exists:
        ## create set of genomic regions
            granges = pr.PyRanges(chromosomes = allDat["Chr"].astype("int"), starts = allDat["Position"], ends = allDat["Position"]+allDat["WindowSize"])
            setattr(granges, "nCpG", allDat["nCpG"])
        ## take minimum of sensitivity and specificity per model
        for i in range(0,int(nCT)):
            setattr(granges, modelType + "_CT" + str(i+1), allDat[["CT" + str(i+1) + "_sensitivity","CT" + str(i+1) + "_specificity"]].min(axis = 1))
    else:
        modelMissing = modelMissing.append(modelType)

if modelMissing is not None and len(modelMissing) > 0:
    modelOpts = [modelType for modelType in modelOpts if modelType not in modelMissing]

del allDat
 
## how many genomic regions with reasonable sensitivity & specificity
allRows = []

## identify models with sufficient accuracy
for threshold in np.arange(0.7,1.00, 0.02):
    print("Aggregating models with accuracy > " + str(threshold))
    for ml in modelOpts:
        print("Merging ML model:" + ml)
        for i in range(0,int(nCT)):
            ## if only 2 CT then only need to do once
            if int(nCT) > 2 or i != 1:
                output = []
                output.append(ml)
                output.append("CT" + str(i+1))
                output.append(threshold)
                boolIndex = getattr(granges,ml + "_CT" + str(i+1)) > threshold
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
    results = pd.DataFrame(allRows, columns = ("Algorithm", "CT", "Threshold", "nModels", "MeanModelSize", "SDModelSize", "nRegions", "MeanRegionSize", "SDRegionSize", "TotalRegionLength", "MeanInterRegionSize", "SDInterRegionSize", "TotalInterRegionSize", "ProportionBasesInRegion", "MeannModelsRegion", "SDnModelsRegion", "MaxnModelsRegion", "nSingleModelRegions", "MeanMinOverlapRegion", "SDMinOverlapRegion"))
    results.to_csv("SummaryModelPerformanceByAccuracyBinaryClassifiers.csv")
