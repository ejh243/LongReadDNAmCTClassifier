# write to file regions with models that have minimum predictive power

import os
import sys
import utils
import pandas as pd
import pyranges as pr

## process command line information
resultsPath = sys.argv[1]
chr = sys.argv[2]
nCT = sys.argv[3]
threshold = 0.95

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
bestModel = mergeDf.idxmax(1)
bestModel.value_counts()
mergeDf['best'] = maxMean
mergeDf['bestModel'] = bestModel

## create set of genomic regions
granges = pr.PyRanges(chromosomes = allDat[modelOpts[0]]["Chr"].astype("int"), starts = allDat[modelOpts[0]]["Position"], ends = allDat[modelOpts[0]]["Position"]+allDat[modelOpts[0]]["WindowSize"])
for each in modelOpts:
    setattr(granges, each, allDat[each]['MeanAccuracy'])

setattr(granges, "BestAccuracy", mergeDf['best'])
setattr(granges, "BestModel", mergeDf['bestModel'])
setattr(granges, "nCpG", allDat[modelOpts[0]]["nCpG"])

## identify models with sufficinet accuracy
col = modelOpts + ["BestAccuracy"]
print("Aggregating models with accuracy > " + str(threshold))
for ml in col:
    
    boolIndex = getattr(granges,ml) > threshold
    ## count number of models
    print("Aggregating " + str(sum(boolIndex)) + " " + ml + " models.")
    if sum(boolIndex) > 0:
        
        ## count number of genomic regions - merge into non-overlapping set
        regions = granges[boolIndex].merge()
        
        ## how many models within each region
        modelOverlaps = regions.coverage(granges[boolIndex])
        modelOverlaps.Length = modelOverlaps.End - modelOverlaps.Start
        modelOverlaps.to_csv(str("MergedPredictiveRegionsThreshold" + str(threshold) + "Model" + ml + "Chr" + str(chr) + ".csv"))
        

        
