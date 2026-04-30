# First argument is the path to the read level results file (output of mergeONTTestingResults.py)


# load libraries
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pandas.api.types import CategoricalDtype

plt.rcParams.update({'font.size': 12})

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

resultsFile  = sys.argv[1]
cellPredict = sys.argv[2] # which cell type are we testing predictions for? (e.g. "Lymphocyte", "Granulocyte", "Monocyte")

outPath = os.path.dirname(os.path.dirname(resultsFile))
if not os.path.exists(outPath + "/Plots"):
    os.makedirs(outPath + "/Plots")

# load read level results
results = pd.read_csv(resultsFile, header = 0, names = ("sample_type","sampleID","region","nCpG","read_id","predicted_cell_type","probability_cell_type"))
# need to classify if predictions are correct or not; depends on which models we are testing
# testing with Lymphocytes
if cellPredict == "Lymphocyte":
    results["correct_prediction"] = results.apply(lambda x: "Correct" if x.sample_type in ["B"] and x.predicted_cell_type == 1 else ("Correct" if x.sample_type in ["Granulocyte", "Monocyte"] and x.predicted_cell_type == 0 else "Incorrect"), axis = 1)


# load read metadata

# match up metadata and read level results

# summarise number of reads tested for each cell type, number of CpGs in the read, etc.
readSummary = results.groupby("sample_type").agg({"read_id": "count", "nCpG": "mean"}).reset_index()
readSummary.columns = ["sample_type", "number_of_reads", "average_cpgs"]
        

# plot results

# plot percentage accurate for each cell type
accuracyByCellType = results.groupby(["sample_type", "correct_prediction"]).size().unstack(fill_value = 0)
accuracyByCellType["Accuracy"] = accuracyByCellType.Correct / (accuracyByCellType.Correct + accuracyByCellType.Incorrect)
accuracyByCellType = accuracyByCellType.reset_index()

plt.figure(figsize = (6,4))
plt.bar(accuracyByCellType.sample_type, accuracyByCellType.Accuracy, color = colors[0])
plt.ylim(0,1)
plt.ylabel("Accuracy")
plt.xlabel("True Cell Type")
plt.title("Read Level Prediction Accuracy by Cell Type")
plt.savefig(outPath + "/Plots/ReadLevelPredictionAccuracyByCellType.png", bbox_inches = "tight")

# plot accuracy as a function of number of cpgs in the read
accuracyByCpGs = results.groupby(["nCpG", "correct_prediction"]).size().unstack(fill_value = 0)
accuracyByCpGs["Accuracy"] = accuracyByCpGs.Correct / (accuracyByCpGs.Correct + accuracyByCpGs.Incorrect)
accuracyByCpGs = accuracyByCpGs.reset_index()   

plt.figure(figsize = (6,4))
plt.plot(accuracyByCpGs.nCpG, accuracyByCpGs.Accuracy, color = colors[1])
plt.ylim(0,1)
plt.ylabel("Accuracy")
plt.xlabel("Number of CpGs in Read")
plt.title("Read Level Prediction Accuracy by Number of CpGs in Read")
plt.savefig(outPath + "/Plots/ReadLevelPredictionAccuracyByCpGs.png", bbox_inches = "tight")

# plot accuracy as a function of cell type and number of cpgs in the read
accuracyByCellTypeCpGs = results.groupby(["sample_type", "nCpG", "correct_prediction"]).size().unstack(fill_value = 0)
accuracyByCellTypeCpGs["Accuracy"] = accuracyByCellTypeCpGs.Correct / (accuracyByCellTypeCpGs.Correct + accuracyByCellTypeCpGs.Incorrect)
accuracyByCellTypeCpGs = accuracyByCellTypeCpGs.reset_index()

plt.figure(figsize = (8,6))
for i, cellType in enumerate(accuracyByCellTypeCpGs.sample_type.unique()):
    subset = accuracyByCellTypeCpGs[accuracyByCellTypeCpGs.sample_type == cellType]
    plt.plot(subset.nCpG, subset.Accuracy, label = cellType, color = colors[i])

plt.legend()
plt.ylabel("Accuracy")
plt.xlabel("Number of CpGs in Read")
plt.title("Read Level Prediction Accuracy by Cell Type and Number of CpGs in Read")
plt.savefig(outPath + "/Plots/ReadLevelPredictionAccuracyByCellTypeAndCpGs.png", bbox_inches = "tight")

# plot by read length

# plot by density of methylation sites in the read