# First argument is the path to folder with read level results file (output of mergeONTTestingResults.py) organisined in subfolders by prediction model and then ML algorithm
# Second argument is the path to the read metadata file (e.g. read length, number of CpGs, etc.)

# load libraries
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

resultsPath  = sys.argv[1]
readDataFile = sys.argv[2] # file with read metadata (e.g. read length, number of CpGs, etc.)


 # load read metadata 
readData = pd.read_csv(readDataFile, header = 0, names = ("read_id","chrom","alignment_start","read_length"), sep = "\t")
# bin read length into 500 bp bins
readData["read_length_bin"] = pd.cut(readData.read_length, bins = np.arange(0, readData.read_length.max() + 500, 500), right = False)


for mlModel in ["SVM", "KNN", "NBayes"]:
    for cellPredict in ["Lymphocytes", "Granulocytes", "Monocytes", "Bcells", "Tcells"]:
        outPath = resultsPath + "/" + cellPredict + "/Plots/" + mlModel + "/" # create output directory if it doesn't exist   
        if not os.path.exists(outPath):
            os.makedirs(outPath)
        # load read level results
        results = pd.read_csv(resultsPath + "/" + cellPredict + "/" + mlModel + "/PredictionOutput/mergedModelPredictions.csv", header = 0, names = ("sample_type","sampleID","region","nCpG","read_id","predicted_cell_type","probability_cell_type"))
        # need to classify if predictions are correct or not; depends on which models we are testing
        # testing with Lymphocytes
        if cellPredict == "Lymphocytes":
            results["correct_prediction"] = results.apply(lambda x: "Correct" if x.sample_type in ["B", "CD4", "CD8"] and x.predicted_cell_type == 1 else ("Correct" if x.sample_type in ["Granulocyte", "Monocyte"] and x.predicted_cell_type == 0 else "Incorrect"), axis = 1)
        elif cellPredict == "Granulocytes":
            results["correct_prediction"] = results.apply(lambda x: "Correct" if x.sample_type == "Granulocyte" and x.predicted_cell_type == 1 else ("Correct" if x.sample_type in ["B", "CD4", "CD8", "Monocyte"] and x.predicted_cell_type == 0 else "Incorrect"), axis = 1)
        elif cellPredict == "Monocytes":
            results["correct_prediction"] = results.apply(lambda x: "Correct" if x.sample_type == "Monocyte" and x.predicted_cell_type == 1 else ("Correct" if x.sample_type in ["B", "CD4", "CD8", "Granulocyte"] and x.predicted_cell_type == 0 else "Incorrect"), axis = 1)
        elif cellPredict == "Bcells":
            results["correct_prediction"] = results.apply(lambda x: "Correct" if x.sample_type == "B" and x.predicted_cell_type == 1 else ("Correct" if x.sample_type in ["Monocyte", "CD4", "CD8", "Granulocyte"] and x.predicted_cell_type == 0 else "Incorrect"), axis = 1)
        elif cellPredict == "Tcells":
            results["correct_prediction"] = results.apply(lambda x: "Correct" if x.sample_type in ["CD4", "CD8"] and x.predicted_cell_type == 1 else ("Correct" if x.sample_type in ["B", "Monocyte", "Granulocyte"] and x.predicted_cell_type == 0 else "Incorrect"), axis = 1)
        else:
            print("Error: cellPredict argument must be one of 'Lymphocytes', 'Granulocytes', 'Monocytes', 'Bcells', or 'Tcells'")
            sys.exit(1)
        # calculate overall accuracy
        overallAccuracy = (results.correct_prediction == "Correct").mean()
        # match up metadata and read level results
        results = results.merge(readData, on = "read_id", how = "left")
        # summarise number of reads tested for each cell type, number of CpGs in the read, etc.
        readSummary = results.groupby("sample_type").agg({"read_id": "count", "nCpG": "mean"}).reset_index()
        readSummary.columns = ["sample_type", "number_of_reads", "average_cpgs"]
        # plot results
        # plot percentage accurate for each cell type
        accuracyByCellType = results.groupby(["sample_type", "correct_prediction"]).size().unstack(fill_value = 0)
        accuracyByCellType["Accuracy"] = accuracyByCellType.Correct / (accuracyByCellType.Correct + accuracyByCellType.Incorrect)
        accuracyByCellType = accuracyByCellType.reset_index()
        # save summary table
        accuracyByCellType.to_csv(outPath + "/ReadLevelPredictionAccuracyByCellType.csv", index = False)
        plt.figure(figsize = (6,4))
        plt.bar(accuracyByCellType.sample_type, accuracyByCellType.Accuracy, color = colors[:len(accuracyByCellType.sample_type.unique())])
        plt.ylim(0,1)
        plt.ylabel("Accuracy")
        plt.xlabel("True Cell Type")
        plt.savefig(outPath + "/ReadLevelPredictionAccuracyByCellType.png", bbox_inches = "tight")
        # plot accuracy as a function of number of cpgs in the read
        accuracyByCpGs = results.groupby(["nCpG", "correct_prediction"]).size().unstack(fill_value = 0)
        accuracyByCpGs["Accuracy"] = accuracyByCpGs.Correct / (accuracyByCpGs.Correct + accuracyByCpGs.Incorrect)
        accuracyByCpGs = accuracyByCpGs.reset_index()   
        # only plot for CpG counts with at least 100 reads tested
        accuracyByCpGs = accuracyByCpGs[accuracyByCpGs.Correct + accuracyByCpGs.Incorrect >= 100]
        plt.figure(figsize = (6,4))
        plt.plot(accuracyByCpGs.nCpG, accuracyByCpGs.Accuracy, color = colors[1])
        plt.ylabel("Accuracy")
        plt.xlabel("Number of CpGs")
        plt.savefig(outPath + "/ReadLevelPredictionAccuracyByCpGs.png", bbox_inches = "tight")
        # plot accuracy as a function of cell type and number of cpgs in the read
        accuracyByCellTypeCpGs = results.groupby(["sample_type", "nCpG", "correct_prediction"]).size().unstack(fill_value = 0)
        accuracyByCellTypeCpGs["Accuracy"] = accuracyByCellTypeCpGs.Correct / (accuracyByCellTypeCpGs.Correct + accuracyByCellTypeCpGs.Incorrect)
        accuracyByCellTypeCpGs = accuracyByCellTypeCpGs.reset_index()
        # only plot for CpG counts with at least 100 reads tested
        accuracyByCellTypeCpGs = accuracyByCellTypeCpGs[accuracyByCellTypeCpGs.Correct + accuracyByCellTypeCpGs.Incorrect >= 100]
        plt.figure(figsize = (6,4))
        for i, cellType in enumerate(accuracyByCellTypeCpGs.sample_type.unique()):
            subset = accuracyByCellTypeCpGs[accuracyByCellTypeCpGs.sample_type == cellType]
            plt.plot(subset.nCpG, subset.Accuracy, label = cellType, color = colors[i])
        plt.legend()
        plt.ylabel("Accuracy")
        plt.xlabel("Number of CpGs")
        plt.savefig(outPath + "/ReadLevelPredictionAccuracyByCellTypeAndCpGs.png", bbox_inches = "tight")
        # plot by read length
        accuracyByReadLength = results.groupby(["read_length_bin", "correct_prediction"]).size().unstack(fill_value = 0)
        # add bin midpoints for plotting
        accuracyByReadLength["bin_midpoint"] = accuracyByReadLength.index.map(lambda x: (x.left + x.right) / 2)
        accuracyByReadLength["Accuracy"] = accuracyByReadLength.Correct / (accuracyByReadLength.Correct + accuracyByReadLength.Incorrect)
        accuracyByReadLength = accuracyByReadLength.reset_index()
        # only plot for read length bins with at least 100 reads tested
        accuracyByReadLength = accuracyByReadLength[accuracyByReadLength.Correct + accuracyByReadLength.Incorrect >= 100]
        plt.figure(figsize = (6,4))
        plt.plot(accuracyByReadLength.bin_midpoint, accuracyByReadLength.Accuracy, color = colors[2])
        plt.xticks(rotation = 45)
        plt.ylabel("Accuracy")
        plt.xlabel("Read Length Bin")
        plt.savefig(outPath + "/ReadLevelPredictionAccuracyByReadLength.png", bbox_inches = "tight")
        # plot by read length and cell type
        accuracyByCellTypeReadLength = results.groupby(["sample_type", "read_length_bin", "correct_prediction"]).size().unstack(fill_value = 0)
        accuracyByCellTypeReadLength["Accuracy"] = accuracyByCellTypeReadLength.Correct / (accuracyByCellTypeReadLength.Correct + accuracyByCellTypeReadLength.Incorrect)           
        # add bin midpoints for plotting
        accuracyByCellTypeReadLength["bin_midpoint"] = accuracyByCellTypeReadLength.index.get_level_values("read_length_bin").map(lambda x: (x.left + x.right) / 2)
        accuracyByCellTypeReadLength = accuracyByCellTypeReadLength.reset_index()
        # only plot for read length bins with at least 100 reads tested
        accuracyByCellTypeReadLength = accuracyByCellTypeReadLength[accuracyByCellTypeReadLength.Correct + accuracyByCellTypeReadLength.Incorrect >= 100]
        plt.figure(figsize = (6,4))
        for i, cellType in enumerate(accuracyByCellTypeReadLength.sample_type.unique()):    
            subset = accuracyByCellTypeReadLength[accuracyByCellTypeReadLength.sample_type == cellType]
            plt.plot(subset.bin_midpoint, subset.Accuracy, label = cellType, color = colors[i])
        plt.legend()
        plt.xticks(rotation = 45)
        plt.ylabel("Accuracy")
        plt.xlabel("Read Length")
        plt.savefig(outPath + "/ReadLevelPredictionAccuracyByCellTypeAndReadLength.png", bbox_inches = "tight")
        # plot by density of methylation sites in the read
        # calculate density of methylation sites in the read
        results["cpg_density"] = results.read_length / results.nCpG
        # bin density into bins of 100bp
        results["cpg_density_bin"] = pd.cut(results.cpg_density, bins = np.arange(0, results.cpg_density.max() + 100, 100), right = False)
        accuracyByCpGDensity = results.groupby(["cpg_density_bin", "correct_prediction"]).size().unstack(fill_value = 0)
        # add bin midpoints for plotting
        accuracyByCpGDensity["bin_midpoint"] = accuracyByCpGDensity.index.get_level_values("cpg_density_bin").map(lambda x: (x.left + x.right) / 2)
        accuracyByCpGDensity["Accuracy"] = accuracyByCpGDensity.Correct / (accuracyByCpGDensity.Correct + accuracyByCpGDensity.Incorrect)
        accuracyByCpGDensity = accuracyByCpGDensity.reset_index()
        # only plot for density bins with at least 100 reads tested
        accuracyByCpGDensity = accuracyByCpGDensity[accuracyByCpGDensity.Correct + accuracyByCpGDensity.Incorrect >= 100]
        plt.figure(figsize = (6,4))
        plt.plot(accuracyByCpGDensity.bin_midpoint, accuracyByCpGDensity.Accuracy, color = colors[1])
        plt.ylabel("Accuracy")
        plt.xlabel("CpG Density")
        plt.savefig(outPath + "/ReadLevelPredictionAccuracyByCpGDensity.png", bbox_inches = "tight")
        # plot by density of methylation sites in the read and cell type
        accuracyByCellTypeCpGDensity = results.groupby(["sample_type", "cpg_density_bin", "correct_prediction"]).size().unstack(fill_value = 0)
        accuracyByCellTypeCpGDensity["Accuracy"] = accuracyByCellTypeCpGDensity.Correct / (accuracyByCellTypeCpGDensity.Correct + accuracyByCellTypeCpGDensity.Incorrect)
        # add bin midpoints for plotting
        accuracyByCellTypeCpGDensity["bin_midpoint"] = accuracyByCellTypeCpGDensity.index.get_level_values("cpg_density_bin").map(lambda x: (x.left + x.right) / 2)
        accuracyByCellTypeCpGDensity = accuracyByCellTypeCpGDensity.reset_index()
        # only plot for density bins with at least 100 reads tested
        accuracyByCellTypeCpGDensity = accuracyByCellTypeCpGDensity[accuracyByCellTypeCpGDensity.Correct + accuracyByCellTypeCpGDensity.Incorrect >= 100]
        plt.figure(figsize = (6,4))
        for i, cellType in enumerate(accuracyByCellTypeCpGDensity.sample_type.unique()):    
            subset = accuracyByCellTypeCpGDensity[accuracyByCellTypeCpGDensity.sample_type == cellType]
            plt.plot(subset.bin_midpoint, subset.Accuracy, label = cellType, color = colors[i])
        plt.legend()
        plt.ylabel("Accuracy")
        plt.xlabel("CpG Density")
        plt.savefig(outPath + "/ReadLevelPredictionAccuracyByCellTypeAndCpGDensity.png", bbox_inches = "tight") 