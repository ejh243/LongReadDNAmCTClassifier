## extract data from beta files assumes in folder 1_raw in data_path
## filter based on min read depth
## matrix of mean dnam level across cell types saved to 2_processed
## output includes anova of cell type
## command line arguments are <cpg_reference_path> <sample_sheet_path> <data_path> <chr>
## sampleSheet format filename, group,cell type

def processBeta(filename, minRD = 10):
    content = np.fromfile(filename, dtype=np.uint8).reshape((-1, 2))
    dnam = pd.DataFrame(np.where(content[:, 1] > minRD,  content[:, 0]/content[:, 1], np.nan))
    return(dnam)

def testCTDiffs(betas, X):
    model = sm.OLS(betas.to_numpy(), X).fit()
    return(model.f_test("x1 = x2 = x3 = x4").pvalue)

import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

## process command line information
cpgFile = sys.argv[1]
sampleSheet = sys.argv[2]
resPath = sys.argv[3]
chr = int(sys.argv[4])

print("Processing chr " + str(chr))

## load CpG location info
cpgLoci = pd.read_csv(cpgFile, sep = "\t", header = None)
print("CpG reference file contains " + str(cpgLoci.shape[0]) + " sites")

## load blood samples
sampleList = pd.read_csv(sampleSheet, header = None)
print("Loading data for " + str(sampleList.shape[0]) + " samples")
fileList = [resPath + "1_raw/" + x for x in sampleList[0]]
betaMat = pd.concat([processBeta(x,10) for x in fileList], axis = 1)

print("Beta matrix loaded for " + str(betaMat.shape[0]) + " sites")

## check reference file matches beta matrix
if  betaMat.shape[0] != cpgLoci.shape[0]:
    print("Number of CpGs in beta file and CpG reference file do not match. Exiting.")
    sys. exit(1)

## filter to chromosome sites
betaMat = betaMat[cpgLoci[0].isin(["chr" + str(chr)])]
cpgLoci = cpgLoci[cpgLoci[0].isin(["chr" + str(chr)])]

## filter out rows with any NANs
countNA = betaMat.isna().sum(axis = 1)
cpgLoci = cpgLoci[countNA == 0]
betaMat = betaMat[countNA == 0]

## create dummy variabs for ANOVA
dummy = pd.get_dummies(sampleList[2]).values
## drop reference category
X = sm.add_constant(dummy[:, 1:], prepend=False)

print("Running ANOVA")

## run ANOVA
cpgLoci['P'] = betaMat.apply(lambda row : testCTDiffs(row, X), axis = 1)

print("Saving files")
## remove "chr 
cpgLoci["chr"] = [str(x).lstrip("chr") for x in cpgLoci[0]]

## write files
if not os.path.exists(resPath + "2_processed"):
    os.makedirs(resPath + "2_processed")
  
cpgLoci[["P","chr",1]].to_csv(resPath + "2_processed/rowanno_chr" + str(chr) + ".csv", index=False)
    
betaMat.to_csv(resPath + "2_processed/betas_chr" + str(chr) + ".csv", index=False)

print("All data successfully writen to file.")