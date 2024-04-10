## extract data from beta files
## filter based on min read depth
## matrix of mean dnam level across cell types
## anova of cell type

def processBeta(filename, minRD = 10):
    content = np.fromfile(filename, dtype=np.uint8).reshape((-1, 2))
    dnam = pd.DataFrame(np.where(content[:, 1] > minRD,  content[:, 0]/content[:, 1], np.nan))
    return(dnam)

def testCTDiffs(betas, X):
    model = sm.OLS(betas.to_numpy(), X).fit()
    return(model.f_test("x1 = x2 = x3 = x4").pvalue)


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

## load blood samples
sampleList = pd.read_csv(sampleSheet, sep = "\t", header = None)
betaMat = pd.concat([processBeta(x,10) for x in sampleList[0]], axis = 1)

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
  
cpgLoci[["P","chr",1]].to_csv(resPath + "rowanno_chr" + str(chr) + ".csv", index=False)
    
betaMat.to_csv(resPath + "betas_chr" + str(chr) + ".csv", index=False)
