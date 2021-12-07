## script to implement a range of machine learning algorithms to predict cell type 
## from contiguous sets of DNAm sites
## uses mean purified cell type DNAm levels in cell types as training data (i.e. continuous input)
## the following arguments are required on the command line in this order
## chromosome to analyse, choice of machine learning model, folder where DNAm and associated meta data  is located and output folder are provided on the command line when script is executed
## there is also an accompanying params.py file which provides parameters to fine tune application

import sys
import os
import pandas as pd
import numpy as np
from params import * ## where params.py contains parameters
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold

def initiateModel(modelType, nCT = 1):
    if (modelType == "KNN"):
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(nCT)
    elif (modelType == "NBayes" ):
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()
    elif (modelType == "RandFor"):
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()
    elif (modelType == "SVM"):
        from sklearn import svm
        model = svm.SVC()
    else:
        sys.exit('Model type not recognised')
    return(model)
   

## process command line information
chr = int(sys.argv[1])
modelType = sys.argv[2]
trainDataPath = sys.argv[3]
resultsPath = sys.argv[4]

## load training & test data
betas = pd.read_csv(trainDataPath + "betas_chr" + str(chr) + ".csv").values
pheno = pd.read_csv(trainDataPath + "colanno.csv").values
probeAnno = pd.read_csv(trainDataPath + "rowanno_chr" + str(chr) + ".csv").values

## array of cell type labels (i.e. what we want to predict)
Y = pheno[:,1]

## calculate the number of CT
nCT = np.unique(pheno[:, 1]).shape[0]

## to speed up computation exclude sites with no evidence of cell type diffs from ANOVA
## equivalent to excluding features that don't vary
betas = betas[probeAnno[:,nCT] < pThres,:]
probeAnno = probeAnno[probeAnno[:,nCT] < pThres,:]

## sort by position
betas = betas[np.argsort(probeAnno[:,nCT+2]),:]
probeAnno = probeAnno[np.argsort(probeAnno[:,nCT+2]),:]

nsites = np.shape(probeAnno)[0]

## create empty array to append results to
results = np.empty((0,6)) ## always six regardless of number of cell types

# define the model
model = initiateModel(modelType, nCT)

# evaluate the model using cross validation (k fold stratified) 
cv = RepeatedStratifiedKFold(n_splits=nCV_splits, n_repeats=nCV_repeats, random_state=randomSeed)

# for each site, iteratively add sites to predict until a max window size is hit
site_index = 0
ncpg = min_cpg
while site_index < nsites:
    print("Running site: " + str(site_index + 1) + " of " + str(nsites))
    start = probeAnno[site_index,nCT+2]
    ## need catch for if size of classifier is greater than number of sites left on chr
    if( site_index+ncpg <= nsites):
        stop = probeAnno[(site_index+ncpg-1),nCT+2]
        windowSize = stop - start
    else:
        windowSize = max_window
    ## check if addition of extra site increase span of predictors beyond max windowSize
    if(windowSize < max_window):
        X = np.transpose(betas[site_index:(site_index+ncpg),:])
        n_scores = cross_val_score(model, X, Y, scoring='accuracy', cv=cv, n_jobs=pThread)
            
        # summarise performance
        results = np.append(results, [[chr, start, ncpg, windowSize, np.mean(n_scores), np.std(n_scores)]], axis=0)
        
        ## increase number of cpgs in predictor for next run
        ncpg += 1
    else:
        ## if so move to next site
        site_index += 1
        ## reset number of cpgs for first classifer
        ncpg = min_cpg


np.savetxt(resultsPath + "ContinuousClassifier" + modelType + "_" + str(nCT) + "CT_Chr" + str(chr) + ".csv", results, delimiter=",")
