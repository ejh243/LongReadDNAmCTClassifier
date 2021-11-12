## test out Naive Bayes classifier to predict cell type based on subset of CpGs
## uses purified cell type DNAm levels in cell types as training data (i.e. continuous input)
## uses stratified k fold cross validation to assess model accuracy
## pilot using filtered chr 22 to infer parameter options

import sys
import os
import pandas as pd
import numpy as np
from params import * ## where params.py contains parameters
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold

os.chdir(workDir)

chr = int(sys.argv[1]) ## provided from command line

betas = pd.read_csv(trainDataPath + "betas_chr" + str(chr) + ".csv").values
pheno = pd.read_csv(trainDataPath + "colanno.csv").values
probeAnno = pd.read_csv(trainDataPath + "rowanno_chr" + str(chr) + ".csv").values

## to speed up computation exclude sites with no evidence of cell type diffs from ANOVA
## equivalent to excluding features that don't vary
betas = betas[probeAnno[:,nCT] < pThres,:]
probeAnno = probeAnno[probeAnno[:,nCT] < pThres,:]

## sort by position
betas = betas[np.argsort(probeAnno[:,nCT+2]),:]
probeAnno = probeAnno[np.argsort(probeAnno[:,nCT+2]),:]

## array of cell type labels (i.e. what we want to predict)
Y = pheno[:,2]

nsites = np.shape(probeAnno)[0]

## create empty array to append results to
results = np.empty((0,6)) ## always six regardless of number of cell types

# define the model
model = GaussianNB()

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


np.savetxt(resultsPath + "NaiveBayesGaussianFilteredChr" + str(chr) + ".csv", results, delimiter=",")