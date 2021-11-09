## use KNN ML algorithm to classify cell type based on subset of CpGs
## create binarized methylation status data from mean DNAm levels in cell types to train and test model
## only runs one chr - provided on command line
## note all features are used - there is no feature selection

import sys
import os
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

chr = int(sys.argv[1])
nobs = int(sys.argv[2]) # how many "single cell" observations to simulate per cell type

os.chdir("/mnt/data1/Eilis/Projects/Asthma/ClassifyCellTypes/")

pThres = 0.5 ## threshold to keep probes with signif diffs
methStatus = (0,1) 
max_window = 10000 # max distance between outer most cpgs in classifier
min_cpg = 2 # number of cpgs in first classifer
nCV = 15 # number of iterations to simulate cross validation



probeAnno = pd.read_csv("TrainingData/rowanno_chr" + str(chr) + ".csv").values

## to speed up computation exclude sites with no evidence of cell type diffs from ANOVA
## equivalent to excluding features that don't vary
probeAnno = probeAnno[probeAnno[:,5] < pThres,:]



## sort by position
probeAnno = probeAnno[np.argsort(probeAnno[:, 7]),:]

ctProbs = probeAnno[:,0:5] ## matrix of probability of being methylated by cell type


## as sensitivity of array is poor at the extremes were meth level is estimated as >0.9 should be effectively 1, and <0.1, 0
## change these values
ctProbs = np.where(ctProbs > 0.9, 1, ctProbs)
ctProbs = np.where(ctProbs < 0.1, 0, ctProbs)

## create bin test and training data used mean DNAm as probability a CpG is methylated for a particular read
nsites, nCT = np.shape(ctProbs)
train_obs = np.empty((nsites, nCT*nobs, nCV), dtype = int)
test_obs = np.empty((nsites, nCT*nobs, nCV), dtype = int)

l = 0
while l < nCV:
    i = 0
    while i < nsites:
        j = 0
        while j < nCT:
            train_obs[i,np.arange((j*nobs),(j*nobs)+nobs),l] = np.random.choice(methStatus, nobs, p=[1-ctProbs[i,j],ctProbs[i,j]])
            test_obs[i,np.arange((j*nobs),(j*nobs)+nobs),l] = np.random.choice(methStatus, nobs, p=[1-ctProbs[i,j],ctProbs[i,j]])
            j +=1
        i +=1
    l += 1


## array of cell type labels (i.e. what we want to predict)
Y = np.repeat(np.arange(0,nCT), nobs)

nsites = np.shape(probeAnno)[0]

## create empty array to append results to
results = np.empty((0,14))

# for each site, iteratively add sites to predict until a max window size is hit
site_index = 0
ncpg = min_cpg

while site_index < nsites:
    print("Running site: " + str(site_index + 1) + " of " + str(nsites))
    start = probeAnno[site_index,7]
    ## need catch for if size of classifier is greater than number of sites left on chr
    if( site_index+ncpg <= nsites):
        stop = probeAnno[(site_index+ncpg-1),7]
        windowSize = stop - start
    else:
        windowSize = max_window
    ## check if addition of extra site increase span of predictors beyond max windowSize
    if(windowSize < max_window):
        l = 0
        ## run nCV times to get average performance
        cvValues = np.empty((nCV, nCT*2))
        while l < nCV:
            X = np.transpose(train_obs[site_index:(site_index+ncpg),:,l])
            model = KNeighborsClassifier(nCT)
            model.fit = model.fit(X,Y) 
            test_pred = model.predict(np.transpose(test_obs[site_index:(site_index+ncpg),:]))
            bool_correct = np.equal(test_pred, Y)
           ## count number correct for each cell type to caluclate sensitivity
            cvValues[l,:nCT] = np.array([sum(x) for x in np.split(bool_correct, nCT)], dtype = float)/nobs
            
            ## calculate specificity
            cvValues[l,nCT:] = np.array([sum((test_pred != x) & (Y != x))/(4*nobs) for x in np.arange(5)], dtype = float)
            
            l += 1
        # summarise performance
        results = np.append(results, [np.append([chr, start, ncpg, windowSize], cvValues.mean(0))], axis=0)
        
        ## increase number of cpgs in predictor for next run
        ncpg += 1
    else:
        ## if so move to next site
        site_index += 1
        ## reset number of cpgs for first classifer
        ncpg = min_cpg


np.savetxt("Results/KNNBinarizedChr" + str(chr) + "_" + str(nobs) + "obsCT.csv", results, delimiter=",")