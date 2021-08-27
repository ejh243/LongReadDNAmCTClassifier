## test out Random Forest algorithm to classify cell type based on subset of CpGs
## create binarized methylation status data from mean DNAm levels in cell types to train and test model
## only runs one chr - provided on command line

import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

chr = int(sys.argv[1])

os.chdir("/mnt/data1/Eilis/Projects/Asthma/ClassifyCellTypes/")

pThres = 1 ## threshold to keep probes with signif diffs
nobs = 100 # how many "single cell" observations to simulate per cell type
methStatus = (0,1) 
celltypes = ("Bcell", "CD4Tcells", "CD8Tcell", "Mono", "Gran")
max_window = 20000 # max distance between outer most cpgs in classifier
min_cpg = 2 # number of cpgs in first classifer


probeAnno = pd.read_csv("TrainingData/rowanno_chr" + str(chr) + ".csv").values

## to speed up computation exclude sites with no evidence of cell type diffs from ANOVA
## equivalent to excluding features that don't vary
probeAnno = probeAnno[probeAnno[:,5] < pThres,:]

## sort by position
probeAnno = probeAnno[np.argsort(probeAnno[:, 7]),:]

ctProbs = probeAnno[:,0:5] ## matrix of probability of being methylated by cell type

## create bin test and training data used mean DNAm as probability a CpG is methylated for a particular read
i = 0
nsites, nCT = np.shape(ctProbs)
train_obs = np.empty((nsites, nCT*nobs), dtype = int)
test_obs = np.empty((nsites, nCT*nobs), dtype = int)
while i < nsites:
    j = 0
    while j < nCT:
        train_obs[i,np.arange((j*nobs),(j*nobs)+nobs)] = np.random.choice(methStatus, nobs, ctProbs[i,j])
        test_obs[i,np.arange((j*nobs),(j*nobs)+nobs)] = np.random.choice(methStatus, nobs, ctProbs[i,j])
        j +=1
    i +=1


## array of cell type labels (i.e. what we want to predict)
Y = np.repeat(np.arange(0,nCT), nobs)

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
        X = np.transpose(train_obs[site_index:(site_index+ncpg),:])
		# define the model
        model = RandomForestClassifier()
        model.fit = model.fit(X,Y) 
        test_pred = model.predict(np.transpose(test_obs[site_index:(site_index+ncpg),:]))
        bool_correct = np.equal(test_pred, Y)
		
		## count number correct for each cell type
        n_correct = np.array([sum(x) for x in np.split(bool_correct, nCT)], dtype = float)
		
		## calculate specificity
        specificity = np.array([sum((test_pred != x) & (Y != x))/(4*nobs) for x in np.arange(5)], dtype = float)
		
		# summarise performance
        results = np.append(results, [np.append(np.append([chr, start, ncpg, windowSize], n_correct/nobs), specificity)], axis=0)
        
        ## increase number of cpgs in predictor for next run
        ncpg += 1
    else:
        ## if so move to next site
        site_index += 1
        ## reset number of cpgs for first classifer
        ncpg = min_cpg


np.savetxt("Results/RandomForestBinarizedChr" + str(chr) + "_" + str(nobs) + "obsCT.csv", results, delimiter=",")