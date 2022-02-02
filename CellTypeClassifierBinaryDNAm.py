## script to implement a range of machine learning algorithms to predict cell type 
## from contiguous sets of DNAm sites
## uses mean purified cell type DNAm levels to simulated binary methylation training data
## the following arguments are required on the command line in this order:
## 1. chromosome to analyse, 
## 2. choice of machine learning model, 
## 3. folder where DNAm and associated meta data  is located  
## 4. output folder are provided on the command line when script is executed
## 5. which column in the phenotype type contains the cell type classifications
## 6. the number of training and test observations per cell type to generate
## there is also an accompanying params.py file which provides parameters to fine tune application

import sys
import os
import pandas as pd
import numpy as np
from params import * ## where params.py contains parameters
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.utils.multiclass import type_of_target

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
   
np.random.seed(randomSeed) ## ensures that the sample random sampling occurs across algorithms and across reruns of the script

## process command line information
chr = int(sys.argv[1])
modelType = sys.argv[2]
trainDataPath = sys.argv[3]
resultsPath = sys.argv[4]
ctCol = int(sys.argv[5])
nobs = int(sys.argv[6]) # how many "single cell" observations to simulate per cell type

## load training data
betas = pd.read_csv(trainDataPath + "betas_chr" + str(chr) + ".csv").values
pheno = pd.read_csv(trainDataPath + "colanno.csv").values
probeAnno = pd.read_csv(trainDataPath + "rowanno_chr" + str(chr) + ".csv").values


## array of cell type labels (i.e. what we want to predict)
inputY = pheno[:,ctCol]

## calculate the number of CT
nCT = np.unique(pheno[:, ctCol]).shape[0]

## check format of Y
if ((nCT == 2) & (type_of_target(inputY) != 'binary')):   
    inputY = inputY.astype(int)

print("Found " + str(nCT) + " cell types to predict")

print("Outcome is a " + type_of_target(inputY) + " variable")

## to speed up computation exclude sites with no evidence of cell type diffs from ANOVA
## equivalent to excluding features that don't vary
pvalCol = probeAnno.shape[1]-3
betas = betas[probeAnno[:,pvalCol] < pThres,:]
probeAnno = probeAnno[probeAnno[:,pvalCol] < pThres,:]

## sort by position
posCol = probeAnno.shape[1]-1
betas = betas[np.argsort(probeAnno[:,posCol]),:]
probeAnno = probeAnno[np.argsort(probeAnno[:,posCol]),:]

nsites = np.shape(probeAnno)[0]

## caluclate mean DNAm level for each cell type
ctProbs = pd.DataFrame(betas).groupby(inputY, axis = 'columns').mean() ## matrix of probability of being methylated by cell type


## as sensitivity of array is poor at the extremes were meth level is estimated as >0.9 should be effectively 1, and <0.1, 0
## change these values
ctProbs = np.where(ctProbs > 0.9, 1, ctProbs)
ctProbs = np.where(ctProbs < 0.1, 0, ctProbs)

## create bin test and training data used mean DNAm as probability a CpG is methylated for a particular read
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


## array of cell type labels for simulated data 
Y = np.repeat(np.arange(0,nCT), nobs)

nsites = np.shape(probeAnno)[0]

## create empty array to append results to
results = np.empty((0,4+(2*nCT))) 

# for each site, iteratively add sites to predict until a max window size is hit
site_index = 0
ncpg = min_cpg

while site_index < nsites:
    print("Running site: " + str(site_index + 1) + " of " + str(nsites))
    start = probeAnno[site_index,posCol]
    ## need catch for if size of classifier is greater than number of sites left on chr
    if( site_index+ncpg <= nsites):
        stop = probeAnno[(site_index+ncpg-1),posCol]
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
            model = initiateModel(modelType, nCT)
            model.fit = model.fit(X,Y) 
            test_pred = model.predict(np.transpose(test_obs[site_index:(site_index+ncpg),:,l]))
            bool_correct = np.equal(test_pred, Y)
           ## count number correct for each cell type to caluclate sensitivity
            cvValues[l,:nCT] = np.array([sum(x) for x in np.split(bool_correct, nCT)], dtype = float)/nobs
            
            ## calculate specificity
            cvValues[l,nCT:] = np.array([sum((test_pred != x) & (Y != x))/(nCT*nobs) for x in np.arange(nCT)], dtype = float)
            
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


np.savetxt(resultsPath + "BinaryClassifier" + modelType + "_" + str(nCT) + "CT_Chr" + str(chr) + "_" + str(nobs) + "Obs.csv", results, delimiter=",")
