## test out KNN algorithm to classify cell type based on subset of CpGs
## apply a hierarchical approach to subdivide cell types
## First separate Lymphocytes (B- & T-cells) from others (Mono & Gran)
## Second within "others" separate Mono and Gran
## Third within Lymphocytes separate B-cells from T-cells
## Fourth within T-cells separate CD4 & CD8
## create binarized methylation status data from mean DNAm levels in cell types to train and test model
## only runs one chr - provided on command line

import sys
import os
import pandas as pd
import numpy as np
from params import * # where params.py contains parameters
from sklearn import svm
from sklearn.metrics import confusion_matrix

chr = int(sys.argv[1]) 
nobs = int(sys.argv[2]) # how many "single cell" observations to simulate per cell type

os.chdir(workDir)

probeAnno = pd.read_csv(trainDataPath + "rowanno_chr" + str(chr) + ".csv").values

np.random.seed(randomSeed) ## ensures that the sample random sampling occurs across algorithms and across reruns of the script

## to speed up computation exclude sites with no evidence of cell type diffs from ANOVA
## equivalent to excluding features that don't vary
probeAnno = probeAnno[probeAnno[:,nCT] < pThres,:]

## sort by position
probeAnno = probeAnno[np.argsort(probeAnno[:, nCT+2]),:]
ctProbs = probeAnno[:,0:nCT] ## matrix of probability of being methylated by cell type


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

## create array of labels for 4 "separation models"
Y1 = np.where(Y < 3, 0, 1) ## separate Lymphocytes (B- & T-cells) from others (Mono & Gran)
Y2 = np.where(Y < 3, np.nan, Y) ## within Non-Lymph separate Mono and Gran
Y2 = Y2[np.invert(np.isnan(Y2))]
Y2 = Y2 - 3 ## convert to (0,1)
Y3 = np.repeat((0,1,1), nobs) ## within Lymphocytes separate B- & T-cells
Y4 = np.repeat(np.arange(0,2), nobs) ## within T-cells separate CD4 & CD8

## create empty array to append results to
results = np.empty((0,22))

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
        l = 0
        ## run nCV times to get average performance
        cvValues = np.empty((nCV, 18))
        while l < nCV:
            X = np.transpose(train_obs[site_index:(site_index+ncpg),:,l])
            # define the models
            model1 = svm.SVC()
            model2 = svm.SVC()
            model3 = svm.SVC()
            model4 = svm.SVC()
            ## create empty array to record final predicted cell type for each test sample after following full hierarchical classification
            test_pred_final = np.array(["NA"]*nCT*nobs)
            ## Model 1 to separate lymphocytes (0) from non-lymphocytes (1)
            model1.fit = model1.fit(X,Y1) 
            test_pred_m1 = model1.predict(np.transpose(test_obs[site_index:(site_index+ncpg),:,l]))
            sens_m1 = confusion_matrix(test_pred_m1, Y1)[0,0]/sum(Y1 == 0)
            spec_m1 = confusion_matrix(test_pred_m1, Y1)[1,1]/sum(Y1 == 1)
            
            ## Model 2 to separate monocytes (0)  from granulocytes (1)
            X2 = X[Y > 2, :]
            model2.fit = model2.fit(X2,Y2)
            # test in actual monocytes and granulocytes
            test_pred_m2 = model2.predict(np.transpose(test_obs[site_index:(site_index+ncpg),(Y > 2),l]))
            sens_m2 = confusion_matrix(test_pred_m2, Y2)[0,0]/sum(Y2 == 0)
            spec_m2 = confusion_matrix(test_pred_m2, Y2)[1,1]/sum(Y2 == 1)
            
            # test in test samples assigned as non-lymphocytes in model 1 (i.e. many contain some incorrectly labbeled lymphocytes)]
            # only worth doing if some test samples are predicted to be non-lymphocytes
            if sum(test_pred_m1 == 1):       
                test_pred_final[(test_pred_m1 == 1)] = np.where(model2.predict(np.transpose(test_obs[site_index:(site_index+ncpg),test_pred_m1 == 1,l])) == 1, 4,3)
             
            ## Model 3 to separate B-cells (0) from T-cells (1)
            X3 = X[Y < 3, :]
            model3.fit = model3.fit(X3,Y3)
            test_pred_m3 = model3.predict(np.transpose(test_obs[site_index:(site_index+ncpg),Y < 3,l]))
            sens_m3 = confusion_matrix(test_pred_m3, Y3)[0,0]/sum(Y3 == 0)
            spec_m3 = confusion_matrix(test_pred_m3, Y3)[1,1]/sum(Y3 == 1)
            # test in test samples assigned as lymphocytes in model 1 
            # only worth doing if some test samples are predicted to be lymphocytes
            if sum(test_pred_m1 == 0):       
                test_pred_final[test_pred_m1 == 0] = np.where(model3.predict(np.transpose(test_obs[site_index:(site_index+ncpg),test_pred_m1 == 0,l])) == 0, 0,-9) ## use -9 as place holder for T-cells either type
            
            ## Model 4 to separate CD4 T-cells (0) from CD8 T-cells (1)
            X4 = X[((Y == 1) | (Y == 2)), :]
            model4.fit = model4.fit(X4,Y4)
            test_pred_m4 = model4.predict(np.transpose(test_obs[site_index:(site_index+ncpg),((Y == 1) | (Y == 2)),l]))
            sens_m4 = confusion_matrix(test_pred_m4, Y4)[0,0]/sum(Y4 == 0)
            spec_m4 = confusion_matrix(test_pred_m4, Y4)[1,1]/sum(Y4 == 1)
            
            # test in test samples assigned as T-cells in model 3 
            # only worth doing if some test samples are predicted to be t-cells
            if sum(test_pred_final == "-9") > 0:       
                test_pred_final[test_pred_final == "-9"] = np.where(model4.predict(np.transpose(test_obs[site_index:(site_index+ncpg),test_pred_final == "-9",l])) == 0, 1, 2)
            
            ## calculate accuracy for cell type predictions through full pipeline
            test_pred_final = test_pred_final.astype("int")
            conTab = confusion_matrix(test_pred_final, Y)
             
            ## sensitivity through hierarchical classification
            sens_overall = np.diag(confusion_matrix(test_pred_final, Y))/nobs
            
            ## specificity through hierarchical classification
            spec_overall = [np.sum(np.delete(np.delete(conTab, i, axis = 0), i,axis = 1))/(4*nobs) for i in range(nCT)]
            
            stats = [sens_m1, spec_m1, sens_m2, spec_m2,sens_m3, spec_m3,sens_m4, spec_m4]
            
            for s1,s2 in zip(sens_overall, spec_overall):
                stats = np.append(stats, s1)
                stats = np.append(stats, s2)
            
            cvValues[l,] = stats
            
            l +=1
            
        # summarise performance
        results = np.append(results, [np.append([chr, start, ncpg, windowSize], cvValues.mean(0))], axis=0)        
        ## increase number of cpgs in predictor for next run
        ncpg += 1
    else:
        ## if so move to next site
        site_index += 1
        ## reset number of cpgs for first classifer
        ncpg = min_cpg


np.savetxt(resultsPath + "SVMHierarchicalBernoulliBinarizedChr" + str(chr) + "_" + str(nobs) + "obsCT.csv", results, delimiter=",")