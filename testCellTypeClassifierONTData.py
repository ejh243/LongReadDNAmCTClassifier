
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
array = sys.argv[7] # if array data

## load training data
betas = pd.read_csv(trainDataPath + "betas_chr" + str(chr) + ".csv").values
pheno = pd.read_csv(trainDataPath + "colanno.csv").values
probeAnno = pd.read_csv(trainDataPath + "rowanno_chr" + str(chr) + ".csv").values

print("Phenotype data loaded for " + str(np.shape(pheno)[0]) + " samples")
print("Beta values loaded for " + str(np.shape(betas)[1]) + " samples")


## array of cell type labels (i.e. what we want to predict)
inputY = pheno[:,ctCol]

## filter to samples with labels
betas = betas[:,(~pd.isnull(inputY))]
inputY = inputY[(~pd.isnull(inputY))]

print("Cell type labels found for " + str(np.shape(betas)[1]) + " samples")

## calculate the number of CT
nCT = np.unique(inputY).shape[0]

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

print("Filtered to " + str(nsites) + " sites")



## as sensitivity of array is poor at the extremes were meth level is estimated as >0.9 should be effectively 1, and <0.1, 0
## change these values
if array:
    ctProbs = np.where(ctProbs > 0.9, 1, ctProbs)
    ctProbs = np.where(ctProbs < 0.1, 0, ctProbs)

## load test data
testData = pd.read_csv("/lustre/projects/Research_Project-MRC190311/DNAm/loyfer/3_analysis/testONT/Granulocyte_6_read_level.tsv",sep='\t').values
print("Methylation status loaded for " + str(np.shape(testData)[1]) + " CpGs")


testData = pd.DataFrame(testData, columns = ["read_id", "forward_read_position","ref_position","chrom","mod_strand","ref_strand","ref_mod_strand","fw_soft_clipped_start","fw_soft_clipped_end","alignment_start","alignment_end","read_length","mod_qual","mod_code","base_qual","ref_kmer","query_kmer","canonical_base","modified_primary_base","inferred","flag"])
# filter 5mC only
testData = testData[testData["mod_code"] == "m"]


## filter overlap with betas matrix above
mask = testData["ref_position"].isin(probeAnno[:, posCol])
testData = testData[mask]
betas = betas[np.isin(probeAnno[:, posCol], testData["ref_position"]),:]
probeAnno = probeAnno[np.isin(probeAnno[:, posCol], testData["ref_position"])]

## caluclate mean DNAm level for each cell type
ctProbs = pd.DataFrame(betas).groupby(inputY, axis = 'columns').mean() ## matrix of probability of being methylated by cell type

## filter to read with at least 5 CpGs

IDcounts = testData["read_id"].value_counts()
testData = testData[testData["read_id"].isin(IDcounts[IDcounts > 5].index)]

## for each read train model & make prediction
readIDs = pd.unique(testData["read_id"])

nsites = IDcounts["01332cb7-881b-4d21-a7ff-0d8993084c43"]

train_obs = np.empty((nsites, nCT*nobs), dtype = int)
cpgProbs = ctProbs[np.isin(probeAnno[:, posCol], testData[testData["read_id"]=="01332cb7-881b-4d21-a7ff-0d8993084c43"]["ref_position"])]
i = 0
while i < nsites:
    j = 0
    while j < nCT:
        train_obs[i,np.arange((j*nobs),(j*nobs)+nobs)] = np.random.choice(methStatus, nobs, p=[1-ctProbs[i,j],ctProbs[i,j]])
        j +=1
    i +=1


X = np.transpose(train_obs[site_index:(site_index+ncpg),:])
model = initiateModel(modelType, nCT)
model.fit = model.fit(X,Y) 
test_pred = model.predict(np.transpose(test_obs[site_index:(site_index+ncpg),:,l]))
bool_correct = np.equal(test_pred, Y)