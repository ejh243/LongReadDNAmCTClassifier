
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
        model = svm.SVC(probability=True)
    else:
        sys.exit('Model type not recognised')
    return(model)

np.random.seed(randomSeed) ## ensures that the sample random sampling occurs across algorithms and across reruns of the script

## process command line information
trainDataPath = sys.argv[1]
testFile = sys.argv[2]
ctCol = int(sys.argv[3])
nobs = int(sys.argv[4]) # how many "single cell" observations to simulate per cell type


# parse filename for model type, chromosome, sample ID
testDir = os.path.dirname(testFile)
testBasename = os.path.basename(testFile) #<sample>_<region>.tsv
modelType = os.path.basename(testDir)  # folder just above the file
testBasenameNoExt = os.path.splitext(testBasename)[0]  # remove .tsv
sample, region = testBasenameNoExt.rsplit("_", 1)  # split into two parts
chr=region.split(":")[0].removeprefix("chr")  # extract chromosome from region

# test if region already tested
if os.path.exists(testDir + "/PredictionOutput/" + sample + "_" + region + "_modelPredictions.csv"):
    print("Region " + region + " already tested. Skipping.")
    sys.exit()

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

print("Filtered to " + str(np.shape(probeAnno)[0]) + " sites in training data")

## load test data
print ("Loading test data from " + testFile)
testData = pd.read_csv(testFile,sep='\t').values
testData = pd.DataFrame(testData, columns = ["read_id", "forward_read_position","ref_position","chrom","mod_strand","ref_strand","ref_mod_strand","fw_soft_clipped_start","fw_soft_clipped_end","alignment_start","alignment_end","read_length","mod_qual","mod_code","base_qual","ref_kmer","query_kmer","canonical_base","modified_primary_base","inferred","flag"])
# filter 5mC only
testData = testData[testData["mod_code"] == "m"]
# filter qual score
testData = testData[testData["base_qual"] > 10]
print("Methylation status loaded for " + str(np.shape(testData)[0]) + " CpGs in test data.")

if np.shape(testData)[0] == 0:
    print("No methylation sites left in test data. Exiting.")
    sys.exit()

# filter to overlap with training data
testData = testData[testData["ref_position"].isin(probeAnno[:,posCol])]
print("After filtering " + str(np.shape(testData)[0]) + " CpGs in test data.")




# filter training data to sites in test data
betas = betas[np.isin(probeAnno[:, posCol], testData["ref_position"]),:]
probeAnno = probeAnno[np.isin(probeAnno[:, posCol], testData["ref_position"]),:]
print("Training data filtered to " + str(np.shape(probeAnno)[0]) + " sites that overlap with test data.")

## caluclate mean DNAm level for each cell type
ctProbs = pd.DataFrame(betas).groupby(inputY, axis = 'columns').mean() ## matrix of probability of being methylated by cell type

## filter to read with at least 5 CpGs
IDcounts = testData["read_id"].value_counts()
testData = testData[testData["read_id"].isin(IDcounts[IDcounts > 5].index)]
IDcounts = testData["read_id"].value_counts()

## array of cell type labels for simulated data 
Y = np.repeat(np.arange(0,nCT), nobs)

## for each read train model & make prediction
readIDs = pd.unique(testData["read_id"])
all_test_pred = []
all_test_prob = []

print("Predicting " + str(len(readIDs)) + " reads in test data")

for read in readIDs:
    subTestData = testData[testData["read_id"]==read]
    nsites = subTestData.shape[0]
    print("Testing data for read " + read + " with " + str(nsites) + " sites")
    # binarise test data
    testMeth = np.where(subTestData["mod_qual"] > 0.1, 1, 0)
    # create object for training data
    train_obs = np.empty((nsites, nCT*nobs), dtype = int)
    # get probability of methylation for each site in this read
    cpgProbs = ctProbs[np.isin(probeAnno[:, posCol], subTestData["ref_position"])]
    print("Training model for read " + read + " with " + str(nsites) + " sites")
    i = 0
    while i < nsites:
        j = 0
        while j < nCT:
            train_obs[i,np.arange((j*nobs),(j*nobs)+nobs)] = np.random.choice(methStatus, nobs, p=[1-cpgProbs.iloc[i,j],cpgProbs.iloc[i,j]])
            j +=1
        i +=1
    # train model
    X = np.transpose(train_obs)
    model = initiateModel(modelType, nCT)
    model.fit = model.fit(X,Y) 
    # make prediction
    all_test_pred.append(model.predict(testMeth.reshape(1,-1))[0])
    all_test_prob.append(model.predict_proba(testMeth.reshape(1, -1))[0][1])



results = pd.DataFrame({
    "nCpG": IDcounts[readIDs].values,     
    "read_id": readIDs,            
    "predicted_cell_type": all_test_pred,       
    "probability_cell_type": all_test_prob       
})

os.makedirs(testDir + "/PredictionOutput/", exist_ok=True)
results.to_csv(testDir + "/PredictionOutput/" + sample + "_" + region + "_modelPredictions.csv", index = False)
