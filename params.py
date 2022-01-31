# file containing parameters for machine learning algorithms

randomSeed = 0
pThres = 0.0005 ## threshold to keep probes with signif diffs
nCT = 5 ## number of cell types
methStatus = (0,1) 
max_window = 20000 # max distance between outer most cpgs in classifier
min_cpg = 2 # number of cpgs in first classifer
nCV_splits = 3 # parameters for cross validation
nCV_repeats = 5 # parameters for cross validation
nCV = 15 # number of iterations to simulate cross validation

pThread = 10 ## number of parallel threads

workDir = "/mnt/data1/Eilis/Projects/Asthma/ClassifyCellTypes/"
trainDataPath = "TrainingData/"
resultsPath = "Results/"

