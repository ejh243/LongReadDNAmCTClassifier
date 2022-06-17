# file containing parameters for machine learning algorithms

randomSeed = 0
pThres = 0.00005 ## threshold to keep probes with signif diffs
methStatus = (0,1) 
max_window = 20000 # max distance between outer most cpgs in classifier
min_cpg = 5 # number of cpgs in first classifer
nCV_splits = 3 # parameters for cross validation
nCV_repeats = 5 # parameters for cross validation
nCV = nCV_splits*nCV_repeats # number of iterations to simulate cross validation

pThread = -1 ## number of parallel threads (-1 uses all)


