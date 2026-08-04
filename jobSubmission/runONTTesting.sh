#!/bin/sh
#SBATCH --export=ALL # export all environment variables to the batch job.
#SBATCH -p mrcq # submit to the serial queue
#SBATCH --time=100:00:00 # Maximum wall time for the job.
#SBATCH -A Research_Project-MRC190311 # research project to submit under. 
#SBATCH --nodes=1 # specify number of nodes.
#SBATCH --ntasks-per-node=16 # specify number of processors per node
#SBATCH --mail-type=END # send email at job completion 
#SBATCH --output=LogFiles/testONTData-%A_%a.o
#SBATCH --error=LogFiles/testONTData-%A_%a.e
#SBATCH --job-name=testONTData-%A_%a.e
#SBATCH --array=1-22

# Assumes github repository folder is in $HOME directory

## submits job for each chromosome
CHR=${SLURM_ARRAY_TASK_ID}

# SCENARIO PARAMETERS
NOBS=100

# Software paths
MODKITPATH=~/software/modkit_v0.6.1_481e3c9/modkit

# Training data 
# Path to beta matrices with DNAm data to train ML models
TRAINPATH=$1

# Test data
# Path to bam file with ONT data to extract read level statistics for testing ML models
BAMPATH=$2

# Model data
# Path to output folder from model simulations to select regions for testing on ONT data
RESULTSPATH=$3
NCT=$4
CTCOL=$5
MODELNAME=$(basename "${RESULTSPATH%/}") # which cell type output is predicted

# Output directory for read-level DNAm information, & model predictions
# Model information will be derived automatically from RESULTSPATH.
# Top-level results directory only; cell-type model and ML-algorithm subfolders will be appended automatically
# Predictions will be found in ${OUTDIR}/${MODELNAME}/${MLTYPE}/PredictionOutput/
OUTDIR=$6

module load Miniconda3
source activate cellclassifier

echo -e "\n=============================="
echo -e "  CHANGING TO SOFTWARE DIRECTORY"
echo -e "==============================\n"

cd ~/LongReadDNAmCTClassifier/

echo -e "\n=============================="
echo -e " PROCESSING CHROMOSOME: ${CHR}"
echo -e "==============================\n"

# select regions with good enough accuracy 
# runs per chr and model output folder, but does all models together.
# outputs one file per modelType, cell type prediction, chr
echo -e "\n=============================="
echo -e "  SELECTING TESTING REGIONS"
echo -e "==============================\n"


python3.9 summariseResults/writeRegionsToFile.py ${RESULTSPATH} ${CHR} ${NCT}


# loop through model types and predict cell types for all reads in all regions that passed the threshold for one chromosome and one model type
for MLTYPE in KNN SVM NBayes; do
    TESTREADPATH=${OUTDIR}/${MODELNAME}/${MLTYPE}/
	echo -e "\n=============================="
	echo -e "  PREPARING TESTING FOR ${MLTYPE} MODEL"
	echo -e "==============================\n"
	# only need to run this for the most permissive threshold, as the regions for the other thresholds are subsets of this one.
	LOWEST_THRESHOLD=$(
    for f in ${RESULTSPATH}MergedPredictiveRegionsThreshold*Model${MLTYPE}Chr*.csv; do
        thres=${f##*MergedPredictiveRegionsThreshold}
        thres=${thres%%Model${MLTYPE}Chr*}
        echo "$thres"
		done | sort -g | uniq | head -n1
	)

	echo "Most permissive threshold for ${MLTYPE}: ${LOWEST_THRESHOLD}"
	REGIONS=${RESULTSPATH}MergedPredictiveRegionsThreshold${LOWEST_THRESHOLD}Model${MLTYPE}Chr${CHR}.csv
	# Extract read level ONT data for one bam file all regions for one regions file (i.e. one ML algorithm for one chr)
	formatFiles/formatONTData.sh ${MODKITPATH} ${BAMPATH} ${OUTDIR} ${REGIONS}
	
	echo -e "\n=============================="
	echo -e "  RUNNING TESTING FOR ${MLTYPE} MODEL"
	echo -e "==============================\n"
	# train and test these regions
	# predicts all reads for one region for one model type
	find "$TESTREADPATH" -maxdepth 1 -name "*chr$CHR:*.tsv" | while read -r testFile; do
		python3.9 testCellTypeClassifierONTData.py "$TRAINPATH" "$testFile" "$CTCOL" "$NOBS"
	done
done

echo -e "\n=============================="
echo -e "  PREDICTIONS COMPLETE "
echo -e "  TIDYING UP INTERMEDIATE FILES "
echo -e "==============================\n"

find "$TESTREADPATH" -maxdepth 1 -name "*chr$CHR:*.tsv" -print0 | xargs -0 rm
