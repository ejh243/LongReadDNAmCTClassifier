#!/bin/sh
#SBATCH --export=ALL # export all environment variables to the batch job.
#SBATCH -p mrcq # submit to the serial queue
#SBATCH --time=100:00:00 # Maximum wall time for the job.
#SBATCH -A Research_Project-MRC190311 # research project to submit under. 
#SBATCH --nodes=1 # specify number of nodes.
#SBATCH --ntasks-per-node=16 # specify number of processors per node
#SBATCH --mail-type=END # send email at job completion 
#SBATCH --output=LogFiles/testONTData-%A.o
#SBATCH --error=LogFiles/testONTData-%A.e
#SBATCH --job-name=testONTData-%A.e


## loops through chromosomes and model types to select regions with good enough accuracy for testing on ONT data

# SCENARIO PARAMETERS
NOBS=100

# Software paths
MODKITPATH=~/software/modkit_v0.6.1_481e3c9/modkit

# Training data 
# path to beta matrices with DNAm data to train ML models
TRAINPATH=$1

# Test data
# path to bam file with ONT data to extract read level statistics for testing ML models
BAMPATH=$2

# Model data
# path to output folder from model simulations to select regions for testing on ONT data
RESULTSPATH=$3
NCT=$4
CTCOL=$5

# Output paths
# path to output folder for read level statistics extracted from ONT data and model predictions for these reads
OUTDIR=$6

module load Miniconda3
source activate cellclassifier

echo -e "\n=============================="
echo -e "  CHANGING TO SOFTWARE DIRECTORY"
echo -e "==============================\n"

cd ~/LongReadDNAmCTClassifier/

# select regions with good enough accuracy 
# runs per chr and model output folder, but does all models together.
# outputs one file per modelType, cell type prediction, chr
echo -e "\n=============================="
echo -e "  SELECTING TESTING REGIONS"
echo -e "==============================\n"
for CHR in {1..22}; do
    python3.9 summariseResults/writeRegionsToFile.py ${RESULTSPATH} ${CHR} ${NCT}
done



# loop through model types and predict cell types for all reads in all regions that passed the threshold for one chromosome and one model type
for MLTYPE in KNN SVM NBayes; do
    TESTREADPATH=${OUTDIR}/Lymphocytes/${MLTYPE}/
    for CHR in {1..22}; do
        REGIONS=${RESULTSPATH}MergedPredictiveRegionsThreshold0.95Model${MLTYPE}Chr${CHR}.csv
        echo -e "\n=============================="
        echo -e "  STARTING TESTING FOR ${MLTYPE} MODEL"
        echo -e "  CHROMOSOME: ${CHR}"
        echo -e "==============================\n"

        # Extract read level ONT data for one bam file all regions for one regions file (i.e. one ML algorithm for one chr)
        formatFiles/formatONTData.sh ${MODKITPATH} ${BAMPATH} ${OUTDIR} ${REGIONS}

        # train and test these regions
        # predicts all reads for one region for one model type
        ls ${TESTREADPATH}*.tsv | while read testFile; do 
            python3.9 testCellTypeClassifierONTData.py ${TRAINPATH} ${testFile} ${CTCOL} ${NOBS} 
        done
    done
done

