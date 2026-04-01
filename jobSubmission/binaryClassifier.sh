#!/bin/sh
#SBATCH --export=ALL # export all environment variables to the batch job.
#SBATCH -p mrcq # submit to the serial queue
#SBATCH --time=400:00:00 # Maximum wall time for the job.
#SBATCH -A Research_Project-MRC190311 # research project to submit under. 
#SBATCH --nodes=1 # specify number of nodes.
#SBATCH --ntasks-per-node=16 # specify number of processors per node
#SBATCH --mail-type=END # send email at job completion 
#SBATCH --output=LogFiles/binclassifier-%A_%a.o
#SBATCH --error=LogFiles/binclassifier-%A_%a.e
#SBATCH --job-name=binclassifier-%A_%a.e
#SBATCH --array=1-22

module load Miniconda3
source activate cellclassifier

modelType=$1
trainPath=$2
outPath=$3
cellCol=$4
array=$5
nobs=100

python ~/LongReadDNAmCTClassifier/CellTypeClassifierBinaryDNAm.py ${SLURM_ARRAY_TASK_ID} ${modelType} ${trainPath} ${outPath} ${cellCol} ${nobs} ${array} > ~/LongReadDNAmCTClassifier/LogFiles/BinaryClassifier_${modelType}_Chr${SLURM_ARRAY_TASK_ID}_cellCol${cellCol}_${nobs}Obs.log
