#!/bin/sh
#SBATCH --export=ALL # export all environment variables to the batch job.
#SBATCH -p mrcq # submit to the serial queue
#SBATCH --time=200:00:00 # Maximum wall time for the job.
#SBATCH -A Research_Project-MRC190311 # research project to submit under. 
#SBATCH --nodes=1 # specify number of nodes.
#SBATCH --ntasks-per-node=16 # specify number of processors per node
#SBATCH --mail-type=END # send email at job completion 
#SBATCH --output=LogFiles/mergeBinRegionsByChr-%A-%a.o
#SBATCH --error=LogFiles/mergeBinRegionsByChr-%A-%a.e
#SBATCH --job-name=mergeBinRegionsByChr-%A-%a.e
#SBATCH --array=1-22

module load Miniconda3
source activate cellclassifier

resultsPath=$1
nCT=$2

python summariseResults/mergeBinaryModelsRegionsByChr.py ${resultsPath} ${SLURM_ARRAY_TASK_ID} ${nCT} > LogFiles/mergeRegions_${SLURM_JOB_ID}_chr${SLURM_ARRAY_TASK_ID}.log
