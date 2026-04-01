#!/bin/sh
#SBATCH --export=ALL # export all environment variables to the batch job.
#SBATCH -p mrcq # submit to the serial queue
#SBATCH --time=100:00:00 # Maximum wall time for the job.
#SBATCH -A Research_Project-MRC190311 # research project to submit under. 
#SBATCH --nodes=1 # specify number of nodes.
#SBATCH --ntasks-per-node=16 # specify number of processors per node
#SBATCH --mail-type=END # send email at job completion 
#SBATCH --output=LogFiles/format-%A_%a.o
#SBATCH --error=LogFiles/format-%A_%a.e
#SBATCH --job-name=format-%A_%a.e



cpgPath=$1
samplePath=$2
outPath=$3


python3.9 ~/LongReadDNAmCTClassifier/formatFiles/formatWGBSFromBetaFiles.py ${cpgPath} ${samplePath} ${outPath} ${SLURM_ARRAY_TASK_ID}  > ~/LongReadDNAmCTClassifier/LogFiles/reformtWGBS_Chr${SLURM_ARRAY_TASK_ID}.log

