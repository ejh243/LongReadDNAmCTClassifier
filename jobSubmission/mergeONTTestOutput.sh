#!/bin/sh
#SBATCH --export=ALL # export all environment variables to the batch job.
#SBATCH -p mrcq # submit to the serial queue
#SBATCH --time=100:00:00 # Maximum wall time for the job.
#SBATCH -A Research_Project-MRC190311 # research project to submit under. 
#SBATCH --nodes=1 # specify number of nodes.
#SBATCH --ntasks-per-node=16 # specify number of processors per node
#SBATCH --mail-type=END # send email at job completion 
#SBATCH --output=LogFiles/mergeTestOutput-%A.o
#SBATCH --error=LogFiles/mergeTestOutput-%A.e
#SBATCH --job-name=mergeTestOutput-%A.e

# Path to folder with results to merge
OUTPATH=$1

sh summariseResults/collateModelPredictions.sh $1