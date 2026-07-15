# for a series of bam file, extract read level statistics for a given region and save as tsv file
# uses modkit
# extract regions from csv file in bed format

MODKITPATH=$1
BAMPATH=$2
OUTDIR=$3
REGIONS=$4

# create output filename: needs to include region information, model type, model prediction output, sample info
sample=$(basename ${BAMPATH%.bam})
dir=$(dirname "$REGIONS")
predictCT=$(basename "$dir")
regionInfo=$(basename "$REGIONS")
# Remove the .csv extension
base=${regionInfo%.csv}
# Extract the model type (KNN)
mlType=${base#*Model}        # removes everything up to "Model"
mlType=${mlType%%Chr*}        # removes everything from "Chr" onward

# if folder doesn't exist, create it
RESDIR=${OUTDIR}/${predictCT}/${mlType}/
mkdir -p ${RESDIR}

## extract read level statistics for each region
tail -n +2 ${REGIONS} | while IFS=',' read -r chr start end _; do
    region="chr${chr}:${start}-${end}"
    # check if region already extracted
    if [ ! -f "${RESDIR}/${sample}_${region}.tsv" ]; then
        echo "Extracting region: " $region "for sample " ${sample}
        ${MODKITPATH} extract full ${BAMPATH} ${RESDIR}/${sample}_${region}.tsv --region ${region}
    fi
done

echo "Completed"

