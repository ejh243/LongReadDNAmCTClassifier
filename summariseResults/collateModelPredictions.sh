# finds all model prediction output files in a folder and appends into a single file

# Check if a folder argument is provided
if [ $# -eq 0 ]; then
  echo "Usage: $0 <folder_path>"
  echo "Example: $0 /path/to/predictions/folder"
  exit 1
fi

folder="${1}"

# Check if the folder exists
if [ ! -d "$folder" ]; then
  echo "Error: Folder '$folder' does not exist"
  exit 1
fi

out="$folder/mergedModelPredictions.csv"
first=1

# Change to the specified folder and process files
cd "$folder" || exit 1

for f in *_modelPredictions.csv; do
  # Skip if no files match the pattern
  if [ ! -f "$f" ]; then
    continue
  fi

  base=${f%_modelPredictions.csv}   # B_1_chr1:...
sample=${base%%_chr*}             # B_1
sample_type=${sample%%_*}         # B
sampleID=${sample#*_}             # 1

  region=${base##*_chr}
  region="chr${region}"


  if [ $first -eq 1 ]; then
    awk -v st="$sample_type" -v sid="$sampleID" -v r="$region" 'BEGIN{OFS=","}
      NR==1 {print "sample_type","sampleID","region",$0}
      NR>1  {print st,sid,r,$0}
    ' "$f" > "$out"
    first=0
  else
    awk -v st="$sample_type" -v sid="$sampleID" -v r="$region" 'BEGIN{OFS=","}
      NR>1 {print st,sid,r,$0}
    ' "$f" >> "$out"
  fi
done

echo "Merged predictions saved to: $out"