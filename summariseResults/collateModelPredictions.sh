# finds all model prediction output files in a folder and appends into a single file

out="merged_modelPredictions.csv"
first=1

for f in *_modelPredictions.csv; do
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
