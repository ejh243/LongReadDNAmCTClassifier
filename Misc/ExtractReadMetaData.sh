# extract read meta data for all reads that overlap the regions selected for testing (i.e. those with good enough accuracy in the training phase) and output one file per model type and chromosome with read level statistics for all reads that overlap the selected regions for testing.
# provide path to folder with tsv files containing read level statistics for all reads that overlap the selected regions for testing (i.e. output from formatFiles/formatONTData.sh) and path to output folder for read level statistics extracted from ONT data and model predictions for these reads as arguments.

cd $1

output="read_meta_data.txt"
: > "$output"

# Write header once
echo -e "read_id\tchrom\tmod_strand\tref_strand\talignment_start\talignment_end\tread_length" > "$output"

for f in *.tsv; do
    echo "Processing $f"

    awk '{ print $1, $4, $5, $6, $10, $11, $12 }' "$f" \
        | awk '!seen[$0]++' \
        >> "$output"
done
