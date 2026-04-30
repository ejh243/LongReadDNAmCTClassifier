# extract read meta data 
# provide path to folder with bam files on command line
# provide path to bed file to filter bam reads on command line

bamPath=$1
regionBed=$2

cd $bamPath
output="read_meta_data.txt"
: > "$output"

# Write header once
echo -e "read_id\tchrom\talignment_start\tread_length" > "$output"

for f in *.bam; do
    echo "Processing $f"

    samtools view -F 4 -L "$regionBed" "$f" | awk '{
        cigar = $6
        total = 0
        # Count S, H, M, I, =, X as read-length operations
        while (match(cigar, /[0-9]+[SHMI=X]/)) {
            val = substr(cigar, RSTART, RLENGTH-1)
            total += val
            cigar = substr(cigar, RSTART + RLENGTH)
        }
        print $1"\t"$3"\t"$4"\t"total
    }' >> "$output"
done
