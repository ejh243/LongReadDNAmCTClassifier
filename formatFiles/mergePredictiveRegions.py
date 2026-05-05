# merge regions to filter bam file

import pandas as pd
import glob
import pyranges as pr
import sys

# Path to your folder
resultsPath=sys.argv[1]

# Get all CSV files
files = glob.glob(f"{resultsPath}/*/MergedPredictiveRegions*.csv")

# Load and concatenate
regions = [pd.read_csv(f) for f in files]
regions = pd.concat(regions, ignore_index=True)

gr = pr.PyRanges(regions)
merged = gr.merge()

merged_df = merged.as_df()

# Save as bed file
merged_df.to_csv(f"{resultsPath}/MergedPredictiveRegions.bed", index=False, sep="\t",
    header=False)
