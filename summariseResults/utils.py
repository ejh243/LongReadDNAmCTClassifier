import pandas as pd
import numpy as np
import pyranges as pr

def loadResults(filename):
    return pd.read_csv(filename, header = 0, names = ("Chr", "Position", "nCpG", "WindowSize", "MeanAccuracy", "SDAccuracy"))


def cumSumAccuracy(data, bins):
	return np.asarray(np.cumsum(np.flip(pd.cut(data, bins = bins).value_counts(sort = False))))


def gaps(granges):
    ## doesn't make sens unless ranges contains non-overlapping regions
    ## get chromosome size info
    chrSizes = pr.data.chromsizes()
    chrList = np.unique(granges.Chromosome)
    newChr = []
    newStart = []
    newEnd = []
    for chr in chrList:
        boolIndex = granges.Chromosome == chr
        newChr.extend([chr]*(sum(boolIndex)+1)) ## need an extra chr
        newStart.extend([0])
        newStart.extend((granges.End[boolIndex].values+1).tolist())
        newEnd.extend((granges.Start[boolIndex].values - 1).tolist())
        if chr.startswith("chr"):
            newEnd.extend(chrSizes.End[chrSizes.Chromosome == chr].tolist())
        else:
            newEnd.extend(chrSizes.End[chrSizes.Chromosome == "chr" + str(chr)].tolist())
    return pr.PyRanges(chromosomes = newChr, starts = newStart, ends = newEnd)


