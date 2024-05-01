# LongReadDNAmCTClassifier

## Training data format
The training data for the cell type predictor is split into text files for each chromosome, where all input files are located in the same directory. There are two input files for each chromosome with the following filenames:

1. betas_chr<chr>.csv
	A matrix of DNA methylation values [0-1], where columns are samples and rows are cg sites. There are no rownames but there are column names. 

2. rowanno_chr<chr>.csv
	A table of annotations for each row of the betas matrix, where the final three columns are ANOVA p-value, chr and bp position. 
	
In addition there is a single csv file which specifies which sample what each row in the beta matrix is (colanno.csv). 

## Scripts to train cell type classifiers


### Classifiers trained with binary feature data

```
python CellTypeClassifierBinaryDNAm.py ${chr} ${modelType} ${trainPath} ${outPath} ${cellCol} ${nobs} ${array}
```

Arguments are:

`chr` chromosome as an integer to process (need to run once for each chromosome)

`modelType` one of "KNN", "SVM", "NBayes" or "RandFor"

`trainPath` full path to folder of training data - formatted as described above

`outPath` full to folder where output should be saved

`cellCol` column number as am integer of colanno.csv that contains the cell type labels

`nobs` number of observations per cell type to synthesise for the train and test data

`array` TRUE/FALSE depending if observed DNAm data is generated from a microarray or not.

Output is a text file.

### Classifiers trained with continuous feature data

```
python CellTypeClassifierContinuousDNAm.py ${chr} ${modelType} ${trainPath} ${outPath} ${cellCol}  
```

Arguments are:

`chr` chromosome as an integer to process (need to run once for each chromosome)

`modelType` one of "KNN", "SVM", "NBayes" or "RandFor"

`trainPath` full path to folder of training data - formatted as described above

`outPath` full to folder where output should be saved

`cellCol` column number as am integer of colanno.csv that contains the cell type labels

Output is a text file.

## Scripts to summarise results from classifiers

### Classifiers trained with binary feature data

```
python summariseResults/summariseBinarisedModels.py ${resultsPath} ${nCT}
```

takes output from `CellTypeClassifierBinaryDNAm.py` and collates across chr to create summary plots

Arguments are:

`resultsPath` full path to folder with chromosome level results 

`nCT` number of cell types predicted

```
python summariseResults/summariseAcrossCellTypesBinary.py ${resultsPath} ${cellTypes}
```

takes output from `CellTypeClassifierBinaryDNAm.py` which has been run across different cell type groupings and create summary plots

Arguments are:

`resultsPath` full path to folder with folders for results of each cell type grouping

`cellTypes` list of names of folders with each cell type grouping with space between each grouping e.g. "Lympocytes" "Bcells"

Output is put into a folder called `Plots` in `resultsPath`

```
python summariseResults/mergeBinaryModelsRegionsByChr.py ${resultsPath} ${chr} ${nCT}
```

takes output from `CellTypeClassifierBinaryDNAm.py` and collapses into regions

Arguments are:

`resultsPath` full path to folder with chromosome level results 

`chr` chromosome number to process (needs to be run sperartely for each chromosome)

`nCT` number of cell types predicted

```
python summariseResults/plotBinaryRegionsAcrossCellTypes.py ${resultsPath} ${cellTypes}
```
takes output from `mergeBinaryModelsRegionsByChr`  which has been run across different cell type groupings and create summary plots

Arguments are:

`resultsPath` full path to folder with chromosome level results 

`cellTypes` list of names of folders with each cell type grouping with space between each grouping e.g. "Lympocytes" "Bcells"

Output is put into a folder called `Plots` in `resultsPath`

### Classifiers trained with continuous feature data


```
python summariseResults/summariseContinuousModels.py ${resultsPath} ${nCT}
```

takes output from `CellTypeClassifierContinuousDNAm.py` and collates across chr to create summary plots

Arguments are:

`resultsPath` full path to folder with chromosome level results 

`nCT` number of cell types predicted

```
python summariseResults/summariseAcrossCellTypesContinuous.py ${resultsPath} ${cellTypes}
```

takes output from `CellTypeClassifierContinuousDNAm.py` which has been run across different cell type groupings and create summary plots

Arguments are:

`resultsPath` full path to folder with folders for results of each cell type grouping

`cellTypes` list of names of folders with each cell type grouping with space between each grouping e.g. "Lympocytes" "Bcells"

Output is put into a folder called `Plots` in `resultsPath`

```
python summariseResults/mergeContinuousModelsRegionsByChr.py ${resultsPath} ${chr} ${nCT}
```

takes output from `CellTypeClassifierContinuousDNAm.py` and collapses into regions

Arguments are:

`resultsPath` full path to folder with chromosome level results 

`chr` chromosome number to process (needs to be run sperartely for each chromosome)

`nCT` number of cell types predicted

```
python summariseResults/combineChrContinuousModelsRegions.py ${resultsPath}

```
takes output from `mergeContinuousModelsRegionsByChr` and aggregates across chromosomes

Output is a csv file 

Arguments are:

`resultsPath` full path to folder with chromosome level results 

```
python summariseResults/plotContinuousRegionsAcrossCellTypes.py ${resultsPath} ${cellTypes}
```
takes output from `mergeContinuousModelsRegionsByChr.py`  which has been run across different cell type groupings and create summary plots

Arguments are:

`resultsPath` full path to folder with chromosome level results 

`cellTypes` list of names of folders with each cell type grouping with space between each grouping e.g. "Lympocytes" "Bcells"

Output is put into a folder called `Plots` in `resultsPath`