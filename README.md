# LongReadDNAmCTClassifier

## Training data format
The training data for the cell type predictor is split into text files for each chromosome, where all input files are located in the same directory. There are two input files for each chromosome with the following filenames:

1. betas_chr<chr>.csv
	A matrix of DNA methylation values [0-1], where columns are samples and rows are cg sites. There are no rownames but there are column names. 

2. rowanno_chr<chr>.csv
	A table of annotations for each row of the betas matrix, where the final three columns are ANOVA p-value, chr and bp position. 
	
In addition there is a single csv file which specifies which sample what each row in the beta matrix is. 