# EPMDA
the implementation of EPMDA
## required python packages
```
python 2.7
sklearn
codecs
re
multiprocessing
numpy
```
## how to use
### predict new miRNA-disease pair
```
run python EPMDA.py
```
### 5-fold cross validation
```
run pythobn 5FoldCV.py
```
### leave one disease out cross validtaion
```
run python LODOCV.py
```
## data
```
miRNA.xlsx            the mapping file of miRNAs
disease.xlsx          the mapping file of diseases
miRNA-disease.txt     miRNA-disease associations
Gaussian_disease.csv  Gaussian kernel similarity between each pair of diseases
Gaussian_miRNA.csv    Gaussian kernel similarity between each pair of miRNAs
```
## predicted results
```
data/newPredicted.csv the new predicted scores of all unknown disease-miRNA pairs
