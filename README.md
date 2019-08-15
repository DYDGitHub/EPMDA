# EPMDA
the implementation of EPMDA
## Required python packages
```
python 2.7
sklearn
codecs
re
multiprocessing
numpy
```
## How to use
### Predict new miRNA-disease pair
```
run python EPMDA.py
```
### 5-fold cross validation
```
run pythobn 5FoldCV.py
```
### Leave one disease out cross validtaion
```
run python LODOCV.py
```
## Data
```
miRNA.xlsx            the mapping file of miRNAs
disease.xlsx          the mapping file of diseases
miRNA-disease.txt     miRNA-disease associations
Gaussian_disease.csv  Gaussian kernel similarity between each pair of diseases
Gaussian_miRNA.csv    Gaussian kernel similarity between each pair of miRNAs
```
## Predicted results
```
data/newPredicted.csv the new predicted scores of all unknown disease-miRNA pairs
