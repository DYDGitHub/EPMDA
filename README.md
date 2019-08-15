# EPMDA
the implementation of EPMDA
## 1. Required python packages
```
python 2.7
sklearn
codecs
re
multiprocessing
numpy
```
## 2. How to use
### 2.1 Predict new miRNA-disease pair
```
run python EPMDA.py
```
### 2.2 5-fold cross validation
```
run python 5FoldCV.py
```
### 2.3 Leave one disease out cross validation
```
run python LODOCV.py
```
## 3. Data
```
miRNA.xlsx            the mapping file of miRNAs
disease.xlsx          the mapping file of diseases
miRNA-disease.txt     each row represents an miRNA-disease pair, the two columns represent miRNA and disease indexes respectively
Gaussian_disease.csv  Gaussian kernel similarity between each pair of diseases
Gaussian_miRNA.csv    Gaussian kernel similarity between each pair of miRNAs
```
## 4. Predicted results
```
data/newPredicted.csv the new predicted scores of all unknown disease-miRNA pairs
