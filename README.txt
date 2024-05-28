
 
All code necessary to reproduce our experiments and results in in the GraphSAGE_NET.ipynb file.

 
first :  download MIMIC data and put DIAGNOSES_ICD.csv, PRESCRIPTIONS.csv, PROCEDURES_ICD.csv in ./data/
download DDI data and put it in ./data/



1:install necessary packages :

!pip install torch==2.1.0
!pip install torch_geometric dgl dill
!pip install rdkit dnc

2: process data with : run processing.py in data folder

3: run SAGENet.Train

