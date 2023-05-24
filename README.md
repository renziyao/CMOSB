# Environment Setup
- python: 3.8.5
- pip install -r requirements.txt

# Dataset

The dataset should be downloaded to `./data`

DefaultCredit: 
https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset

Sensorless: 
https://archive.ics.uci.edu/ml/datasets/dataset+for+sensorless+drive+diagnosis

# Run Experiments

#### Binary Classification

Run `python optimizer_binary.py`

#### Multi-class Classification

Run `python optimizer_multi.py`

#### Optional Parameter

1. dataset

Available datasets: synthetic1, synthetic2, defaultcredit, sensorless

2. generation

The generation in the proposed CMOSB algorithm. 

3. dir

Directory for storing experimental results.
