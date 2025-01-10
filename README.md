# Data Valuation Using Reinforcment Learning For Time Series Data

This repository contains an adapted implementation of Data Valuator using Reinforcement Learning (DVRL) for handling time series datasets. The original [DVRL model](https://github.com/google-research/google-research/tree/master/dvrl), designed for regression and classification tasks, has been modified to evaluate and remove corrupted or low-quality data from time series datasets.

## Overview
Goal is to clean time series datasets by identifying and removing bad or corrupted data samples using reinforcement learning techniques.
Following is done to accomplish this:

1. Data Preprocessing: The time series dataset is prepared for DVRL using custom preprocessing steps.

2. Data Sampler: Time series data is sampled based on binomial distribution

3. Predictor Model: An LSTM-based predictor model is trained on these samples.

4. Data Valuation: Based on how well the LSTM learns on these samples compared to the moving average, the valuation of the data samples is done.

## Repository Structure

```plaintext
.
├── data_preprocess.py         # Preprocessing utilities for time series datasets
├── data_valuator.py           # Core DVRL implementation for time series
├── functions_for_dvrl.py      # Supporting functions for DVRL
├── generate_dataset.py        # Scripts to generate synthetic time series datasets
├── lstm_encoder_decoder.py    # LSTM encoder-decoder for predictor model
├── train.ipynb                # Notebook to train DVRL and evaluate results
├── requirements.txt           # Python dependencies
├── testPuneAQMNew_22.csv      # Example dataset: Air Quality Monitoring in Pune
└── README.md                  # Project documentation
```

## Installation

1. Clone the repo and install the dependencies

2. Run the `train.ipynb`
