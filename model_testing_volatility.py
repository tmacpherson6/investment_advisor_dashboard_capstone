#123456789012345678901234567890123456789012345678901234567890123456789012345678
"""This module tests model predictions of future market sector volatility.

We chose a variety of Exchange Traded Funds (ETFs) to represent different 
market sectors, and we test three types of models, using three different 
feature sets each, to predict realized (historical) volatilty for each sector.

We test each model several times, recording predictions (for posterity) and
mean absolute error (MAE) of predictions, to obtain a distribution of MAE 
for each model, for each ETF.  We will then compare these figures against a
standard economic model for forecasting volatility (the VIX index) to assess
how well our models compare to a baseline.

A few financial concepts are important for understanding.

The return for an asset (in this case, we study ETFs that represent market
sectors) reflects a change in price of the asset over time.

An asset's volatility is a measure of the variance in the asset's returns.
There are many ways to estimate an asset's volatility, and we rely on 
"historical volatility," which is the standard deviation of past returns
over a period of time.  We also rely on a "zero-mean" assumption for 
*expected return*, which states that for any asset, we believe the true mean 
return to be zero.  This assumption generally holds in the very long run.
When returns are much higher or lower than usual (i.e., expectation), we say
that the asset is experiencing a period of high volatility.
"""
import json
import re
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    make_scorer,
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
    mean_squared_error
)
import torch

import data_prep as dp
import models

# Data folders
DATA_FOLDER = 'data/downsample/W/train-val-test/'
TRAIN_DATA_FILE = DATA_FOLDER + 'train_pca_data.csv'
VAL_DATA_FILE = DATA_FOLDER + 'val_pca_data.csv'
TEST_DATA_FILE = DATA_FOLDER + 'test_pca_data.csv'
# List of ETFs for volatility prediction
ETFS = [
    'BIL', 'BND', 'GLD', 'HYG', 'IEF', 'IWM', 'LQD',
    'QQQ', 'SPY', 'TIP', 'TLT', 'XLB', 'XLE', 'XLF',
    'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY'
]
# Number of trials to run
NUM_TRIALS = 10
# LSTM and Dataset/Dataloader parameters
SEQUENCE_LENGTH = 4
BATCH_SIZE = 12
LOSS_FUNCTION = torch.nn.L1Loss(reduction='mean')


def import_data(train_file, val_file, test_file):
    """Import training, validation, and testing data sets; scale features."""
    # Read data from files
    train_df = pd.read_csv(
        train_file,
        index_col='date',
        parse_dates=True
    )
    val_df = pd.read_csv(
        val_file,
        index_col='date',
        parse_dates=True
    )
    test_df = pd.read_csv(
        test_file,
        index_col='date',
        parse_dates=True
    )
    # Separate features (X) from labels (y)
    X_cols, y_cols = dp.get_X_y_cols(train_df, returns=True)
    # Scale features (X), taking care to fit only to the training set
    scaler = MinMaxScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(train_df[X_cols]),
        columns=X_cols,
        index=train_df.index
    )
    y_train = train_df[y_cols].copy()
    X_val = pd.DataFrame(
        scaler.transform(val_df[X_cols]),
        columns=X_cols,
        index=val_df.index
    )
    y_val = val_df[y_cols].copy()
    X_test = pd.DataFrame(
        scaler.transform(test_df[X_cols]),
        columns=X_cols,
        index=test_df.index
    )
    y_test = test_df[y_cols].copy()
    # Combine scaled features with labels
    scaled_train_df = pd.concat([X_train, y_train], axis=1)
    scaled_val_df = pd.concat([X_val, y_val], axis=1)
    scaled_test_df = pd.concat([X_test, y_test], axis=1)
    return scaled_train_df, scaled_val_df, scaled_test_df


def get_baseline(splits, etfs):
    """Computes MAE for baseline model volatility predictions for each ETF."""
    baseline_MAE = {}
    for etf in etfs:
        for split, df in splits.items():
            y_true = df[f'{etf}_vol_target'].values
            # Note: VIX predicts annualized volatility, so we standardize
            y_pred = df['^VIX'].values / np.sqrt(252) / 100
            baseline_MAE.setdefault(etf, {})
            baseline_MAE[etf][split] = mean_absolute_error(y_true, y_pred)
    return baseline_MAE


def run_trial(splits, etfs, baseline):
    """Conduct a trial for all models against all ETFs and record results."""
    # Define model types, feature sets, and ETFs
    model_types = ['Ridge', 'GBR', 'LSTM']
    feature_sets = ['limited', 'full', 'PCA']
    # Collect results
    results = {}
    for etf in etfs:
        for feature_set in feature_sets:
            for model_type in model_types:
                if (model_type == 'LSTM') and (feature_set == 'limited'):
                    print(
                        f'...LSTM ({feature_set}) is forecasting for {etf}...'
                    )
                # Train model
                model = train_model(model_type, feature_set, etf, train_df)
                for split, df in splits.items():
                    # Record predictions and metrics
                    mae, y_pred = test_model(
                        model, model_type, feature_set, etf, (split, df)
                    )
                    results.setdefault(etf, {})
                    results[etf].setdefault(model_type, {})
                    results[etf][model_type].setdefault(feature_set, {})
                    results[etf][model_type][feature_set].setdefault(split, {})
                    results[etf][model_type][feature_set][split]['y_pred'] = (
                        y_pred
                    )
                    results[etf][model_type][feature_set][split]['MAE'] = mae
                    results[etf][model_type][feature_set][split]['R2_OOS'] = (
                        1 - mae / baseline[etf][split]
                    )
    return results


def train_model(model_type, feature_set, etf, train_df):
    """Trains a model to predict volatility for an etf.
    
    Note: This function has been scaled back so that we only train the LSTM
    using PCA-extracted features, due to time and compute constraints.  A more
    complete implementation would actually train all models.
    """
    X_cols = get_features(train_df, feature_set, etf)
    y_col = [f'{etf}_vol_target']
    # LSTM uses a different training paradigm, unlike scikit-learn models
    if model_type == 'LSTM':
        # Set parameters specific to the LSTM
        input_size, label_size = len(X_cols), len(y_col)
        # Certain hyperparameters were better for different feature sets
        hypers = {
            'limited': [16, 1000],  # hidden layer size, number of epochs
            'full': [64, 100],
            'PCA': [24, 200]
        }
        lstm_params = [
            input_size,
            label_size
        ] + [hypers[feature_set][0]]
        # Create Dataset and DataLoader
        train_dataset = dp.FinancialDataset(
            train_df[X_cols].values.astype(np.float32),
            train_df[y_col].values.astype(np.float32),
            sequence_length=SEQUENCE_LENGTH,
            target=True
        )
        train_dataloader = dp.DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=False,
            drop_last=False
        )
        # Specify model and move to device for computation
        device = get_device()
        model = get_model(model_type, lstm_params).to(device)
        # Define loss function and optimizer
        loss_fn = LOSS_FUNCTION
        optimizer = torch.optim.Adam(model.parameters())
        # Train model over number of epochs (tuned for the feature set)
        epochs = hypers[feature_set][1]
        for epoch in range(epochs):
            models.train_model(train_dataloader, model, loss_fn, optimizer)
    else:
        model = None  # Placeholder for future implementation
    return model


def get_features(df, feature_set, etf):
    """Provides a list of feature columns for the chosen feature set."""
    ## Limited features
    limited = [f'{etf}_ret-h', f'{etf}_vol-h']
    ## Full features
    full, _ = dp.get_X_y_cols(df, returns=True)
    ## PCA features
    p = re.compile(r'PCA.+')
    pca = [col for col in train_df.columns if p.match(col) is not None]
    pca = pca + [f'{etf}_ret-h', f'{etf}_vol-h']
    X_cols = {
        'limited': limited,
        'full': full,
        'PCA': pca
    }
    return X_cols[feature_set]


def get_device():
    """Sets the device for PyTorch tensor computation."""
    # Set an accelerator device (e.g., CUDA) if available.
    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator().type
    else:
        device = 'cpu'
    return device


def get_model(model_type, lstm_params=[1, 1, 16]):
    """Provides a specified model for a given type and feature set."""
    # -------------------------------------------------------------------------
    # This section defines parameters specific to the LSTM model
    input_size = lstm_params[0]  # Number of features in data (X)
    hidden_size = lstm_params[2]  # Number of neurons in hidden layer
    num_layers = 1  # Number of LSTM layers in the model stack
    batch_first = True  # Determines order of dimensions for tensors
    bidirectional = False  # If True, becomes a bidirectional LSTM
    proj_size = 0  # If > 0, will use LSTM with projections
    # Define variables for LSTM input/output tensor dimensions IAW PyTorch docs
    N = BATCH_SIZE  # Batch size
    L = SEQUENCE_LENGTH  # Sequence length (no. timestamps in rolling window)
    D = int(bidirectional) + 1   # D = 1 (not birectional) or 2 (bidirectional)
    H_in = input_size
    H_cell = hidden_size
    H_out = (proj_size if proj_size > 0 else H_in)
    # Define number of dimensions in target labels (y)
    label_size = lstm_params[1]
    # End of LSTM parameter definitions.
    # -------------------------------------------------------------------------
    lstm = models.FinancialLSTM(
        input_size=input_size,
        label_size=label_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=batch_first,
        bidirectional=bidirectional,
        proj_size=proj_size
    )
    model = {
        'Ridge': None,  # Placeholder for future implementation
        'GBR': None,  # Placeholder for future implementation
        'LSTM': lstm
    }
    return model[model_type]


def test_model(model, model_type, feature_set, etf, split_df):
    """Predict ETF volatility using a model with defined feature set.
    
    Note: This function has been scaled back so that we only test the LSTM
    using PCA-extracted features, due to time and compute constraints.  A more
    complete implementation would actually test all models.
    """
    split, df = split_df
    # Assign features and label for this testing run
    X_cols = get_features(df, feature_set, etf)
    y_col = [f'{etf}_vol_target']
    # LSTM uses a different training paradigm, unlike scikit-learn models
    if model_type == 'LSTM':
        # Create Dataset and DataLoader
        dataset = dp.FinancialDataset(
            df[X_cols].values.astype(np.float32),
            df[y_col].values.astype(np.float32),
            sequence_length=SEQUENCE_LENGTH,
            target=True
        )
        dataloader = dp.DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=False,
            drop_last=False
        )
        # Record predictions and compute MAE
        mae, y_pred = models.evaluate_model(dataloader, model, LOSS_FUNCTION)
        # Convert tensors to native Python types for JSON serializability
        mae = mae.item()
        y_pred = list(np.asarray(y_pred).astype(float))
    else:
        mae, y_pred = 0, []  # Placeholder for future development
    return mae, y_pred


def results_to_df(results):
    """Converts a Python dictionary of results to a CSV file."""
    columns = [
        'ETF', 'Model', 'Features', 'Split', 'Trial', 'y_pred', 'MAE', 'R2-OOS'
    ]
    data = []
    for trial, etf_d in results.items():
        for etf, model_d in etf_d.items():
            for model, features_d in model_d.items():
                for features, split_d in features_d.items():
                    for split, result_d in split_d.items():
                        row = [
                            etf, model, features, split, trial,
                            result_d['y_pred'],
                            result_d['MAE'],
                            result_d['R2_OOS']
                        ]
                        data.append(row)
    results_df = pd.DataFrame(data, columns=columns)
    return results_df


if __name__ == '__main__':
    # Import data from CSV files and scale features to a uniform range (0, 1)
    print('---\nImporting data and establishing baseline.')
    train_df, val_df, test_df = import_data(
        TRAIN_DATA_FILE, VAL_DATA_FILE, TEST_DATA_FILE
    )
    splits = {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }
    # Establish and record baseline performance
    baseline = get_baseline(splits, ETFS)
    with open('volatility_baseline.json', 'w', encoding='utf-8') as f:
        json.dump(baseline, f)
    # Conduct a number of trials and record results
    trial_results = {}
    print(f'Testing each model on each ETF {NUM_TRIALS} times:\n---')
    for i in range(NUM_TRIALS):
        t_start = time.time()
        print(f'Conducting trial {i + 1}...')
        trial_results[i + 1] = run_trial(splits, ETFS, baseline)
        duration = (time.time() - t_start) / 60
        print(f'Trial {i + 1} took {duration} minutes.')
    # Save results as JSON and CSV files
    with open('volatility_results.json', 'w', encoding='utf-8') as f:
        json.dump(trial_results, f)
    results_df = results_to_df(trial_results)
    results_df.to_csv('volatility_results.csv', index=False)
    print('---\nTrials complete, results recorded.')
    