#123456789012345678901234567890123456789012345678901234567890123456789012345678
"""This module contains helper functions to prepare data for analysis."""

import math
import re

from fredapi import Fred
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BusinessDay
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader


# -----------------------------------------------------------------------------
# Data Retrieval and Loading Functions:
# A. get_fred_data -- retrieves data from Federal Reserve Economic Data website
# B. make_pca_df -- creates a DataFrame combining PCA and original variables
# -----------------------------------------------------------------------------


# Copied from Tom's notebook (slightly modified)
def get_fred_data(fred):
    """Fetch the latest data from FRED."""
    try:
        fred_data = {
            "nominal_GDP" : fred.get_series("GDP"),  
            "real_GDP" : fred.get_series("GDPC1"),
            "debt_to_GDP" : fred.get_series("GFDEGDQ188S"),
            "debt_interest" : fred.get_series("A091RC1Q027SBEA"),
            "consumer_price_index" : fred.get_series("CPIAUCSL"),
            "core_PCI" : fred.get_series("CPILFESL"),
            "personal_consumption_expenditure" : fred.get_series("PCEPI"),
            "core_PCE" : fred.get_series("PCEPILFE"),
            "producer_price_index" : fred.get_series("PPIFIS"),
            "unemployment_rate" : fred.get_series("UNRATE"),
            "initial_jobless_claims" : fred.get_series("ICSA"),
            "continued_jobless_claims" : fred.get_series("CCSA"),
            "teenager_unemployment_rate" : fred.get_series("LNS14000012"),
            "adult_unemployment_rate" : fred.get_series("LNS14000025"),
            "male_unemployment_rate" : fred.get_series("LNS14000001"),
            "female_unemployment_rate" : fred.get_series("LNS14000002"),
            "average_duration_of_unemployment" : fred.get_series("UEMPMEAN"),
            "one_month_yield" : fred.get_series("DGS1MO"),
            "three_month_yield" : fred.get_series("DGS3MO"),
            "six_month_yield" : fred.get_series("DGS6MO"),
            "one_year_yield" : fred.get_series("DGS1"),
            "two_year_yield" : fred.get_series("DGS2"),
            "five_year_yield" : fred.get_series("DGS5"),
            "ten_year_yield" : fred.get_series("DGS10"),
            "thirty_year_yield" : fred.get_series("DGS30"),
        }
        if not fred_data["unemployment_rate"].empty:
            return fred_data
    except Exception as e:
        print(f"Unemployment Rate FRED failed: {e}")


def make_pca_df(base_data_file, pca_data_file, ticker, returns=False):
    """Combines PCA data with base data for analysis of a single ETF."""
    base_df = pd.read_csv(
        base_data_file,
        index_col='date',
        parse_dates=True
    )
    pca_df = pd.read_csv(
        pca_data_file,
        index_col='date',
        parse_dates=True
    )
    # Combine PCA data with base data. From base data, select only historical
    # volatility and a future volatility target by default.
    target = [ticker + '_vol_target']
    base_cols = [ticker + '_vol-h']
    if returns:
        # Include past near-term return for asset if desired
        base_cols = [ticker + '_ret-h'] + base_cols
    df = pd.concat([base_df[base_cols + target], pca_df], axis=1)
    return df


# -----------------------------------------------------------------------------
# General Purpose Helper Functions:
#   A. view_nan -- displays data as an image to highlight missing values
#   B. get_start_dates -- gets the starting date for a feature
#   C. time_series_split -- custom splitting function; preserves chron. order
#   D. remove_series -- drops an ETF and its features/labels from a DataFrame
#   E. get_X_y_cols -- separates data columns from labels (targets)
# -----------------------------------------------------------------------------


def view_nan(df):
    """Provides an image of a DataFrame showing NaN values as white boxes.
    
    Credit to Michael P. Notter for demonstrating this technique.
    https://miykael.github.io/blog/2022/advanced_eda/
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.subplots()
    if 'date' in df.columns:
        df = df.set_index('date')
        ax.set_ylabel('Timestamp')
    ax.set_xlabel('Column number')
    ax.imshow(df.isna(), aspect="auto", interpolation="nearest", cmap="gray")
    return fig


def get_start_dates(df):
    """Retrieves the starting date for each column of time-series data."""
    start_date = {}
    for column in df.columns:
        start_date[column] = df[column].dropna().index[0]
    return sorted([(k, v) for k, v in start_date.items()], key=lambda i:i[1])


def time_series_split(df, start_date, buffer_size, date_col='date'):
    """Splits DataFrame into train/test splits with a buffer in between.

    Keyword arguments:
      df -- Pandas DataFrame
      start_date -- str for start date of test/validation split: 'YYYY-MM-DD'
      buffer_size -- number of business days between splits
    """
    start_date = pd.to_datetime(start_date)
    buffer = BusinessDay(buffer_size)
    # Sometimes we move the datetime index to a column
    if date_col in df.columns:
        test_df = df[df[date_col] >= start_date]
        train_df = df[df[date_col] < start_date - buffer]
    # Otherwise, assume the index has the timestamp
    else:
        test_df = df[df.index >= start_date]
        train_df = df[df.index < start_date - buffer]
    return train_df.copy(), test_df.copy()


def remove_series(df, tickers_to_remove=None):
    """Removes data and labels for a financial asset from a DataFrame.
    
    Keyword arguments:
      df -- Pandas DataFrame
      tickers_to_remove -- list of ticker symbols to be removed from df
    Return:
      df -- DataFrame with all columns containing listed tickers removed
    """
    assert type(df) == type(pd.DataFrame()), 'df must be a DataFrame'
    if tickers_to_remove is not None:
        for ticker in tickers_to_remove:
            p = re.compile ('^' + ticker)
            for column_name in df.columns:
                if p.match(column_name) is None:
                    continue
                else:
                    df = df.drop(columns=[column_name]).copy()
    return df


def get_X_y_cols(df, label_marker='target', returns=False):
    """Creates separate lists of column names for data features and labels.

    Keyword arguments:
      df -- Pandas DataFrame containing labelled data
      label_marker -- str that is part of a column name for a label
    Return:
      (feature_columns, label_columns) -- tuple of lists of column names
    """
    assert type(df) == type(pd.DataFrame()), 'df must be a DataFrame'
    feature_columns, label_columns = [], []
    for column_name in df.columns:
        if label_marker in column_name:
            label_columns.append(column_name)
        else:
            if (label_marker not in column_name) and (column_name != 'date'):
                # We may want to exclude return data for other assets
                if returns:
                    feature_columns.append(column_name)
                else:
                    if '_ret' not in column_name:
                        feature_columns.append(column_name)
    return (feature_columns, label_columns)


# -----------------------------------------------------------------------------
# Manual Feature Engineering Functions:
#   A. log_return -- computes log return for an asset over a period of time
#   B. volatility -- computes historic volatility for an asset
# -----------------------------------------------------------------------------


def log_return(array):
    """Computes log_return from an array of sequential price values."""
    return np.log(array[-1] / array[0])


def volatility(array):
    """Computes volatility from an array of sequential return values."""
    # We use the zero-mean assumption for expected return, E[R] = 0
    # historic volatility = sqrt(1/n * sum(return^2))
    return np.sqrt(np.sum(array**2) / len(array))
    

# -----------------------------------------------------------------------------
# Datasets:
#   A. UnivariateTimeSeriesDataset -- a basic dataset for time series data
#   B. FinancialDataset -- a custom dataset for predicting asset volatility
# -----------------------------------------------------------------------------


class UnivariateTimeSeriesDataset(Dataset):
    """Dataset for univariate time series data.
    
    This dataset implements a backward-looking rolling window to generate
    feature vectors consisting of successive values in a time series.
    In this implementation, the label for each feature vector is the value of
    the sample (sequence) at the next time step.
    """
    
    def __init__(self, data, window_size):
        """Create a Dataset from time series data (x = f(t)).

        Keyword arguments:
          data -- Numpy Array containing the values of the series
          window_size -- number of timestamps in the rolling window
        """
        self.data = torch.from_numpy(data.copy())
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        """Generate a univariate feature vector (x_i) and its label (y_i).
        
        The feature vector (x_i) is a sequence of successive values for the
        time series, generated by a rolling window.  The label is simply the
        value of the time series at the next timestamp after the window.
        """
        x_i = self.data[idx : idx + self.window_size]
        y_i = self.data[idx + self.window_size]
        # Reshape values as required for PyTorch DataLoader implementation
        return x_i.reshape(-1, 1), y_i.reshape(1)


class FinancialDataset(Dataset):
    """Dataset for time series financial data."""
    
    def __init__(self, data, labels, sequence_length=21):
        """Create a Dataset from time series data (X) and labels (y).

        Keyword arguments:
          data -- NumPy array of shape = (num_timestamps, feature_dim)
          labels -- NumPy array of shape = (num_timestamps, label_dim)
          sequence_length -- length of each input sequence
        """
        assert len(data) == len(labels),\
            'data (X) and labels (y) must be same length'
        self.data = torch.from_numpy(data.copy())
        self.labels = torch.from_numpy(labels.copy())
        self.sequence_length = sequence_length
        self.X_dim = data.shape[-1]  # Number of features in data sample
        self.y_dim = labels.shape[-1]  # Number of features in label

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        """Retreive a sample sequence (X_i) and its label (y_i).
        
        We rely on a financial theory where historical observations are useful
        in predicting future values of return and volatility.
         - Each data sample is a sequence from a vector-valued time series;
           thus, each sample is a matrix (X_i) where each row is an
           observation vector at a given timestamp of dimension (X_dim).
         - Each label (y_i) is a scalar value representing the target, either
           return or volatility over a future outlook defined by a number of
           business days.
        ------------------
        Keyword arguments:
          idx -- starting index for the data sample
        Returns:
          X_i -- PyTorch tensor of shape (sequence_length, X_dim)
          y_i -- PyTorch tensor of shape (1)
        """
        # We sample a sequence of data
        X_i = self.data[idx : idx + self.sequence_length]
        # For L=1, label for sample at idx=0 is located at idx=0
        y_i = self.labels[idx + self.sequence_length - 1]
        return X_i, y_i
        