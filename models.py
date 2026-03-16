#123456789012345678901234567890123456789012345678901234567890123456789012345678
"""This module defines the models we will use in our analysis.

We selected several models appropriate to time series financial data:
 - Long Short-Term Memory (LSTM, PyTorch)
 - Generalized AutoRegressive Conditional Heteroscedasticity (GARCH, PyFlux)
 - AutoRegressive Integrated Moving Average eXogenous (ARIMAX, PyFlux)
"""

import pandas as pd
import torch
from torch import nn
from tqdm import tqdm


class FinancialLSTM(nn.Module):
    """Multivariate LSTM model to predict annualized return and volatility."""

    def __init__(
        self,
        input_size,
        label_size,
        hidden_size,
        num_layers=1,
        batch_first=True,
        bidirectional=False,
        proj_size=0,
    ):
        """Instantiate FinancialLSTM model.
        
        Keyword Arguments:
          label_size -- number of features in each label (y)
        """
        super(FinancialLSTM, self).__init__()
        # LSTM layer parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.proj_size = proj_size
        self.H_out = (proj_size if proj_size > 0 else hidden_size)
        # Fully connected (output) layer parameter
        self.label_size=label_size
        # Model layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
            proj_size=proj_size
        )
        self.fc = nn.Linear(hidden_size, label_size)

    def forward(self, x):
        """Make a prediction based on input x."""
        # Final hidden state h_n has shape=(D * num_layers, N, H_out)
        output, (h_n, c_n) = self.lstm(x)
        # Change the shape of h_n's activation to (N, H_out)
        out = h_n.view(-1, self.hidden_size)
        pred = self.fc(out)
        return pred

    
def train_model(dataloader, model, loss_fn, optimizer, batch_first=True):
    """Generic training steps for any PyTorch deep learning model.
    
    Copied nearly verbatim from the PyTorch tutorial.
    """
    model.train()
    for X, y in dataloader:
        # Compute prediction and loss; permute input (X) if necessary
        if batch_first == True:
            pred = model(X)
        else:
            pred = model(X.permute(1, 0, 2))
        loss = loss_fn(pred, y)
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def evaluate_model(dataloader, model, loss_fn, batch_first=True):
    """Generic evaluation steps for any PyTorch deep learning model.

    Copied nearly verbatim from the PyTorch tutorial.
    """
    model.eval()
    preds = []
    # No need to compute gradients during evaluation
    with torch.no_grad():
        for X, y in dataloader:
            # Record loss history; permute input (X) if necessary
            if batch_first == True:
                pred = model(X)
            else:
                pred = model(X.permute(1, 0, 2))
            loss = loss_fn(pred, y)
            preds.append(pred)
    return preds, loss
