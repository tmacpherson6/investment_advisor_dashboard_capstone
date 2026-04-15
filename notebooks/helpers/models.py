#123456789012345678901234567890123456789012345678901234567890123456789012345678
"""This module defines the deep-learning models we use in our analyses.

Author: Pete King
"""

import pandas as pd
import torch
from torch import nn
from tqdm import tqdm


class FinancialLSTM(nn.Module):
    """LSTM model to predict annualized return and volatility."""

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
        self.D = int(bidirectional) + 1
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
        self.fc = nn.Linear(self.D * self.H_out, label_size)

    def forward(self, x):
        """Make a prediction based on input x."""
        output, (h_n, c_n) = self.lstm(x)
        # Final hidden state h_n has shape=(D * num_layers, N, H_out), so we
        # change the shape of h_n's activation to (N, H_out) for FC layer.
        out = h_n[-1, :, :].view(-1, self.H_out)
        pred = self.fc(out)
        return pred

    
def train_model(dataloader, model, loss_fn, optimizer):
    """Generic training steps for any PyTorch deep learning model.
    
    Modeled after the example from the PyTorch tutorial.
    """
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        optimizer.zero_grad()
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        loss.backward()
        optimizer.step()
    return loss


def evaluate_model(dataloader, model, loss_fn):
    """Generic evaluation steps for any PyTorch deep learning model.

    Modeled after the example from the PyTorch tutorial.
    """
    model.eval()
    preds = []
    # No need to compute gradients during evaluation
    with torch.no_grad():
        for X, y in dataloader:
            # Record loss history
            pred = model(X)
            loss = loss_fn(pred, y)
            preds = preds + list(pred.flatten())
    return loss, torch.tensor(preds)
