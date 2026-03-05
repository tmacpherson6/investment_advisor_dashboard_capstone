# Investment Portfolio Generator - MADS Capstone Project

## Overview
*The Portfolio Generator forms an integrated decision framework that connects heterogeneous investor characteristics to a data-driven investment strategy.*

## Features
- Investor Risk Tolerance Estimate: We use machine learning model to provide an estimate of investor risk tolerance, based on a brief questionnaire.
- Market Sector Forecast: We use a deep learning model (LSTM) to forecast six-month future return and volatility for selected market sectors.
- Portfolio Generation: We combine the outputs (above) to construct an optimal investment portfolio that maximizes returns while respecting an investor's risk tolerance.

## Installation

### Prerequisites
- Python 3.11 or higher
- pip

### Setup
1. Clone the repository:
```bash
   git clone https://github.com/tmacpherson6/investment_advisor_dashboard_capstone capstone
```

2. Create and activate virtual environment:
```bash
   python -m venv capstone
   source capstone/bin/activate  # Or on Windows: capstone\Scripts\activate
```

3. Install dependencies:
```bash
   pip install -r requirements.txt
```

## Usage

### Running the App Locally

1. Make sure your virtual environment is activated (see Setup above).

2. Navigate to the project folder:
```bash
cd path/to/capstone
```

3. Run the Flask app:
```bash
python app.py
```

4. Once the server starts, you'll see output in the terminal similar to:
```
 * Running on http://127.0.0.1:5000
```
Open that URL in your browser to use the dashboard. To stop the server, press `Ctrl+C` in the terminal.

> **Note:** The app is currently configured for local development only (`debug=True`). Do not expose it to the public internet in this state.

## Project Structure
```
project/
├── src/
├── resources/
├── notebooks/
├── documentation/
├── requirements.txt
└── README.md
```

## Configuration
Details about environment variables, API keys, etc.

## Technologies Used
- Python
- yfinance
- FRED® API
- NumPy
- Pandas
- Matplotlib
- Altair
- PyTorch
- (add others as we build)

## License
[We need to choose a license]