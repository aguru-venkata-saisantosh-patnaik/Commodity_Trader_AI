# Commodity_Trader_AI

## Overview

CommodityTraderAI is a comprehensive project aimed at analyzing, predicting, and optimizing trading strategies for various commodities. It combines technical analysis, sentiment analysis, and machine learning to create actionable trading signals and assess their effectiveness. By integrating data-driven techniques, the project offers a robust framework for informed commodity trading.

---

## Project Details

The project is structured as follows:

### 1. **Data Fetching and Preprocessing**
   - Historical price data is fetched for multiple commodities (e.g., crude oil, gold, silver, natural gas) using the `yfinance` API.
   - Data is cleaned and preprocessed to remove missing values, ensure proper formatting, and create a consistent dataset for analysis.

### 2. **Exploratory Data Analysis (EDA)**
   - Commodity trends are visualized to understand market dynamics.
   - Correlation analysis is performed to explore relationships between different commodities.
   - Key technical indicators like Moving Averages, Bollinger Bands, RSI, and MACD are computed and analyzed.

### 3. **Feature Engineering**
   - Advanced features such as daily returns, moving averages, exponential moving averages, Bollinger Bands, RSI, and MACD are generated to enhance predictive capabilities.

### 4. **Sentiment Analysis**
   - News sentiment for each commodity is analyzed using the VADER Sentiment Analyzer and News API.
   - Aggregated sentiment scores are incorporated into the decision-making process, providing a holistic view of market trends.

### 5. **Backtesting Strategies**
   - Multiple trading strategies are implemented and evaluated:
     - **Momentum Strategy**: Trades based on price momentum.
     - **Range Trading Strategy**: Buys at support levels and sells at resistance levels.
     - **Breakout Strategy**: Captures trends during price breakouts.
     - **Mean Reversion Strategy**: Exploits deviations from the average price.
     - **Seasonal Strategy**: Trades based on seasonal price patterns.
   - Strategies are evaluated using metrics like cumulative returns and CAGR.

### 6. **Machine Learning Model (XGBoost)**
   - An XGBoost regressor is used to predict future commodity prices.
   - The model is fine-tuned using grid search for optimal hyperparameters.
   - Backtesting on hidden data evaluates the predictive performance of the model.

### 7. **Signal Generation and Execution**
   - Buy, sell, and hold signals are generated based on the predictions.
   - Trading is simulated on historical data, and cumulative profit and loss (PnL) is calculated.

### 8. **Output and Results**
   - Comprehensive outputs include:
     - Cleaned datasets for each commodity.
     - Sentiment analysis results.
     - Backtesting results for each strategy.
     - XGBoost predictions and trading signals.
     - Profit and loss data visualizations.
   - Final trading results are consolidated in a summary CSV file.

---

## Key Features

1. **Data Fetching and Preprocessing**
   - Fetches historical commodity prices from Yahoo Finance using the `yfinance` library.
   - Cleans and preprocesses the data, including handling missing values and formatting date columns.

2. **Exploratory Data Analysis (EDA)**
   - Visualizes trends and relationships in commodity prices.
   - Analyzes technical indicators like Moving Averages, Bollinger Bands, RSI, and MACD.
   - Correlation analysis among different commodities.

3. **Feature Engineering**
   - Generates features such as daily returns, moving averages, exponential moving averages, Bollinger Bands, RSI, and MACD to enhance predictive modeling.

4. **Sentiment Analysis**
   - Analyzes news sentiment for each commodity using the VADER sentiment analyzer and News API.
   - Aggregates sentiment data to complement technical indicators for trading decisions.

5. **Backtesting Strategies**
   - Implements multiple trading strategies:
     - Momentum Strategy
     - Range Trading Strategy
     - Breakout Strategy
     - Mean Reversion Strategy
     - Seasonal Strategy
   - Calculates cumulative returns and Compound Annual Growth Rate (CAGR) for each strategy.

6. **Machine Learning Model (XGBoost)**
   - Predicts future commodity prices using an XGBoost regressor.
   - Fine-tunes the model using grid search to optimize hyperparameters.
   - Backtests predictions on hidden data to evaluate performance.

7. **Signal Generation and Execution**
   - Generates buy/sell/hold signals based on predictions.
   - Executes trades and calculates cumulative profit and loss (PnL).

---

## Repository Structure

```
├── commodity_data/          # Contains raw, cleaned, and feature-enhanced CSV files
├── sentiment_data/          # Contains cleaned sentiment data and sentiment analysis results
├── signals_*.csv            # Trading signals for each commodity
├── xgboost_predictions_*.csv  # XGBoost predictions for test datasets
├── finetuned_xgboost_predictions_*.csv  # Fine-tuned XGBoost predictions
├── signals_with_pnl_*.csv   # Signals with profit and loss data
├── final_trading_results.csv  # Summary of final cumulative returns for all commodities
├── notebook.ipynb           # Main Python notebook with the full pipeline
├── README.md                # This README file
```

---

## Requirements

### Libraries and Frameworks
The following Python libraries are required to run the notebook:

- `yfinance`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `vaderSentiment`
- `xgboost`
- `scikit-learn`

### Installation
Run the following commands to install the required libraries:

```bash
pip install yfinance pandas numpy matplotlib seaborn vaderSentiment xgboost scikit-learn
```

---

## Usage

### 1. Clone the Repository

```bash
git clone https://github.com/aguru-venkata-saisantosh-patnaik/Commodity_Trader_AI.git
cd Commodity_Trader_AI
```

### 2. Fetch Historical Data
Run the notebook to fetch historical commodity prices and save them to the `commodity_data/` directory.

### 3. Preprocess and Analyze Data
- Preprocess data to clean and format it for analysis.
- Perform EDA to visualize trends and compute technical indicators.

### 4. Sentiment Analysis
- Fetch and analyze news sentiment for commodities using the `News API`.
- Save sentiment data to the `sentiment_data/` directory.

### 5. Backtest Strategies
- Evaluate various trading strategies and calculate their cumulative returns and CAGR.
- Identify the best strategy for each commodity.

### 6. Train XGBoost Model
- Train and fine-tune XGBoost models for price prediction.
- Save predictions and signals for each commodity.

### 7. Execute Trades
- Execute trades on hidden data using generated signals.
- Save final profit and loss results for each commodity.

### 8. Visualize and Analyze Results
- Visualize PnL charts and analyze performance metrics for each commodity.
- Review final results in `final_trading_results.csv`.

---

## Outputs

- **Cleaned Data**: Preprocessed commodity data files saved in `commodity_data/`.
- **Sentiment Analysis Results**: Sentiment scores and aggregated results in `sentiment_data/`.
- **Trading Strategies**: CSV files summarizing the performance of different strategies for each commodity.
- **Machine Learning Predictions**: XGBoost prediction outputs and corresponding signals for test datasets.
- **Profit and Loss Analysis**: Cumulative PnL data and visualizations for all commodities.
- **Final Trading Results**: Consolidated results showcasing the best strategy and its performance for each commodity, saved in `final_trading_results.csv`.

---

## Results
- **Final Trading Results**: Detailed in `final_trading_results.csv`.
- **Performance Metrics**: Includes MAE, RMSE, cumulative returns, and CAGR for each commodity and strategy.
- **PnL Visualization**: Charts illustrating the cumulative profit and loss for each commodity.

---

## Contribution
Contributions are welcome! Feel free to open issues or submit pull requests to improve the repository.

---

## License
This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments
- Yahoo Finance for historical data.
- News API for sentiment analysis.
- XGBoost library for machine learning.
- VADER Sentiment Analyzer for text sentiment analysis.

---

## Contact
For queries, please reach out to [agurusantosh@gmail.com](mailto:agurusantosh@gmail.com).

