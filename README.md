# Time Series Forecasting and Portfolio Optimization

![GitHub](https://img.shields.io/badge/Python-3.8%2B-blue)
![GitHub](https://img.shields.io/badge/License-MIT-green)
![GitHub](https://img.shields.io/badge/Status-Completed-brightgreen)

## 📑 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)
- [Contact](#-contact)

---

## 📌 Overview
This project focuses on leveraging **time series forecasting** and **portfolio optimization** to enhance investment strategies for **Guide Me in Finance (GMF) Investments**. Using historical financial data for **Tesla (TSLA)**, **Vanguard Total Bond Market ETF (BND)**, and **S&P 500 ETF (SPY)**, the project aims to:
- Predict future stock prices using advanced forecasting models like **ARIMA**, **SARIMA**, and **LSTM**.
- Optimize portfolio allocation to maximize returns while minimizing risks.
- Provide actionable insights for financial analysts to make data-driven investment decisions.

---

## 🚀 Key Features
- **Data Preprocessing**: Clean and prepare historical financial data for analysis.
- **Exploratory Data Analysis (EDA)**: Visualize trends, volatility, and key metrics like Value at Risk (VaR) and Sharpe Ratio.
- **Time Series Forecasting**: Implement ARIMA, SARIMA, and LSTM models to predict future stock prices.
- **Portfolio Optimization**: Optimize asset allocation using covariance matrices and risk-return analysis.
- **Visualizations**: Generate interactive charts and graphs for forecasts and portfolio performance.

---

## 📂 Repository Structure
```bash
  Directory structure:
└── dagiteferi-timeseries-portfolio-optimization/
    ├── README.md
    ├── requirements.txt
    ├── logs/
    ├── model/
    │   └── optimized_lstm.keras
    ├── notebooks/
    │   ├── README.md
    │   ├── Future_Forecasting.ipynb
    │   ├── Preprocess_Explore_Data.ipynb
    │   ├── __init__.py
    │   ├── arima_forecasting.ipynb
    │   ├── lstm.ipynb
    │   ├── portfolio_optimization.ipynb
    │   └── sarima_forecasting.ipynb
    ├── scripts/
    │   ├── README.md
    │   ├── __init__.py
    │   ├── arima.py
    │   ├── data_cleaning.py
    │   ├── eda.py
    │   ├── future_forecast.py
    │   ├── lstm.py
    │   ├── portfolio_optimization.py
    │   ├── sarima.py
    │   ├── seasonal_decompose.py
    │   ├── volatility_analysis.py
    │   └── __pycache__/
    ├── src/
    │   ├── __init__.py
    │   ├── data_loader.py
    │   ├── fetch_data.py
    │   ├── file_structure.py
    │   └── __pycache__/
    ├── tests/
    │   └── __init__.py
    └── .github/
        └── workflows/
            └── unittests.yml

   ```
---

## 🛠️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/dagiteferi/TimeSeries-Portfolio-Optimization.git
   ```
2.Navigate to the project directory:

```bash
  cd TimeSeries-Portfolio-Optimization
   ```
3.Install the required dependencies:
```bash
 pip install -r requirements.txt
   ```
🧑‍💻 Usage
1. Data Preprocessing:

Run notebooks/01_Data_Preprocessing.ipynb to clean and prepare the data.

2. Exploratory Data Analysis:

Use notebooks/02_EDA.ipynb to visualize trends and analyze volatility.

3. Time Series Forecasting:

Train and evaluate models in notebooks/03_TimeSeries_Forecasting.ipynb.

4. Portfolio Optimization:

Optimize asset allocation in notebooks/04_Portfolio_Optimization.ipynb.

📊 Results
Forecasted stock prices with confidence intervals.

Optimized portfolio allocations for maximum Sharpe Ratio.

Visualizations of portfolio performance and risk-return analysis.

🙏 Acknowledgments
Data sourced from YFinance.

Inspired by portfolio optimization techniques from PyPortfolioOpt.

📧 Contact
For questions or feedback, feel free to reach out:
📩 dagiteferi2011@gmail.com
🌐(https://dagmawipro.netlify.app/)
