

# DealEdge: Probabilistic Merger Arbitrage Engine

## Overview

DealEdge is a quantitative event-driven trading system that models merger arbitrage opportunities using machine learning and risk-adjusted return estimation. The system evaluates M&A deals by estimating completion probabilities and computing expected returns under uncertainty.

---

## Problem Statement

Merger arbitrage opportunities arise due to uncertainty in deal completion. Markets price this uncertainty through spreads, but mispricing can occur due to:

* Regulatory risks
* Time delays
* Information asymmetry

This project aims to identify profitable arbitrage opportunities by combining probability modeling with financial return estimation.

---

## Methodology

### 1. Data Pipeline

* M&A deal dataset (synthetic + extendable to real data)
* Feature engineering:

  * Spread (%)
  * Time to close
  * Deal size (log)
  * Deal type encoding

### 2. Probability Model

* Logistic Regression
* Predicts likelihood of deal completion

### 3. Return Model

Expected return calculated using probability-weighted payoff:

* Deal success payoff
* Deal failure (break price) loss

### 4. Backtesting Engine

* Portfolio construction (top N deals)
* Performance metrics:

  * Average return
  * Volatility
  * Equity curve

### 5. Risk Analysis

Includes 20+ charts:

* Distribution risk
* Drawdown analysis
* Tail risk (VaR, CVaR)
* Correlation analysis
* Stress scenarios

---

## Key Features

* End-to-end quant pipeline
* Machine learning-based deal probability estimation
* Portfolio-level backtesting
* Interactive dashboard (Streamlit)
* Risk analytics with scenario simulation

---

## Results (Example)

* Positive expected return portfolio
* Diversified deal exposure
* Controlled drawdowns (based on model assumptions)

---

## Limitations

* Synthetic dataset (initial version)
* Simplified break price estimation
* No transaction cost modeling
* Limited real-time data integration

---

## Tech Stack

* Python (Pandas, NumPy)
* Scikit-learn
* Plotly
* Streamlit

---

## How to Run

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

---

## Future Improvements

* Integrate real M&A datasets (SEC EDGAR)
* Add transaction cost modeling
* Improve break price estimation
* Use advanced ML models (XGBoost)
* Incorporate news sentiment (NLP)

---

## Author

Quant-focused candidate building event-driven trading systems and risk models.
