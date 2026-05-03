import streamlit as st
import pandas as pd

from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineering
from src.model import ProbabilityModel
from src.returns import ReturnEngine
from src.backtest import Backtester
from src.risk import RiskAnalyzer

st.set_page_config(page_title="DealEdge - M&A Arbitrage Engine", layout="wide")

st.title("DealEdge: Merger Arbitrage System")

# ---------------------------
# Load Data
# ---------------------------
@st.cache_data
def load_data():
    loader = DataLoader(
        raw_path="data/raw/mna_raw.csv",
        processed_path="data/processed/mna_processed.csv"
    )
    df = loader.load_raw()
    return df

df_raw = load_data()

# ---------------------------
# Feature Engineering
# ---------------------------
fe = FeatureEngineering()
df = fe.create_features(df_raw)

# ---------------------------
# Model
# ---------------------------
features = [col for col in df.columns if col not in [
    'deal_id','announcement_date','target_ticker','acquirer_ticker',
    'end_date','status'
]]

model = ProbabilityModel()
model.train(df, features)

df['prob_success'] = model.predict_prob(df, features)

# ---------------------------
# Returns
# ---------------------------
df = ReturnEngine.calculate_expected_return(df)

# ---------------------------
# Backtest
# ---------------------------
bt = Backtester()
results = bt.run_backtest(df)

# ---------------------------
# Risk Analyzer
# ---------------------------
risk = RiskAnalyzer(df)

# ---------------------------
# Sidebar Filters
# ---------------------------
st.sidebar.header("Filters")

min_prob = st.sidebar.slider("Min Probability", 0.0, 1.0, 0.5)
min_return = st.sidebar.slider("Min Expected Return", -0.5, 1.0, 0.0)

filtered_df = df[
    (df['prob_success'] >= min_prob) &
    (df['expected_return'] >= min_return)
]

# ---------------------------
# Tabs
# ---------------------------
tabs = st.tabs([
    "Dashboard",
    "Deal Screener",
    "Model Insights",
    "Backtest",
    "Risk Analysis"
])

# ---------------------------
# TAB 1: Dashboard
# ---------------------------
with tabs[0]:
    st.subheader("Key Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Expected Return", round(df['expected_return'].mean(), 4))
    col2.metric("Avg Probability", round(df['prob_success'].mean(), 4))
    col3.metric("Deals Count", len(df))

    st.subheader("Top Opportunities")
    st.dataframe(
        df.sort_values(by="expected_return", ascending=False)
        [['deal_id','prob_success','expected_return']]
        .head(10)
    )

# ---------------------------
# TAB 2: Deal Screener
# ---------------------------
with tabs[1]:
    st.subheader("Filtered Deals")

    st.dataframe(
        filtered_df[['deal_id','spread_pct','prob_success','expected_return']]
        .sort_values(by="expected_return", ascending=False)
    )

# ---------------------------
# TAB 3: Model Insights
# ---------------------------
with tabs[2]:
    st.subheader("Feature Importance (Proxy via Coefficients)")

    try:
        import numpy as np
        coef = model.model.coef_[0]
        importance = pd.DataFrame({
            "feature": features,
            "importance": coef
        }).sort_values(by="importance", ascending=False)

        st.bar_chart(importance.set_index("feature"))
    except:
        st.write("Model does not support coefficients")

# ---------------------------
# TAB 4: Backtest
# ---------------------------
with tabs[3]:
    st.subheader("Portfolio Performance")

    st.line_chart(results['equity_curve'])

    st.write("Average Return:", results['avg_return'])
    st.write("Volatility:", results['volatility'])

# ---------------------------
# TAB 5: Risk Analysis (20 charts in 5 groups)
# ---------------------------
with tabs[4]:
    st.subheader("Risk Analytics")

    risk_tabs = st.tabs([
        "Distribution",
        "Drawdown",
        "Tail Risk",
        "Correlation",
        "Scenario"
    ])

    # -------- Distribution --------
    with risk_tabs[0]:
        st.plotly_chart(risk.return_distribution(), use_container_width=True)
        st.plotly_chart(risk.volatility_distribution(), use_container_width=True)
        st.plotly_chart(risk.spread_distribution(), use_container_width=True)
        st.plotly_chart(risk.success_vs_failure(), use_container_width=True)

    # -------- Drawdown --------
    with risk_tabs[1]:
        st.plotly_chart(risk.equity_curve(), use_container_width=True)
        st.plotly_chart(risk.drawdown_curve(), use_container_width=True)
        st.plotly_chart(risk.rolling_volatility(), use_container_width=True)
        st.plotly_chart(risk.rolling_return(), use_container_width=True)

    # -------- Tail Risk --------
    with risk_tabs[2]:
        st.plotly_chart(risk.var_95(), use_container_width=True)
        st.plotly_chart(risk.cvar(), use_container_width=True)
        st.plotly_chart(risk.worst_losses(), use_container_width=True)
        st.plotly_chart(risk.best_gains(), use_container_width=True)

    # -------- Correlation --------
    with risk_tabs[3]:
        st.plotly_chart(risk.correlation_matrix(), use_container_width=True)
        st.plotly_chart(risk.scatter_spread_return(), use_container_width=True)
        st.plotly_chart(risk.scatter_time_return(), use_container_width=True)
        st.plotly_chart(risk.deal_value_vs_return(), use_container_width=True)

    # -------- Scenario --------
    with risk_tabs[4]:
        st.plotly_chart(risk.stress_down(), use_container_width=True)
        st.plotly_chart(risk.stress_up(), use_container_width=True)
        st.plotly_chart(risk.failure_impact(), use_container_width=True)
        st.plotly_chart(risk.success_impact(), use_container_width=True)
        st.plotly_chart(risk.allocation_pie(), use_container_width=True)
