import pandas as pd
import numpy as np

class Backtester:

    def run_backtest(self, df, top_n=10):
        df_sorted = df.sort_values(by='expected_return', ascending=False)

        portfolio = df_sorted.head(top_n)

        portfolio['return'] = portfolio['expected_return']

        equity_curve = (1 + portfolio['return']).cumprod()

        return {
            "portfolio": portfolio,
            "equity_curve": equity_curve,
            "avg_return": portfolio['return'].mean(),
            "volatility": portfolio['return'].std()
        }
