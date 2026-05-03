import plotly.express as px
import plotly.graph_objects as go
import numpy as np

class RiskAnalyzer:

    def __init__(self, df):
        self.df = df

    # ---------------- TAB 1: Distribution ----------------
    def return_distribution(self):
        return px.histogram(self.df, x="expected_return", title="Return Distribution")

    def volatility_distribution(self):
        return px.histogram(self.df, x="time_to_close", title="Time to Close Distribution")

    def spread_distribution(self):
        return px.histogram(self.df, x="spread_pct", title="Spread Distribution")

    def success_vs_failure(self):
        return px.histogram(self.df, x="status", title="Success vs Failure")

    # ---------------- TAB 2: Drawdown ----------------
    def equity_curve(self):
        returns = self.df['expected_return']
        equity = (1 + returns).cumprod()
        return px.line(y=equity, title="Equity Curve")

    def drawdown_curve(self):
        returns = self.df['expected_return']
        equity = (1 + returns).cumprod()
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        return px.line(y=drawdown, title="Drawdown Curve")

    def rolling_volatility(self):
        vol = self.df['expected_return'].rolling(10).std()
        return px.line(y=vol, title="Rolling Volatility")

    def rolling_return(self):
        ret = self.df['expected_return'].rolling(10).mean()
        return px.line(y=ret, title="Rolling Return")

    # ---------------- TAB 3: Tail Risk ----------------
    def var_95(self):
        var = np.percentile(self.df['expected_return'], 5)
        fig = go.Figure()
        fig.add_vline(x=var)
        fig.update_layout(title="VaR 95%")
        return fig

    def cvar(self):
        var = np.percentile(self.df['expected_return'], 5)
        cvar = self.df[self.df['expected_return'] <= var]['expected_return'].mean()
        fig = go.Figure()
        fig.add_vline(x=cvar)
        fig.update_layout(title="CVaR")
        return fig

    def worst_losses(self):
        return px.bar(self.df.nsmallest(10, 'expected_return'),
                      x='deal_id', y='expected_return',
                      title="Worst Deals")

    def best_gains(self):
        return px.bar(self.df.nlargest(10, 'expected_return'),
                      x='deal_id', y='expected_return',
                      title="Best Deals")

    # ---------------- TAB 4: Correlation ----------------
    def correlation_matrix(self):
        corr = self.df[['spread_pct','time_to_close','expected_return']].corr()
        return px.imshow(corr, title="Correlation Matrix")

    def scatter_spread_return(self):
        return px.scatter(self.df, x='spread_pct', y='expected_return',
                          title="Spread vs Return")

    def scatter_time_return(self):
        return px.scatter(self.df, x='time_to_close', y='expected_return',
                          title="Time vs Return")

    def deal_value_vs_return(self):
        return px.scatter(self.df, x='deal_value', y='expected_return',
                          title="Deal Value vs Return")

    # ---------------- TAB 5: Scenario ----------------
    def stress_down(self):
        stressed = self.df['expected_return'] - 0.05
        return px.histogram(stressed, title="Stress Scenario Down")

    def stress_up(self):
        stressed = self.df['expected_return'] + 0.05
        return px.histogram(stressed, title="Stress Scenario Up")

    def failure_impact(self):
        fail = self.df[self.df['status'] == 0]
        return px.histogram(fail, x='expected_return', title="Failure Impact")

    def success_impact(self):
        success = self.df[self.df['status'] == 1]
        return px.histogram(success, x='expected_return', title="Success Impact")

    def allocation_pie(self):
        return px.pie(self.df, names='status', title="Deal Outcome Allocation")
