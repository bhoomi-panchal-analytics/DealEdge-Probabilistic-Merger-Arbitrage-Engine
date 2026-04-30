import pandas as pd
import numpy as np

class FeatureEngineering:

    @staticmethod
    def create_features(df):
        df = df.copy()

        df['time_to_close'] = (df['end_date'] - df['announcement_date']).dt.days
        df['time_to_close'] = df['time_to_close'].fillna(df['time_to_close'].median())

        df['spread_pct'] = (df['offer_price'] - df['current_price']) / df['current_price']

        df['log_deal_value'] = np.log1p(df['deal_value'])

        df = pd.get_dummies(df, columns=['deal_type'], drop_first=True)

        df = df.fillna(0)

        return df
