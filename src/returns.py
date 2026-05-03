import numpy as np

class ReturnEngine:

    @staticmethod
    def calculate_expected_return(df):
        df = df.copy()

        df['break_price'] = df['current_price'] * 0.8

        df['expected_return'] = (
            df['prob_success'] * ((df['offer_price'] - df['current_price']) / df['current_price']) +
            (1 - df['prob_success']) * ((df['break_price'] - df['current_price']) / df['current_price'])
        )

        return df
