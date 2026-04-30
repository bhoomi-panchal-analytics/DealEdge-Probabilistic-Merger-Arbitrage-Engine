import pandas as pd

class DataLoader:
    def __init__(self, raw_path, processed_path):
        self.raw_path = raw_path
        self.processed_path = processed_path

    def load_raw(self):
        df = pd.read_csv(self.raw_path)
        df['announcement_date'] = pd.to_datetime(df['announcement_date'])
        df['end_date'] = pd.to_datetime(df['end_date'])
        return df

    def load_processed(self):
        df = pd.read_csv(self.processed_path)
        return df

    def validate_data(self, df):
        assert not df.empty, "Dataset is empty"
        assert 'offer_price' in df.columns, "Missing critical column"
        return True
