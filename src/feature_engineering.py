import pandas as pd
import numpy as np

def add_sentiment_lag_features(merged_df, lag_days=[1,2,3]):
    """
    Adds lagged sentiment values (value) and classification shifts.
    merged_df: must contain 'Date' (datetime.date) and 'value'
    """
    df = merged_df.copy()
    df = df.sort_values('Date')
    # convert Date to datetime if needed
    if not isinstance(df['Date'].dtype, pd.core.dtypes.dtypes.DatetimeTZDtype) and not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
    for ld in lag_days:
        df[f'value_lag_{ld}'] = df['value'].shift(ld)
    # rolling mean sentiment
    df['value_rolling_3'] = df['value'].rolling(window=3, min_periods=1).mean()
    return df

def encode_classification(df, col='classification'):
    """
    Map common classification strings to numeric codes.
    Returns copy with new column 'classification_code'
    """
    mapping = {
        'Extreme Fear': 0,
        'Fear': 25,
        'Neutral': 50,
        'Greed': 75,
        'Extreme Greed': 100,
        'Unknown': np.nan
    }
    out = df.copy()
    out['classification_code'] = out[col].map(mapping)
    return out
