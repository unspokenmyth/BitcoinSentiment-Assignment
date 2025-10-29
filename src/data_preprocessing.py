import pandas as pd

def load_and_clean_trader_data(file_path):
    df = pd.read_csv(file_path)

    # Standardize column names (lowercase + underscores)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Ensure timestamp/date handling
    if "timestamp_ist" in df.columns:
        df["timestamp_ist"] = pd.to_datetime(df["timestamp_ist"], errors="coerce")
        df["date"] = df["timestamp_ist"].dt.date
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["date"] = df["timestamp"].dt.date

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Drop completely empty or irrelevant rows
    df.dropna(subset=["execution_price", "size_tokens"], inplace=True)

    df.reset_index(drop=True, inplace=True)
    return df


def load_and_clean_sentiment_data(file_path):
    df = pd.read_csv(file_path)

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Ensure date format is proper
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Rename classification column (so we always have "classification")
    if "classification" not in df.columns and "class" in df.columns:
        df.rename(columns={"class": "classification"}, inplace=True)

    df.dropna(subset=["date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def merge_datasets(trader_df, sentiment_df):
    # Merge on date (both already standardized)
    merged = pd.merge(trader_df, sentiment_df, on="date", how="left")

    # Keep only rows where sentiment classification is known
    merged = merged.dropna(subset=["classification"])
    return merged

def save_cleaned_data(df, file_path):
    df.to_csv(file_path, index=False)