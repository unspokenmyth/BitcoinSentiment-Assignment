import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_sentiment_trend(df):
    """Plot daily Fear-Greed Index trend."""
    plt.figure(figsize=(10, 5))
    sns.lineplot(x='Date', y='value', data=df, marker='o', linewidth=2)
    plt.title('Bitcoin Fear-Greed Index Over Time')
    plt.xlabel('Date')
    plt.ylabel('Index Value')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_pnl_by_sentiment(df):
    """Visualize average PnL per sentiment class."""
    plt.figure(figsize=(8, 5))
    grouped = df.groupby('classification')['Closed PnL'].mean().reset_index()
    sns.barplot(x='classification', y='Closed PnL', data=grouped, palette='coolwarm')
    plt.title('Average Trader PnL by Market Sentiment')
    plt.xlabel('Sentiment Classification')
    plt.ylabel('Avg Closed PnL')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()


def correlation_heatmap(df, cols):
    """Show correlation between key numerical variables."""
    corr = df[cols].corr()
    plt.figure(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap='YlGnBu', fmt=".2f", linewidths=0.3)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()


def trade_activity_bar(df):
    """Plot number of trades grouped by sentiment."""
    counts = df.groupby('classification').size().reset_index(name='Trade Count')
    plt.figure(figsize=(8, 5))
    sns.barplot(x='classification', y='Trade Count', data=counts, palette='viridis')
    plt.title('Number of Trades by Market Sentiment')
    plt.xlabel('Market Sentiment')
    plt.ylabel('Number of Trades')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()
