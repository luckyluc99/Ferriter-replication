import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

start_date = pd.to_datetime('2015-01-01')
end_date = pd.to_datetime('2019-12-31')
country = 'US'

mispricing_score_dfs = {}
returns_dict = {}

data_dir = './Output'

# load mispricing scores
data_mispricing_bond = pd.read_csv(os.path.join(data_dir, f"mispricing_bond_{country}.csv"))
data_mispricing_equity = pd.read_csv(os.path.join(data_dir, f"mispricing_equity_{country}.csv"))

mispricing_score_dfs[country] = {
    'equity': data_mispricing_equity['Mispricing'].tolist(),
    'datetime_equity': pd.to_datetime(data_mispricing_equity['Date']).tolist(),
    'bond': data_mispricing_bond['Mispricing'].tolist(),
    'datetime_bond': pd.to_datetime(data_mispricing_bond['Date']).tolist()
}

# load returns
data_returns = np.load(os.path.join(data_dir, f"weekly_returns_{country}.npz"))
returns_dict[country] = {
    'equity': data_returns['weekly_return_equity'],
    'datetime_equity': pd.to_datetime(data_returns['dates_equity']),
    'bond': data_returns['weekly_return_bond'],
    'datetime_bond': pd.to_datetime(data_returns['dates_bond'])
}

# Create dataframes for equity and bond mispricing and returns
equity_mispricing_df = pd.DataFrame({
    'datetime': pd.to_datetime(mispricing_score_dfs[country]['datetime_equity']),
    'mispricing': mispricing_score_dfs[country]['equity']
})

bond_mispricing_df = pd.DataFrame({
    'datetime': pd.to_datetime(mispricing_score_dfs[country]['datetime_bond']),
    'mispricing': mispricing_score_dfs[country]['bond']
})

equity_returns_df = pd.DataFrame({
    'datetime': pd.to_datetime(returns_dict[country]['datetime_equity']),
    'returns': returns_dict[country]['equity']
})

bond_returns_df = pd.DataFrame({
    'datetime': pd.to_datetime(returns_dict[country]['datetime_bond']),
    'returns': returns_dict[country]['bond']
})

# Filter dataframes based on start and end dates
equity_mispricing_df = equity_mispricing_df[(equity_mispricing_df['datetime'] >= start_date) & (equity_mispricing_df['datetime'] <= end_date)]
equity_returns_df = equity_returns_df[(equity_returns_df['datetime'] >= start_date) & (equity_returns_df['datetime'] <= end_date)]
bond_mispricing_df = bond_mispricing_df[(bond_mispricing_df['datetime'] >= start_date) & (bond_mispricing_df['datetime'] <= end_date)]
bond_returns_df = bond_returns_df[(bond_returns_df['datetime'] >= start_date) & (bond_returns_df['datetime'] <= end_date)]

# Merge mispricing and returns dataframes for equity and bonds
equity_merged_df = pd.merge(equity_mispricing_df, equity_returns_df, on='datetime', how='inner')
bond_merged_df = pd.merge(bond_mispricing_df, bond_returns_df, on='datetime', how='inner')

def plot_correlation(asset_type, country, start_date, end_date, mispricing_score_dfs, returns_dict, past_week=False):
    """
    Plots the correlation between mispricing score and future weekly returns for a given asset type.

    Args:
        asset_type (str): 'equity' or 'bond'.
        country (str): The country code (e.g., 'US').
        start_date (pd.Timestamp): The start date for the analysis.
        end_date (pd.Timestamp): The end date for the analysis.
        mispricing_score_dfs (dict): Dictionary containing mispricing scores and dates.
        returns_dict (dict): Dictionary containing returns and dates.
    """

    # Convert the mispricing data to a pandas Series with datetime index
    mispricing_scores = mispricing_score_dfs[country][asset_type]
    mispricing_dates = pd.to_datetime(mispricing_score_dfs[country][f'datetime_{asset_type}'])
    mispricing_series = pd.Series(mispricing_scores, index=mispricing_dates)

    # Convert the returns data to a pandas Series with datetime index
    returns = returns_dict[country][asset_type]
    returns_dates = pd.to_datetime(returns_dict[country][f'datetime_{asset_type}'])
    returns_series = pd.Series(returns, index=returns_dates)

    # Filter the data based on the start and end dates
    mispricing_series = mispricing_series[start_date:end_date]
    returns_series = returns_series[start_date:end_date]


    # Calculate the correlation between mispricing score and *future* weekly return for 52 lags
    correlations = []
    for lag in range(1, 53):
        # Shift the mispricing scores by the lag.  Crucially, shift *forward* in time,
        # so we are correlating *past* mispricing with *current* returns.
        if past_week:
            shifted_mispricing = mispricing_series.shift(-lag)
        else:
            shifted_mispricing = mispricing_series.shift(lag)

        # Align the shifted mispricing scores and returns, dropping any resulting NaN values.
        aligned_data = pd.DataFrame({'mispricing': mispricing_series, 'returns': returns_series}).dropna()
        aligned_data_shifted = pd.DataFrame({'mispricing': shifted_mispricing, 'returns': returns_series}).dropna()


        # Calculate the correlation using the aligned data
        correlation = aligned_data_shifted['mispricing'].corr(aligned_data_shifted['returns']) #use the shifted series
        correlations.append(correlation)


    # --- Plotting ---
    plt.figure(figsize=(12, 6))  # Adjust figure size for better readability

    # Use different colors based on correlation sign
    colors = ['#E54729' if c < 0 else 'green' for c in correlations]  # Red for negative, Green for positive
    plt.bar(np.arange(1, 53), correlations, color=colors)

    plt.xlabel('Weeks', fontsize=14)
    plt.ylabel('Correlation Coefficient', fontsize=14)
    if past_week:
        plt.title(f'Correlation of {asset_type.capitalize()} Mispricing Score and Past Weekly Returns ({country})', fontsize=16)
    else:
        plt.title(f'Correlation of {asset_type.capitalize()} Mispricing Score and Subsequent Weekly Returns ({country})', fontsize=16)
    plt.xticks(range(1, 53, 4), fontsize=12)  # Show x-axis labels every 4 weeks
    plt.yticks(fontsize=12)
    plt.grid(True, which='both', linestyle='-', color='lightgray')
    plt.axhline(y=0, color='black', linewidth=0.5)  # Add a horizontal line at y=0
    plt.xlim(0.5, 52.5)
    plt.ylim(-0.4, 0.4)   # consistent y-axis limits
    plt.tight_layout()  # Adjust layout to prevent labels from overlapping
    if past_week:
        plt.savefig(f'./Output/{country}_{asset_type}_correlation_past_week.pdf')
    else:
        plt.savefig(f'./Output/{country}_{asset_type}_correlation.pdf')
    plt.show()

# Create the correlation plots
plot_correlation('equity', country, start_date, end_date, mispricing_score_dfs, returns_dict)
plot_correlation('bond', country, start_date, end_date, mispricing_score_dfs, returns_dict)

# Create the correlation plots for past week
plot_correlation('equity', country, start_date, end_date, mispricing_score_dfs, returns_dict, past_week=True)
plot_correlation('bond', country, start_date, end_date, mispricing_score_dfs, returns_dict, past_week=True)