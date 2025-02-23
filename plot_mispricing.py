import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# load data
data_bond = pd.read_csv('./Output/mispricing_bond_US.csv')
data_equity = pd.read_csv('./Output/mispricing_equity_US.csv')

# Convert 'Date' to datetime objects
data_bond['Date'] = pd.to_datetime(data_bond['Date'])
data_equity['Date'] = pd.to_datetime(data_equity['Date'])

# Set 'Date' as index
data_bond.set_index('Date', inplace=True)
data_equity.set_index('Date', inplace=True)

# Filter data for the period 2015-01-01 to 2019-12-31
start_date = '2015-01-01'
end_date = '2019-12-31'
data_bond_filtered = data_bond.loc[start_date:end_date]
data_equity_filtered = data_equity.loc[start_date:end_date]

# plot
plt.figure(figsize=(10, 6))
plt.plot(data_bond_filtered.index, data_bond_filtered['Mispricing'], label='Bond Mispricing')
plt.xlabel('Date')
plt.ylabel('Mispricing')
plt.title('Bond Mispricing Over Time (2015-2019)')
plt.legend()
plt.grid(True)
plt.xlim(data_bond_filtered.index.min(), data_bond_filtered.index.max())
plt.tight_layout()  # Add this line
plt.savefig('./Output/mispricing_bond.pdf')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(data_equity_filtered.index, data_equity_filtered['Mispricing'], label='Equity Mispricing')
plt.xlabel('Date')
plt.ylabel('Mispricing')
plt.title('Equity Mispricing Over Time (2015-2019)')
plt.legend()
plt.xlim(data_equity_filtered.index.min(), data_equity_filtered.index.max())
plt.grid(True)
plt.tight_layout()  # Add this line
plt.savefig('./Output/mispricing_equity.pdf')
plt.show()