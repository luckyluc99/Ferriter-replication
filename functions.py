import pandas as pd
import numpy as np
import os
import statsmodels.api as sm

from utils import (
    convert_date,
    macrovariables,
    get_time_range,
    weighted_least_squares,
    calculate_weekly_returns,
)

# For EWSTD calculation
frequency_to_periods = {
    "Q": 4,   # Quarterly
    "M": 12,  # Monthly
    "W": 52,  # Weekly
    "D": 252  # Daily (trading days)
}
lambda_dict = {
    freq: np.exp(-np.log(2) / (3 * periods))
    for freq, periods in frequency_to_periods.items()
}

# Only US abbreviations
US_abbreviations = ("US", "ES", "TY")

# Initial window size and half-life for the exponential weights
window_size = 200 # For US doesn't really matter as the data starts in 2002
half_life_years = 3

def get_filtered_macro_variables(data_dir, output_dir, g4_abbreviations = ['US']):
    data_dict = {}
    for abbr in g4_abbreviations:
        file_path = os.path.join(data_dir, f"Economic Releases Events/{abbr}_economic_releases_events.csv")

        # Load the data
        data = pd.read_csv(file_path, low_memory=False)

        # Retain only these columns
        data = data[['Event', 'Actual', 'Prior', 'Freq.', 'Date', 'Time', 'Surv(M)', 'Std Dev']]
        
        # Convert the 'Actual' and Surv(M) column to numeric, coercing errors
        # some values are strings like 100k or 50M or -1M or $25b also convert these to numeric (scaling doesn't matter)
        data['Actual'] = data['Actual'].str.replace(r'[^\d.-]', '', regex=True)
        data['Surv(M)'] = data['Surv(M)'].str.replace(r'[^\d.-]', '', regex=True)
        data['Actual'] = pd.to_numeric(data['Actual'], errors='coerce')
        data['Surv(M)'] = pd.to_numeric(data['Surv(M)'], errors='coerce')

        # Save non-normalised surprises and initialise the 'Surprise' column
        data['nn-surprise'] = data['Actual'] - data['Surv(M)']
        data['Surprise'] = 0

        # Drop rows with missing 'Std Dev', 'SActual', 'Surv(M)', or 'Time' entries
        data = data.dropna(subset=['Std Dev', 'Actual', 'Surv(M)', 'Time'])
        
        # Convert the 'Date' and 'Time' column to the desired format and make datetime column
        data['Date'] = data['Date'].apply(convert_date)
        data['Time'] = pd.to_datetime(data['Time'], format='%H:%M:%S').dt.time
        data['DateTime'] = pd.to_datetime(data['Date'].astype(str) + ' ' + data['Time'].astype(str))

        # Drop rows with invalid dates
        data = data.dropna(subset=['Date'])
        
        # Filter out rows before January 1, 2002
        data['Date'] = pd.to_datetime(data['Date'])
        data = data[data['Date'] >= pd.to_datetime('2002-01-01')]
        
        # Only retain macro variables that are in the selected list
        macro_vars_list = macrovariables(abbr)
        data = data[data['Event'].isin(macro_vars_list)]

        # Ensure the 'Surprise' column is of float dtype
        data['Surprise'] = data['Surprise'].astype(float)

        # Get the unique macro variables names
        macro_vars_names = data['Event'].unique()

        # Initialise the EWMA and EWMV dictionaries
        current_EWMA = {name: 1 for name in macro_vars_names}
        current_EWMV = {name: 1 for name in macro_vars_names}

        # Loop through the rows to compute the EWMSD and normalise the surprise
        for index, row in data.iterrows():
            current_EWMA[row['Event']] = lambda_dict[row['Freq.']] * current_EWMA[row['Event']] + (1 - lambda_dict[row['Freq.']]) * row['nn-surprise']
            current_EWMV[row['Event']] = lambda_dict[row['Freq.']] * current_EWMV[row['Event']] + (1 - lambda_dict[row['Freq.']]) * (row['nn-surprise'] - current_EWMA[row['Event']]) ** 2
            
            data.at[index, 'Surprise'] = (row['nn-surprise']) / np.sqrt(current_EWMV[row['Event']])

        # Save the filtered data to the dictionary
        data_dict[f"data_{abbr}"] = data

    # Save the filtered data to CSV files into the Output folder
    for key, value in data_dict.items():
        value.to_csv(f"{output_dir}/filtered_macro_{key}.csv", index=False)


def get_impact_coefficients_data(data_dir, output_dir, abbrvs = [US_abbreviations]):
    for abbr in abbrvs:
        dir_equity = os.path.join(data_dir, "Intraday price data/Equity")
        dir_bond = os.path.join(data_dir, "Intraday price data/Rates/")
        
        data_macro = pd.read_csv(f"{output_dir}/filtered_macro_data_{abbr[0]}.csv")
        data_equity = pd.read_csv(f"{dir_equity}/{abbr[1]}.csv")
        data_bond = pd.read_csv(f"{dir_bond}/{abbr[2]}.csv")
        # Combine 'Date' and 'Time' into a new 'DateTime' column
        data_equity['Date'] = pd.to_datetime(data_equity['Date'], format='%m/%d/%Y')
        data_equity['Time'] = pd.to_datetime(data_equity['Time'], format='%H:%M').dt.time
        data_equity['DateTime'] = pd.to_datetime(data_equity['Date'].astype(str) + ' ' + data_equity['Time'].astype(str))

        data_bond['Date'] = pd.to_datetime(data_bond['Date'], format='%m/%d/%Y')
        data_bond['Time'] = pd.to_datetime(data_bond['Time'], format='%H:%M').dt.time
        data_bond['DateTime'] = pd.to_datetime(data_bond['Date'].astype(str) + ' ' + data_bond['Time'].astype(str))

        # Get the unique macro releases 
        macro_vars_names = data_macro['Event'].unique()

        # Initialize the X matrix and the return vectors (is faster to append to a list)
        r_equity = []
        r_bond = []
        X_matrix_equity = []
        X_matrix_bond = [] # different X matrix due to possibly different intraday data
        datetime_equity = []
        datetime_bond = []

        current_datetime_equity = ""
        current_datetime_bond = ""

        for index, row in data_macro.iterrows():
            # Progress of loop
            print(f"{row['DateTime']} - {row['Event']}", end="\r")
            
            # if there is no surprise, skip (for future research, could add dummy)
            if row['Surprise'] == 0.0:
                continue

            # Get time 15 minutes before and 30 after the macro event
            date_time_15, date_time_30 = get_time_range(row['DateTime'])
            
            # Do equity and bond seperately due to potential missing data
            try:
                # Get the equity value 15 minutes before the macro event
                equity_value_15 = data_equity.loc[data_equity['DateTime'] == date_time_15, 'Open'].iloc[0]
                equity_value_30 = data_equity.loc[data_equity['DateTime'] == date_time_30, 'Open'].iloc[0]
            
                # calculate the intraday return
                intraday_return_equity = np.log(equity_value_30 / equity_value_15)
                
                # get the index of the current macro event
                current_index_equity = np.where(macro_vars_names == row['Event'])[0][0]
                # if the current macro has same time as previous release, add to the same row in the X matrix (this works due to the loop being in order)
                if row['DateTime'] == current_datetime_equity: # this does not account for releases being done in the 15-30 minute window (shouldn't matter due to no such occurences)
                    current_equity_row = X_matrix_equity[-1]
                    current_equity_row[current_index_equity] = row['Surprise']
                    X_matrix_equity[-1] = current_equity_row
                else:
                    # if no other releasecreate a new row for the X matrix and add return
                    r_equity.append(intraday_return_equity)
                    new_row = np.zeros(len(macro_vars_names))
                    new_row[current_index_equity] = row['Surprise']
                    X_matrix_equity.append(new_row)
                    current_datetime_equity = row['DateTime']
                    datetime_equity.append(current_datetime_equity)

            except Exception as e:
                # if no data print and skip (for example holidays)
                print(e)
            
            # Do the same for the bond data (seperately due to different intraday data)
            try:
                bond_value_15 = data_bond.loc[data_bond['DateTime'] == date_time_15, 'Open'].iloc[0]
                bond_value_30 = data_bond.loc[data_bond['DateTime'] == date_time_30, 'Open'].iloc[0]
                
                intraday_return_bond = np.log(bond_value_30 / bond_value_15)

                # get the index of the current macro event
                current_index_bond = np.where(macro_vars_names == row['Event'])[0][0]
                
                if row['DateTime'] == current_datetime_bond: # if the current macro has same time as previous, add to the same row
                    current_bond_row = X_matrix_bond[-1]
                    current_bond_row[current_index_bond] = row['Surprise']
                    X_matrix_bond[-1] = current_bond_row
                else:
                    # create a new row for the X matrix and add return
                    r_bond.append(intraday_return_bond)
                    new_row = np.zeros(len(macro_vars_names))
                    new_row[current_index_bond] = row['Surprise']
                    X_matrix_bond.append(new_row)
                    current_datetime_bond = row['DateTime']
                    datetime_bond.append(current_datetime_bond)

            except Exception as e:
                # if no data print and skip (for example holidays)
                print(e)
        
        # add column of ones as constant to first column of X matrix (for constant) and give name to macro vars
        X_matrix_equity = np.array(X_matrix_equity)
        X_matrix_bond = np.array(X_matrix_bond)
        X_matrix_equity = np.insert(X_matrix_equity, 0, 1, axis=1)
        X_matrix_bond = np.insert(X_matrix_bond, 0, 1, axis=1)
        macro_vars_names = np.insert(macro_vars_names, 0, "Constant")

        np.savez_compressed(os.path.join(output_dir, f"impact_coefficients_data_{abbr[0]}.npz"),
                        r_equity=r_equity,
                        r_bond=r_bond,
                        X_matrix_equity=X_matrix_equity,
                        X_matrix_bond=X_matrix_bond,
                        datetime_equity=datetime_equity,
                        datetime_bond=datetime_bond,
                        macro_vars_names=macro_vars_names)
        
def impact_coefficients_regression(output_dir, abbrvs = ["US"]):
    for country_name in abbrvs:
        # Load the data from the .npz file
        dir = os.path.join(output_dir, f"impact_coefficients_data_{country_name}.npz")
        data = np.load(dir, allow_pickle=True)
        # Dictionary to organize the variables
        variables = {
            'equity': {
                'r': data['r_equity'],
                'X_matrix': data['X_matrix_equity'],
                'datetime': pd.to_datetime(data['datetime_equity'])
            },
            'bond': {
                'r': data['r_bond'],
                'X_matrix': data['X_matrix_bond'],
                'datetime': pd.to_datetime(data['datetime_bond'])
            }
        }
        macro_vars_names = data['macro_vars_names']

        # Create empty dictionaries to store the dataframes
        all_coefs_dfs = {}
        all_p_values_dfs = {}

        # Loop through the variables and perform your desired operations
        for asset_type, asset_data in variables.items():
            # Access variables for this asset type
            r = np.asarray(asset_data['r'])
            X_matrix = np.asarray(asset_data['X_matrix'])
            datetime = asset_data['datetime']

            # Create empty DataFrames with datetime index and with correct column names (increasing window size)
            all_coefs_df = pd.DataFrame(index=datetime[window_size:], columns=macro_vars_names, dtype = float)
            all_p_values_df = pd.DataFrame(index=datetime[window_size:], columns=macro_vars_names, dtype = float)
            
            # Get column names
            all_column_names = np.array(macro_vars_names)

            # Perform the weighted least squares regression for each window
            for i in range(window_size, len(datetime)):
                print(f"Processing window {i} of {len(datetime)} for {country_name} {asset_type}...", end="\r")
                current_datatimes = datetime[0:i]
                current_return = r[0:i]
                current_X = X_matrix[0:i]
                
                # Identify non-zero columns and zero columns
                non_zero_cols = np.where(np.any(current_X != 0, axis = 0))[0]
                
                # Remove zero columns from current_X
                if non_zero_cols.size > 0: # Do the regression only if there is one non-zero column

                    current_X_select = current_X[:, non_zero_cols]

                    # perform the regression
                    coefs, p_values, t_stats = weighted_least_squares(current_return, current_X_select, current_datatimes, half_life_years)
                
                    # Create a dictionary to map coefficients to column names (also works if there are zeros in a full column)
                    coef_dict = dict(zip(all_column_names[non_zero_cols], coefs))
                    p_values_dict = dict(zip(all_column_names[non_zero_cols], p_values))

                    # Create series with correct index using only the coefficients that are non-nan
                    coefs_series = pd.Series(index=all_column_names, dtype=float)  # dtype added
                    p_values_series = pd.Series(index=all_column_names, dtype=float) # dtype added

                    # Populate the series, giving zeros to columns that were all zero, and the corresponding coefficients to the other
                    for col_name in all_column_names: # loop through all column names
                        if col_name in coef_dict: # if the column was in the regression
                            coefs_series[col_name] = coef_dict[col_name] #add the coefficients
                            p_values_series[col_name] = p_values_dict[col_name]
                    
                    # Append to DataFrame by position, not by appending rows
                    all_coefs_df.loc[datetime[i], coefs_series.index] = coefs_series.values
                    all_p_values_df.loc[datetime[i], p_values_series.index] = p_values_series.values
            
                else: # If all columns are zeros populate the columns as Nans
                    all_coefs_df.loc[datetime[i], :] = np.nan
                    all_p_values_df.loc[datetime[i], :] = np.nan
        
            # Remove columns that contain only missing values
            all_coefs_df = all_coefs_df.dropna(how='all', axis = 1)
            all_p_values_df = all_p_values_df.dropna(how='all', axis = 1)

            # Fill the remaining NaNs with zeros
            all_coefs_df = all_coefs_df.fillna(0)
            all_p_values_df = all_p_values_df.fillna(0)

            all_coefs_dfs[asset_type] = all_coefs_df
            all_p_values_dfs[asset_type] = all_p_values_df
            
        np.savez_compressed(os.path.join(output_dir, f"coefficients_{country_name}.npz"),
                            beta_equity=all_coefs_dfs['equity'],
                            p_value_equity=all_p_values_dfs['equity'],
                            datetime_equity=data['datetime_equity'],
                            beta_bond=all_coefs_dfs['bond'],
                            p_value_bond=all_p_values_dfs['bond'],
                            datetime_bond=data['datetime_bond'],
                            macro_vars_names=macro_vars_names)
        
def get_weekly_returns(data_dir, output_dir, abbreviations = [US_abbreviations]):
    for abbr in abbreviations:
        data_equity_path = os.path.join(data_dir, f"Intraday price data/Equity/{abbr[1]}.csv")
        data_bond_path = os.path.join(data_dir, f"Intraday price data/Rates/{abbr[2]}.csv")

        # Calculate weekly log returns for both bond and equity markets
        weekly_log_returns_df_equity = calculate_weekly_returns(data_equity_path)
        weekly_log_returns_df_bond = calculate_weekly_returns(data_bond_path)

        np.savez_compressed(os.path.join(output_dir, f"weekly_returns_{abbr[0]}.npz"),
                            dates_equity=weekly_log_returns_df_equity.index,
                            weekly_return_equity=weekly_log_returns_df_equity,
                            dates_bond=weekly_log_returns_df_bond.index,
                            weekly_return_bond=weekly_log_returns_df_bond)
        
def get_mispricing_scores(output_dir, abbreviations = ["US"]):
    for abbr in abbreviations:
        # Load the data from the .npz file
        data_coef = np.load(os.path.join(output_dir, f"coefficients_{abbr}.npz"), allow_pickle=True)
        data_surprises = np.load(os.path.join(output_dir, f"impact_coefficients_data_{abbr}.npz"), allow_pickle=True)
        data_returns = np.load(os.path.join(output_dir, f"weekly_returns_{abbr}.npz"), allow_pickle=True)

        # Dictionary to organize the variables
        variables = {
            'equity': {
                'beta': data_coef['beta_equity'],
                'X_matrix': data_surprises['X_matrix_equity'],
                'datetime': pd.to_datetime(data_coef['datetime_equity']),
                'weekly_returns': data_returns['weekly_return_equity'],
                'datetime_returns' : data_returns['dates_equity']
            },
            'bond': {
                'beta': data_coef['beta_bond'],
                'X_matrix': data_surprises['X_matrix_bond'],
                'datetime': pd.to_datetime(data_coef['datetime_bond']),
                'weekly_returns': data_returns['weekly_return_bond'],
                'datetime_returns' : data_returns['dates_bond']
            }
        }

        weekly_nowcast_dfs = {}
        mispricing_score_dfs = {}

        # Loop through the variables and perform your desired operations
        for asset_type, asset_data in variables.items():
            
            # --- Calculate weighted sum (daily nowcast) ---
            beta = np.array(asset_data['beta'])
            X_matrix = np.array(asset_data['X_matrix'])
            dates = asset_data['datetime'].values

            # Dimension checks
            if beta.shape[1] != X_matrix.shape[1]:
                raise ValueError("Number of variables (columns) in beta and X_matrix must be equal.")
            if beta.shape[0] != (X_matrix.shape[0] - window_size):
                raise ValueError("Beta matrix needs to have a row for every row in X_matrix after the first 200 rows.")
            if len(dates) != X_matrix.shape[0]:
                raise ValueError("Dates need to be the same length as the amount of rows in X_matrix")

            # Calculate weighted sum (daily nowcast)
            X_matrix_subset = X_matrix[window_size:, :]
            weighted_sums = np.sum(beta * X_matrix_subset, axis=1)
            daily_nowcast_df = pd.DataFrame({'macro_variables': weighted_sums}, index=dates[window_size:])
            daily_nowcast_df.index = pd.to_datetime(daily_nowcast_df.index)

            # --- Calculate weekly sum (weekly nowcast) ---
            weekly_nowcast_df = daily_nowcast_df.resample('W-SAT').sum()
            weekly_nowcast_df.index = weekly_nowcast_df.index.to_period('W-SAT').to_timestamp('W-SAT')
            
            # --- Calculate Time Series Score ---
            mean_ni = weekly_nowcast_df['macro_variables'].ewm(halflife=36, min_periods=1).mean()
            std_ni = (weekly_nowcast_df['macro_variables'] - mean_ni).ewm(halflife=36, min_periods=1).std()
            weekly_nowcast_df['time_series_score'] = (weekly_nowcast_df['macro_variables'] - mean_ni) / std_ni

            weekly_nowcast_dfs[asset_type] = weekly_nowcast_df.copy() # store the weekly nowcast

            # --- Calculate Exponential Weighted Mean and Stdev of Weekly Returns ---
            weekly_return_score = pd.DataFrame({'time_series_score': asset_data['weekly_returns']}, index=asset_data['datetime_returns'])
            mean_ai = weekly_return_score['time_series_score'].ewm(halflife=36, min_periods=1).mean()
            std_ai = (weekly_return_score['time_series_score'] - mean_ai).ewm(halflife=36, min_periods=1).std()
            weekly_return_score['time_series_score'] = (weekly_return_score['time_series_score'] - mean_ai) / std_ai
            
            # Align Indexes Before Joining
            common_index = weekly_return_score.index.intersection(weekly_nowcast_dfs[asset_type].index)
            weekly_return_score = weekly_return_score.loc[common_index]
            weekly_nowcast_score = weekly_nowcast_dfs[asset_type].loc[common_index]
                    
            weekly_return_score = weekly_return_score.rename(columns={"time_series_score": "aiw"})
            weekly_nowcast_score = weekly_nowcast_score.rename(columns={"time_series_score": "niw"})

            df = weekly_return_score.join(weekly_nowcast_score, how='inner')
            # Remove rows with missing values
            df.dropna(inplace=True)

            # --- Calculate Mispricing Score ---
            X = df['niw']
            model = sm.OLS(df['aiw'], X)
            results = model.fit()
            df['miw'] = results.resid
            df['mispricing_score'] = df['miw'].ewm(halflife=8, min_periods=1).mean()

            mispricing_score_dfs[asset_type] = df['mispricing_score'].to_frame() # Store mispricing score
        
        # remove first 8 weeks due to EMAs
        mispricing_score_dfs['equity'] = mispricing_score_dfs['equity'].iloc[8:]    
        mispricing_score_dfs['bond'] = mispricing_score_dfs['bond'].iloc[8:]

        # Save equity mispricing score to CSV
        equity_df = mispricing_score_dfs['equity'].copy()
        equity_df.index.name = 'Date'
        equity_df.rename(columns={'mispricing_score': 'Mispricing'}, inplace=True)
        equity_df.to_csv(os.path.join(output_dir, f"mispricing_equity_{abbr}.csv"))

        # Save bond mispricing score to CSV
        bond_df = mispricing_score_dfs['bond'].copy()
        bond_df.index.name = 'Date'
        bond_df.rename(columns={'mispricing_score': 'Mispricing'}, inplace=True)
        bond_df.to_csv(os.path.join(output_dir, f"mispricing_bond_{abbr}.csv"))