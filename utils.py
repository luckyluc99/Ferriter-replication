import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import t
from math import floor


def macrovariables(abbr: str):
    if abbr == "US":
        return [
            "Unemployment Rate",
            "U. of Mich. Sentiment",
            "Retail Sales Ex Auto and Gas",
            "PPI Ex Food and Energy MoM",
            "Initial Jobless Claims",
            "Industrial Production MoM",
            "ISM Manufacturing",
            "Housing Starts MoM",
            "GDP Annualized QoQ",
            "Change in Nonfarm Payrolls",
            "CPI Ex Food and Energy YoY",
        ]
    else:
        return []
    
# Function to convert dates
def convert_date(date_str):
    """
    Converts date strings to a consistent 'MM/DD/YYYY' format.

    Args:
      date_str: The date string to standardize.

    Returns:
      A date string in 'MM/DD/YYYY' format, or None if the input is invalid.
    """
    try:
        # Attempt to parse as YYYY-MM-DD HH:MM:SS
        date_obj = pd.to_datetime(date_str)
        return date_obj.strftime('%m/%d/%Y')
    except ValueError:
        try:
            # Attempt to parse as MM/DD/YYY and add 2000 to the year if it's less than 1000
            month, day, year = map(int, date_str.split('/'))
            if year < 1000:
                year += 2000
            return "{:02d}/{:02d}/{}".format(month, day, year)

        except ValueError:
            return None  # Return None for invalid date formats
        
# Function to get time range
def get_time_range(datetime_str):
    """
    Calculates the time 15 minutes before and 30 minutes after a given datetime string.

    Args:
        datetime_str: A datetime string in the format "YYYY-MM-DD HH:MM:SS".

    Returns:
        A tuple containing two strings:
        - The datetime 15 minutes earlier, in the same format.
        - The datetime 30 minutes later, in the same format.
    """
    try:
        dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return "Invalid datetime format. Please use 'YYYY-MM-DD HH:MM:SS'"

    earlier_dt = dt - timedelta(minutes=15)
    later_dt = dt + timedelta(minutes=30)

    return earlier_dt.strftime("%Y-%m-%d %H:%M:%S"), later_dt.strftime("%Y-%m-%d %H:%M:%S")

def calculate_exponential_weights(datetimes, half_life_years):
  """
  Calculates exponentially decaying weights for an array of datetimes.

  Args:
    datetimes: A NumPy array of datetime strings (e.g., "YYYY-MM-DD HH:MM:SS").
    half_life_years: The half-life in years for the exponential decay.

  Returns:
    A NumPy array of the same size as datetimes, containing the corresponding weights.
  """

  # Convert datetime strings to NumPy datetime64 objects
  datetimes = np.array(datetimes, dtype='datetime64[ns]')

  # Get today's date as the reference point (last element of the array)
  today = datetimes[-1]

  # Calculate the time difference in days between each date and today
  time_diffs = (today - datetimes).astype('timedelta64[D]').astype(int)

  # Calculate the decay factor (lambda) based on the half-life
  decay_factor = np.log(2) / (half_life_years * 365.25)  # Convert half-life to days

  # Calculate the weights using the exponential decay formula
  weights = np.exp(-decay_factor * time_diffs)

  return weights

def weighted_least_squares(y, X, datetimes, half_life_years, lag=None):
    """
    Performs a single weighted least squares regression with exponential weights, 
    and calculates HAC standard errors, t-stats, and p-values.

    Args:
        y (array-like): 1D array of intraday returns.
        X (array-like): 2D array of macro surprises.
        datetimes (array-like): Datetime values corresponding to the returns and surprises.
        half_life_years (float): Half-life of the exponential weighting scheme in years.
        lag (int, optional): Lag for the Newey-West estimator of covariance matrix.

    Returns:
        tuple: (np.array: Vector of coefficients, np.array: Vector of p-values, np.array: Vector of t_stats).
    """
    y = np.asarray(y)
    X = np.asarray(X)
    datetimes = pd.to_datetime(datetimes)

    weights = calculate_exponential_weights(datetimes, half_life_years)

    if X.shape[1] >= len(X):  # Handle case where not enough data points to conduct a regression
        coef = np.full(X.shape[1], np.nan)
        p_values = np.full(X.shape[1], np.nan)
        t_stats = np.full(X.shape[1], np.nan)
    else:
        W = np.diag(weights)
        try:
            XTWX_inv = np.linalg.inv(X.T @ W @ X)  # Compute XTWX inverse matrix
            coef = XTWX_inv @ X.T @ W @ y  # Compute coefficients as a matrix

            residuals = y - X @ coef  # Compute residuals

            # Calculate the HAC Covariance Matrix (Newey-West)
            if lag is None:
                lag = floor(4 * (len(y) / 100) ** (2 / 9))  # Rule of thumb if not provided

            T = len(y)
            omega = np.zeros((X.shape[1], X.shape[1]))
            for i in range(T):
                omega += weights[i] * np.outer(X[i] * residuals[i], X[i] * residuals[i])

            for i in range(1, lag + 1):
                gamma = np.zeros((X.shape[1], X.shape[1]))
                for j in range(i, T):
                    gamma += weights[j] * np.outer(X[j] * residuals[j], X[j - i] * residuals[j - i])
                omega += gamma + gamma.T

            cov_matrix = XTWX_inv @ omega @ XTWX_inv  # Compute covariance matrix

            std_err = np.sqrt(np.diag(cov_matrix))  # Compute standard error

            t_stats = coef / std_err  # Compute t-statistics
            df = len(y) - X.shape[1]  # Degrees of Freedom
            p_values = np.array([2 * (1 - t.cdf(np.abs(t_stat), df)) for t_stat in t_stats])  # Corrected line

            try: # get R-squared of the regression
                y_hat = X @ coef
                y_bar = np.mean(y)
                ssr = np.sum((y - y_hat) ** 2)
                sst = np.sum((y - y_bar) ** 2)
                r_squared = 1 - ssr / sst
            except ZeroDivisionError:
                r_squared = np.nan

        except np.linalg.LinAlgError:
            coef = np.full(X.shape[1], np.nan)  # Handle case of singular matrix
            p_values = np.full(X.shape[1], np.nan)
            t_stats = np.full(X.shape[1], np.nan)
            r_squared = np.nan

    return coef, p_values, t_stats

def calculate_weekly_returns(file_path):
    """
    Calculates daily df from intraday data.

    Args:
      file_path (str): Path to the CSV file containing intraday data.

    Returns:
      pd.DataFrame: DataFrame with weekly log returns and the week's end date as the index.
    """
    # Read the CSV file and parse Date and Time into Datetime
    df = pd.read_csv(file_path, parse_dates={'Datetime': ['Date', 'Time']})
    df.set_index('Datetime', inplace=True)

    # Resample and calculate daily log returns
    df_daily = df['Close'].resample('D').agg(['first', 'last'])
    df_daily['daily_return'] = np.log(df_daily['last'] / df_daily['first'])

    df_weekly = df_daily['daily_return'].resample('W-SAT').sum() * 100
    return df_weekly