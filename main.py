from functions import (
    get_filtered_macro_variables,
    get_impact_coefficients_data,
    impact_coefficients_regression,
    get_weekly_returns,
    get_mispricing_scores,
)
import os


data_directory = './Data/'
output_directory = './Output/'

# Create the directory if it does not exist
os.makedirs(output_directory, exist_ok=True)

get_filtered_macro_variables(data_directory, output_directory)
get_impact_coefficients_data(data_directory, output_directory)
impact_coefficients_regression(output_directory)
get_weekly_returns(data_directory, output_directory)
get_mispricing_scores(output_directory)
