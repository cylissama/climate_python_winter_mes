import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

# Configuration
site = 'BMTN'
file_mes1 = '01-Mar-2014_01-Aug-2023_BMTN_daily'
date_start = pd.Timestamp('2015-01-01')
date_end = pd.Timestamp('2022-12-31')

# Define paths
dir_cli1 = '/Volumes/Mesonet/winter_break/CCdata/'
dir_cli = os.path.join(dir_cli1, site, '')
file_thresh = os.path.join(dir_cli, f"{site}_CLIthresh_daily.pkl")
file_mes = os.path.join(dir_cli, f"{file_mes1}.pkl")

# Load Mesonet, Climate Data and Thresholds
# In Python, we'll use pandas to load the data instead of MATLAB's load
# Assuming the data is saved in a pickle format
import pickle

# Load threshold data
with open(file_thresh, 'rb') as f:
    thresh_data = pickle.load(f)

# Load Mesonet data
with open(file_mes, 'rb') as f:
    tt_daily_mes = pickle.load(f)

# Extract time information
time_full = tt_daily_mes['TimestampCollected']
s_time_full = time_full[0]
e_time_full = time_full[-1]

# Find indices for date range
is_d = tt_daily_mes[tt_daily_mes['TimestampCollected'] == date_start].index[0]
ie_d = tt_daily_mes[tt_daily_mes['TimestampCollected'] == date_end].index[0]

# Define old and new variable names
var_old = ["TAIR", "TAIRx", "TAIRn", "DWPT", "PRCP", "PRES", "RELH", "SM02", "SRAD", "WDIR", "WSPD", "WSMX"]
var = ["TAIR_annual", "TAIRx_annual", "TAIRn_annual", "DWPT_annual", "PRCP_annual", "PRES_annual",
       "RELH_annual", "SM02_annual", "SRAD_annual", "WDIR_annual", "WSPD_annual", "WSMX_annual"]

# Extract subset of data for date range
tt = tt_daily_mes.iloc[is_d:ie_d+1]

# Remove variables
columns_to_remove = ['SM20', 'SM04', 'SM40', 'SM08', 'ST02', 'ST20', 'ST04', 'ST40', 'ST08', 'WSMN']
for col in columns_to_remove:
    if col in tt.columns:
        tt = tt.drop(columns=col)

# Rename variables
var_mapping = dict(zip(var_old, var))
tt = tt.rename(columns=var_mapping)

# Extract time and unique years
time = tt['TimestampCollected']
years = np.unique(time.dt.year)

# Initialize data structure
data_annual = {
    'year': years,
    'var': var
}

# Process data by year
ny = len(years)
nv = len(var)
data_year = []

for k in range(ny):
    tt_a = tt[tt['TimestampCollected'].dt.year == years[k]]
    data_year.append(tt_a)

data_annual['data'] = data_year

# Save the processed data
output_file = os.path.join(dir_cli, f"{site}_annual_data.pkl")
with open(output_file, 'wb') as f:
    pickle.dump(data_annual, f)

print(f"Annual climate data processed and saved to {output_file}")