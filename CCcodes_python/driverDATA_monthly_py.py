"""
driverDATA_monthly.py

This script processes meteorological data from Mesonet stations and 
organizes it into a monthly data structure. It is a Python implementation 
of the MATLAB script driverDATA_monthly.m.

Functions:
- Loads daily meteorological data from CSV files
- Filters data by date range
- Reorganizes data into monthly structures
- Saves processed data for later use in climate indices calculation

Author: Converted from MATLAB
Date: 2025-03-24
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import h5py

def process_monthly_data(site='HCKM', 
                        file_mes='01-Nov-2009_01-Aug-2023_HCKM_daily.csv',
                        date_start='2009-11-01',
                        date_end='2023-07-31'):
    """
    Process daily meteorological data and organize by month
    
    Parameters:
    -----------
    site : str
        Site code
    file_mes : str
        Input meteorological data file
    date_start : str
        Start date in format YYYY-MM-DD
    date_end : str
        End date in format YYYY-MM-DD
        
    Returns:
    --------
    data_month : dict
        Dictionary with organized monthly data
    """
    print(f"Processing monthly data for site {site}")
    
    # Base directory
    base_dir = Path('/Users/erappin/Documents/Mesonet/ClimateIndices/sitesTEST_CCindices/')
    site_dir = base_dir / site
    
    # Load threshold file
    thresh_file = site_dir / f"{site}_CLIthresh_daily.csv"
    try:
        thresh_data = pd.read_csv(thresh_file)
        print(f"Loaded threshold data from {thresh_file}")
    except FileNotFoundError:
        print(f"Threshold file not found: {thresh_file}")
        thresh_data = None
    
    # Load meteorological data
    mes_file = site_dir / file_mes
    try:
        # Try reading as CSV first
        df = pd.read_csv(mes_file, parse_dates=['TimestampCollected'])
        print(f"Loaded meteorological data from {mes_file}")
    except (FileNotFoundError, pd.errors.ParserError):
        try:
            # If CSV fails, try as MATLAB file
            import scipy.io as sio
            mat_file = str(mes_file).replace('.csv', '.mat')
            mat_data = sio.loadmat(mat_file)
            # Extract data from the MATLAB structure
            data = mat_data['TT_dailyMES']
            # Convert to DataFrame
            if isinstance(data, np.ndarray) and data.dtype.names:
                df = pd.DataFrame()
                for name in data.dtype.names:
                    df[name] = data[name].flatten()
                
                # Convert timestamp to datetime
                if 'TimestampCollected' in df.columns:
                    df['TimestampCollected'] = pd.to_datetime(df['TimestampCollected'])
            else:
                print("Unexpected format in MAT file")
                return None
            print(f"Loaded meteorological data from {mat_file}")
        except Exception as e:
            print(f"Error loading meteorological data: {e}")
            return None
    
    # Extract time information
    time_full = df['TimestampCollected']
    
    # Filter by date range
    date_start = datetime.strptime(date_start, '%Y-%m-%d')
    date_end = datetime.strptime(date_end, '%Y-%m-%d')
    
    # Find indices for date range
    is_d = np.where(time_full >= date_start)[0][0] if np.any(time_full >= date_start) else 0
    ie_d = np.where(time_full <= date_end)[0][-1] if np.any(time_full <= date_end) else len(time_full) - 1
    
    # Filter data
    tt = df.iloc[is_d:ie_d+1].copy()
    
    # Rename variables for consistency
    var_mapping = {
        'TAIR': 'TAIR_month',
        'TAIRx': 'TAIRx_month',
        'TAIRn': 'TAIRn_month',
        'DWPT': 'DWPT_month',
        'PRCP': 'PRCP_month',
        'PRES': 'PRES_month',
        'RELH': 'RELH_month',
        'SM02': 'SM02_month',
        'SRAD': 'SRAD_month',
        'WDIR': 'WDIR_month',
        'WSPD': 'WSPD_month',
        'WSMX': 'WSMX_month'
    }
    
    # Check if columns exist and rename
    rename_dict = {}
    for old_name, new_name in var_mapping.items():
        if old_name in tt.columns:
            rename_dict[old_name] = new_name
    
    if rename_dict:
        tt = tt.rename(columns=rename_dict)
    
    # Remove unnecessary variables if present
    cols_to_remove = ['SM20', 'SM04', 'SM40', 'SM08', 'ST02', 'ST20', 'ST04', 'ST40', 'ST08', 'WSMN']
    for col in cols_to_remove:
        if col in tt.columns:
            tt = tt.drop(columns=[col])
    
    # Extract time and unique years
    time = tt['TimestampCollected']
    years = np.unique(time.dt.year)
    
    # Define months
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    # Initialize monthly data structure
    data_month = {
        'year': years,
        'month': months,
        'var': list(rename_dict.values()),
        'data': [[] for _ in range(len(months))]
    }
    
    # Organize data by month and year
    for month_idx, month_name in enumerate(months):
        month_num = month_idx + 1
        month_data = []
        
        for year in years:
            # Filter data for this month and year
            mask = (time.dt.year == year) & (time.dt.month == month_num)
            month_year_data = tt[mask].copy() if np.any(mask) else None
            
            # If no data for this month/year, create empty dataframe with correct structure
            if month_year_data is None or len(month_year_data) == 0:
                # Try to find a template from adjacent year
                if year < years[-1]:
                    next_year = year + 1
                    next_mask = (time.dt.year == next_year) & (time.dt.month == month_num)
                    template = tt[next_mask].copy() if np.any(next_mask) else None
                else:
                    prev_year = year - 1
                    prev_mask = (time.dt.year == prev_year) & (time.dt.month == month_num)
                    template = tt[prev_mask].copy() if np.any(prev_mask) else None
                
                if template is not None and len(template) > 0:
                    # Create empty dataframe with same structure
                    month_year_data = template.copy()
                    for col in month_year_data.columns:
                        if col != 'TimestampCollected':
                            month_year_data[col] = np.nan
                    
                    # Adjust timestamps to current year
                    timestamps = month_year_data['TimestampCollected'].dt.to_pydatetime()
                    adjusted_timestamps = [dt.replace(year=year) for dt in timestamps]
                    month_year_data['TimestampCollected'] = adjusted_timestamps
                else:
                    # Create empty dataframe if no template is available
                    month_year_data = pd.DataFrame(columns=tt.columns)
            
            month_data.append(month_year_data)
        
        data_month['data'][month_idx] = month_data
    
    # Save data to CSV and HDF5
    output_dir = site_dir / 'processed'
    output_dir.mkdir(exist_ok=True, parents=True)
    output_csv = output_dir / f"{site}_DATAinput_monthly.csv"
    output_h5 = output_dir / f"{site}_DATAinput_monthly.h5"
    
    # Save metadata to CSV
    metadata = pd.DataFrame({
        'year': np.repeat(years, len(months)),
        'month': np.tile(months, len(years)),
        'var': [','.join(data_month['var'])] * (len(years) * len(months))
    })
    metadata.to_csv(output_csv, index=False)
    
    # Save each month/year as a separate CSV
    for month_idx, month_name in enumerate(months):
        for year_idx, year in enumerate(years):
            month_year_data = data_month['data'][month_idx][year_idx]
            if month_year_data is not None and len(month_year_data) > 0:
                month_year_file = output_dir / f"{site}_{year}_{month_name}.csv"
                month_year_data.to_csv(month_year_file, index=False)
    
    # Save to HDF5 format (more efficient for Python)
    with h5py.File(output_h5, 'w') as f:
        # Create groups
        years_group = f.create_group('years')
        months_group = f.create_group('months')
        vars_group = f.create_group('vars')
        data_group = f.create_group('data')
        
        # Store years, months, and variables
        years_group.create_dataset('values', data=years)
        months_group.create_dataset('names', data=np.array(months, dtype='S'))
        vars_group.create_dataset('names', data=np.array(data_month['var'], dtype='S'))
        
        # Store data for each month and year
        for month_idx, month_name in enumerate(months):
            month_group = data_group.create_group(month_name)
            
            for year_idx, year in enumerate(years):
                year_group = month_group.create_group(str(year))
                df = data_month['data'][month_idx][year_idx]
                
                if df is not None and len(df) > 0:
                    # Store timestamp as ISO format strings
                    timestamps = df['TimestampCollected'].dt.strftime('%Y-%m-%dT%H:%M:%S').values
                    year_group.create_dataset('TimestampCollected', data=np.array(timestamps, dtype='S'))
                    
                    # Store each variable
                    for var in rename_dict.values():
                        if var in df.columns:
                            year_group.create_dataset(var, data=df[var].values)
    
    print(f"Saved monthly data to {output_csv} and {output_h5}")
    
    return data_month

if __name__ == "__main__":
    # Example usage
    site = 'HCKM'
    file_mes = '01-Nov-2009_01-Aug-2023_HCKM_daily.csv'
    date_start = '2009-11-01'
    date_end = '2023-07-31'
    
    data = process_monthly_data(site, file_mes, date_start, date_end)
