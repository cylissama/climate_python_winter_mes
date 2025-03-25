"""
driverFULL_monthly.py

A Python implementation of the MATLAB script driverFULL_monthly.m.
This script handles monthly data processing and preparation for climate indices calculations.

Functions:
- Loads monthly meteorological data
- Processes data into usable structure
- Creates year and month groupings
- Prepares data for subsequent climate indices calculations

Author: Converted from MATLAB
Date: 2025-03-24
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from datetime import datetime
import h5py

def get_data_month(tt, month_mask, time):
    """
    Extract monthly data for all variables
    
    Parameters:
    -----------
    tt : DataFrame
        Full dataset
    month_mask : array-like
        Boolean mask for the month
    time : Series
        Timestamps
    
    Returns:
    --------
    tt_month : DataFrame
        Monthly data
    """
    time_month = time[month_mask]
    
    # List of variables to process
    variables = [
        'TAIR', 'DWPT', 'TAIRx', 'TAIRn', 'PRCP', 'RELH', 
        'PRES', 'SM02', 'WDIR', 'WSPD', 'WSMX', 'SRAD'
    ]
    
    # Create a new DataFrame for monthly data
    data_dict = {'TIME_month': time_month}
    
    # Extract each variable for the month
    for var in variables:
        if var in tt.columns:
            data_dict[var + '_month'] = tt[var][month_mask].values
    
    # Create DataFrame and set index
    tt_month = pd.DataFrame(data_dict)
    tt_month.set_index('TIME_month', inplace=True)
    
    return tt_month

def process_monthly_data(site='HCKM', file_name=None):
    """
    Process monthly meteorological data for climate indices calculation
    
    Parameters:
    -----------
    site : str
        Site code
    file_name : str, optional
        Input data file name. If None, uses a default pattern.
    
    Returns:
    --------
    my_struct_m : dict
        Dictionary containing processed monthly data
    """
    print(f"Processing monthly data for site {site}")
    
    # Default file name if not provided
    if file_name is None:
        file_name = f'01-Nov-2009_31-Jul-2023_{site}_daily.csv'
    
    # Base directory
    base_dir = Path('/Users/erappin/Documents/Mesonet/ClimateIndices/cliSITES/')
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
    data_file = base_dir / file_name
    try:
        # Try to load as CSV first
        df = pd.read_csv(data_file, parse_dates=['TimestampCollected'])
        print(f"Loaded data from {data_file}")
    except (FileNotFoundError, pd.errors.ParserError):
        try:
            # If CSV fails, try as MATLAB file
            import scipy.io as sio
            mat_file = str(data_file).replace('.csv', '.mat')
            mat_data = sio.loadmat(mat_file)
            # Extract data from the MATLAB structure
            tt_daily = mat_data['TT_dailyMES']
            # Convert to DataFrame (structure depends on MATLAB file)
            if isinstance(tt_daily, np.ndarray) and hasattr(tt_daily, 'dtype') and tt_daily.dtype.names:
                df = pd.DataFrame()
                for name in tt_daily.dtype.names:
                    df[name] = tt_daily[name].flatten()
                # Convert timestamp to datetime
                if 'TimestampCollected' in df.columns:
                    df['TimestampCollected'] = pd.to_datetime(df['TimestampCollected'])
            else:
                print("Unexpected format in MAT file")
                return None
            print(f"Loaded data from {mat_file}")
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    # Extract time information
    time = df['TimestampCollected']
    s_time = time.iloc[0]
    e_time = time.iloc[-1]
    
    # Get first and last years
    s_year, s_month, s_day = s_time.year, s_time.month, s_time.day
    e_year, e_month, e_day = e_time.year, e_time.month, e_time.day
    
    # Create date range for complete years
    date_values = pd.date_range(
        start=datetime(s_year, 1, 1),
        end=datetime(e_year, 12, 31),
        freq='D'
    )
    
    # Ensure data has complete date range (fill with zeros for missing dates)
    df_complete = df.set_index('TimestampCollected').reindex(date_values, fill_value=0)
    df_complete.reset_index(inplace=True)
    df_complete.rename(columns={'index': 'TimestampCollected'}, inplace=True)
    
    # Extract actual time range used
    time_full = df_complete['TimestampCollected']
    is_d = time_full[time_full == s_time].index[0] if np.any(time_full == s_time) else 0
    ie_d = time_full[time_full == e_time].index[0] if np.any(time_full == e_time) else len(time_full) - 1
    
    # Get unique years
    all_years = np.unique(time_full.dt.year)
    
    # Set up structure for monthly data
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    variables = [
        "TAIR_month", "DWPT_month", "TAIRx_month", "TAIRn_month", "PRCP_month", "RELH_month",
        "PRES_month", "SM02_month", "WDIR_month", "WSPD_month", "WSMX_month", "SRAD_month"
    ]
    
    my_struct_m = {
        'year': all_years,
        'month': months,
        'var': variables,
        'data': [[] for _ in range(len(months))],
        'years': [[] for _ in range(len(months))]
    }
    
    # Process data by month
    for i, month in enumerate(range(1, 13)):
        # Create month mask
        month_mask = time_full.dt.month == month
        month_data = df_complete[month_mask]
        
        # Group by year for this month
        year_groups = time_full[month_mask].dt.year
        unique_years = np.unique(year_groups)
        
        # Process each year for this month
        for j, year in enumerate(unique_years):
            # Create mask for this month and year
            year_mask = year_groups == year
            
            # Get data for this month and year
            tt_month = get_data_month(df_complete, month_mask & (time_full.dt.year == year), time_full)
            
            # Store in structure
            if j >= len(my_struct_m['data'][i]):
                my_struct_m['data'][i].append(tt_month)
                my_struct_m['years'][i].append(year)
            else:
                my_struct_m['data'][i][j] = tt_month
                my_struct_m['years'][i][j] = year
    
    # Save data to file
    output_dir = site_dir / 'processed'
    output_dir.mkdir(exist_ok=True, parents=True)
    output_h5 = output_dir / f"{site}_monthly_processed.h5"
    
    with h5py.File(output_h5, 'w') as f:
        # Store metadata
        f.create_dataset('years', data=all_years)
        f.attrs['months'] = np.array(months, dtype='S')
        f.attrs['variables'] = np.array(variables, dtype='S')
        
        # Create group for data
        data_group = f.create_group('data')
        
        # Store data for each month
        for i, month in enumerate(months):
            month_group = data_group.create_group(month)
            
            for j, year in enumerate(my_struct_m['years'][i]):
                if j < len(my_struct_m['data'][i]):
                    year_group = month_group.create_group(str(year))
                    df = my_struct_m['data'][i][j]
                    
                    if df is not None and not df.empty:
                        # Store timestamp
                        timestamps = df.index.strftime('%Y-%m-%dT%H:%M:%S').values
                        year_group.create_dataset('timestamps', data=np.array(timestamps, dtype='S'))
                        
                        # Store variables
                        for var in df.columns:
                            year_group.create_dataset(var, data=df[var].values)
    
    print(f"Saved processed monthly data to {output_h5}")
    
    return my_struct_m

if __name__ == "__main__":
    # Example usage
    site = 'HCKM'
    data = process_monthly_data(site)
