"""
driverDATA_annual.py

This script processes meteorological data from Mesonet stations and 
organizes it into an annual data structure. It is a Python implementation 
of the MATLAB script driverDATA_annual.m.

Functions:
- Loads daily meteorological data from CSV files
- Filters data by date range
- Reorganizes data into annual structures
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

def process_annual_data(site='BMTN', 
                       file_mes='01-Mar-2014_01-Aug-2023_BMTN_daily.csv',
                       date_start='2015-01-01',
                       date_end='2022-12-31'):
    """
    Process daily meteorological data and organize by year
    
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
    data_annual : dict
        Dictionary with organized annual data
    """
    print(f"Processing annual data for site {site}")
    
    # Base directory
    data_dir = Path(f'/Volumes/Mesonet/winter_break/CCdata/')
    site_dir = data_dir / site
    
    # Load threshold file
    thresh_file = site_dir / f"{site}_CLIthresh_daily.csv"
    try:
        thresh_data = pd.read_csv(thresh_file)
        print(f"Loaded threshold data from {thresh_file}")
    except FileNotFoundError:
        print(f"Threshold file not found: {thresh_file}")
        thresh_data = None
    
    # Load meteorological data
    mes_file = data_dir / site / file_mes
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
            # This depends on exact structure of the .mat file
            data = mat_data['TT_dailyMES']
            # Convert to DataFrame (assumes specific structure in MATLAB file)
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
        'TAIR': 'TAIR_annual',
        'TAIRx': 'TAIRx_annual',
        'TAIRn': 'TAIRn_annual',
        'DWPT': 'DWPT_annual',
        'PRCP': 'PRCP_annual',
        'PRES': 'PRES_annual',
        'RELH': 'RELH_annual',
        'SM02': 'SM02_annual',
        'SRAD': 'SRAD_annual',
        'WDIR': 'WDIR_annual',
        'WSPD': 'WSPD_annual',
        'WSMX': 'WSMX_annual'
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
    
    # Initialize annual data structure
    data_annual = {
        'year': years,
        'var': list(rename_dict.values()),
        'data': []
    }
    
    # Organize data by year
    for k, year in enumerate(years):
        year_mask = (time.dt.year == year)
        data_annual['data'].append(tt[year_mask].copy())
    
    # Save data to CSV and HDF5
    output_dir = site_dir / 'processed'
    output_dir.mkdir(exist_ok=True, parents=True)
    output_csv = output_dir / f"{site}_DATAinput_annual.csv"
    output_h5 = output_dir / f"{site}_DATAinput_annual.h5"
    
    # Save metadata and yearly data to separate CSV files
    metadata = pd.DataFrame({
        'year': years,
        'var': [','.join(data_annual['var'])] * len(years)
    })
    metadata.to_csv(output_csv, index=False)
    
    # Save each year as a separate CSV
    for i, year_data in enumerate(data_annual['data']):
        year_file = output_dir / f"{site}_{years[i]}_annual.csv"
        year_data.to_csv(year_file, index=False)
    
    # Save to HDF5 format (more efficient for Python)
    with h5py.File(output_h5, 'w') as f:
        # Create groups
        years_group = f.create_group('years')
        vars_group = f.create_group('vars')
        data_group = f.create_group('data')
        
        # Store years and variables
        years_group.create_dataset('values', data=years)
        vars_group.create_dataset('names', data=np.array(data_annual['var'], dtype='S'))
        
        # Store data for each year
        for i, year in enumerate(years):
            year_group = data_group.create_group(str(year))
            df = data_annual['data'][i]
            
            # Store timestamp as ISO format strings
            timestamps = df['TimestampCollected'].dt.strftime('%Y-%m-%dT%H:%M:%S').values
            year_group.create_dataset('TimestampCollected', data=np.array(timestamps, dtype='S'))
            
            # Store each variable
            for var in rename_dict.values():
                if var in df.columns:
                    year_group.create_dataset(var, data=df[var].values)
    
    print(f"Saved annual data to {output_csv} and {output_h5}")
    
    return data_annual

if __name__ == "__main__":
    # Example usage
    site = 'BMTN'
    file_mes = '01-Mar-2014_01-Aug-2023_BMTN_daily.csv'
    date_start = '2015-01-01'
    date_end = '2022-12-31'
    
    data = process_annual_data(site, file_mes, date_start, date_end)
