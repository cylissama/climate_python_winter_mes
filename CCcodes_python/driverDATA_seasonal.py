"""
driverDATA_seasonal.py

This script processes meteorological data from Mesonet stations and 
organizes it into a seasonal data structure. It is a Python implementation 
of the MATLAB script driverDATA_seasonal.m.

Functions:
- Loads daily meteorological data from CSV files
- Filters data by date range
- Reorganizes data into seasonal structures
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

def process_seasonal_data(site='HCKM', 
                        file_mes='01-Nov-2009_01-Aug-2023_HCKM_daily.csv',
                        date_start='2009-12-01',
                        date_end='2023-05-31'):
    """
    Process daily meteorological data and organize by season
    
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
    data_season : dict
        Dictionary with organized seasonal data
    """
    print(f"Processing seasonal data for site {site}")
    
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
        'TAIR': 'TAIR_season',
        'TAIRx': 'TAIRx_season',
        'TAIRn': 'TAIRn_season',
        'DWPT': 'DWPT_season',
        'PRCP': 'PRCP_season',
        'PRES': 'PRES_season',
        'RELH': 'RELH_season',
        'SM02': 'SM02_season',
        'SRAD': 'SRAD_season',
        'WDIR': 'WDIR_season',
        'WSPD': 'WSPD_season',
        'WSMX': 'WSMX_season'
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
    
    # Define seasons
    seasons = ["Winter", "Spring", "Summer", "Fall"]
    
    # Initialize seasonal data structure
    data_season = {
        'year': years,
        'season': seasons,
        'var': list(rename_dict.values()),
        'data': [[] for _ in range(len(seasons))]
    }
    
    # Create season masks
    spring_mask = time.dt.month.isin([3, 4, 5])
    summer_mask = time.dt.month.isin([6, 7, 8])
    fall_mask = time.dt.month.isin([9, 10, 11])
    winter_mask = time.dt.month.isin([12, 1, 2])
    
    # Adjust winter year (December is part of next year's winter)
    # For winter, we need to create a winter_year that assigns December to the following year
    winter_year = time.dt.year.copy()
    december_mask = time.dt.month == 12
    winter_year[december_mask] += 1
    
    # Extract data by season
    spring_data = tt[spring_mask].copy()
    summer_data = tt[summer_mask].copy()
    fall_data = tt[fall_mask].copy()
    winter_data = tt[winter_mask].copy()
    
    # Organize data by season and year
    for year in years:
        # Process winter data
        # For winter, we need to handle December of the previous year
        # and January/February of the current year
        winter_year_mask = winter_year == year
        winter_year_data = winter_data[winter_year_mask].copy() if np.any(winter_year_mask) else None
        
        # Process other seasons (simpler)
        year_mask = time.dt.year == year
        spring_year_mask = spring_mask & year_mask
        summer_year_mask = summer_mask & year_mask
        fall_year_mask = fall_mask & year_mask
        
        spring_year_data = spring_data[spring_year_mask].copy() if np.any(spring_year_mask) else None
        summer_year_data = summer_data[summer_year_mask].copy() if np.any(summer_year_mask) else None
        fall_year_data = fall_data[fall_year_mask].copy() if np.any(fall_year_mask) else None
        
        # Handle empty dataframes
        for season_idx, season_data in enumerate([winter_year_data, spring_year_data, summer_year_data, fall_year_data]):
            if season_data is None or len(season_data) == 0:
                # Try to find a template from adjacent year
                if year < years[-1]:
                    next_year = year + 1
                    if season_idx == 0:  # Winter
                        next_mask = winter_year == next_year
                        template = winter_data[next_mask].copy() if np.any(next_mask) else None
                    else:
                        next_mask = (time.dt.year == next_year) & [spring_mask, summer_mask, fall_mask][season_idx-1]
                        season_next_data = [spring_data, summer_data, fall_data][season_idx-1]
                        template = season_next_data[next_mask].copy() if np.any(next_mask) else None
                else:
                    prev_year = year - 1
                    if season_idx == 0:  # Winter
                        prev_mask = winter_year == prev_year
                        template = winter_data[prev_mask].copy() if np.any(prev_mask) else None
                    else:
                        prev_mask = (time.dt.year == prev_year) & [spring_mask, summer_mask, fall_mask][season_idx-1]
                        season_prev_data = [spring_data, summer_data, fall_data][season_idx-1]
                        template = season_prev_data[prev_mask].copy() if np.any(prev_mask) else None
                
                if template is not None and len(template) > 0:
                    # Create empty dataframe with same structure
                    if season_idx == 0:
                        winter_year_data = template.copy()
                        for col in winter_year_data.columns:
                            if col != 'TimestampCollected':
                                winter_year_data[col] = np.nan
                    elif season_idx == 1:
                        spring_year_data = template.copy()
                        for col in spring_year_data.columns:
                            if col != 'TimestampCollected':
                                spring_year_data[col] = np.nan
                    elif season_idx == 2:
                        summer_year_data = template.copy()
                        for col in summer_year_data.columns:
                            if col != 'TimestampCollected':
                                summer_year_data[col] = np.nan
                    elif season_idx == 3:
                        fall_year_data = template.copy()
                        for col in fall_year_data.columns:
                            if col != 'TimestampCollected':
                                fall_year_data[col] = np.nan
                    
                    # Adjust timestamps to current year
                    timestamps = template['TimestampCollected'].dt.to_pydatetime()
                    adjusted_timestamps = [dt.replace(year=year) for dt in timestamps]
                    
                    if season_idx == 0:
                        winter_year_data['TimestampCollected'] = adjusted_timestamps
                    elif season_idx == 1:
                        spring_year_data['TimestampCollected'] = adjusted_timestamps
                    elif season_idx == 2:
                        summer_year_data['TimestampCollected'] = adjusted_timestamps
                    elif season_idx == 3:
                        fall_year_data['TimestampCollected'] = adjusted_timestamps
        
        # Store season data for this year
        data_season['data'][0].append(winter_year_data)
        data_season['data'][1].append(spring_year_data)
        data_season['data'][2].append(summer_year_data)
        data_season['data'][3].append(fall_year_data)
    
    # Save data to CSV and HDF5
    output_dir = site_dir / 'processed'
    output_dir.mkdir(exist_ok=True, parents=True)
    output_csv = output_dir / f"{site}_DATAinput_seasonal.csv"
    output_h5 = output_dir / f"{site}_DATAinput_seasonal.h5"
    
    # Save metadata to CSV
    metadata = pd.DataFrame({
        'year': np.repeat(years, len(seasons)),
        'season': np.tile(seasons, len(years)),
        'var': [','.join(data_season['var'])] * (len(years) * len(seasons))
    })
    metadata.to_csv(output_csv, index=False)
    
    # Save each season/year as a separate CSV
    for season_idx, season_name in enumerate(seasons):
        for year_idx, year in enumerate(years):
            season_year_data = data_season['data'][season_idx][year_idx]
            if season_year_data is not None and len(season_year_data) > 0:
                season_year_file = output_dir / f"{site}_{year}_{season_name}.csv"
                season_year_data.to_csv(season_year_file, index=False)
    
    # Save to HDF5 format (more efficient for Python)
    with h5py.File(output_h5, 'w') as f:
        # Create groups
        years_group = f.create_group('years')
        seasons_group = f.create_group('seasons')
        vars_group = f.create_group('vars')
        data_group = f.create_group('data')
        
        # Store years, seasons, and variables
        years_group.create_dataset('values', data=years)
        seasons_group.create_dataset('names', data=np.array(seasons, dtype='S'))
        vars_group.create_dataset('names', data=np.array(data_season['var'], dtype='S'))
        
        # Store data for each season and year
        for season_idx, season_name in enumerate(seasons):
            season_group = data_group.create_group(season_name)
            
            for year_idx, year in enumerate(years):
                year_group = season_group.create_group(str(year))
                df = data_season['data'][season_idx][year_idx]
                
                if df is not None and len(df) > 0:
                    # Store timestamp as ISO format strings
                    timestamps = df['TimestampCollected'].dt.strftime('%Y-%m-%dT%H:%M:%S').values
                    year_group.create_dataset('TimestampCollected', data=np.array(timestamps, dtype='S'))
                    
                    # Store each variable
                    for var in rename_dict.values():
                        if var in df.columns:
                            year_group.create_dataset(var, data=df[var].values)
    
    print(f"Saved seasonal data to {output_csv} and {output_h5}")
    
    return data_season

if __name__ == "__main__":
    # Example usage
    site = 'HCKM'
    file_mes = '01-Nov-2009_01-Aug-2023_HCKM_daily.csv'
    date_start = '2009-12-01'
    date_end = '2023-05-31'
    
    data = process_seasonal_data(site, file_mes, date_start, date_end)