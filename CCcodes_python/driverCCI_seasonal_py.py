"""
driverCCI_seasonal.py

This script processes climate data and calculates seasonal climate indices.
It is a Python implementation of the MATLAB script driverCCI_seasonal.m.

Functions:
- Processes daily meteorological data by season
- Calculates climate indices for each season and year
- Handles empty data frames and missing values
- Generates seasonal statistics

Author: Converted from MATLAB
Date: 2025-03-24
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from datetime import datetime, timedelta
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def run_length(arr):
    """
    Calculate run lengths of True values in boolean array
    
    Parameters:
    -----------
    arr : array-like
        Boolean array
        
    Returns:
    --------
    B : array
        Boolean values (True for runs)
    N : array
        Length of each run
    BI : array
        Starting indices of runs
    """
    # Convert to numpy array if not already
    arr = np.asarray(arr)
    
    # Find run starts and ends
    diff = np.diff(np.concatenate(([False], arr, [False])))
    run_starts = np.where(diff > 0)[0]
    run_ends = np.where(diff < 0)[0]
    
    # Calculate run lengths
    N = run_ends - run_starts
    B = np.ones_like(N, dtype=bool)
    BI = run_starts
    
    return B, N, BI

def calculate_climate_indices(site='FARM', start_date='2008-03-01', end_date='2023-05-31'):
    """
    Calculate seasonal climate indices
    
    Parameters:
    -----------
    site : str
        Site code
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
        
    Returns:
    --------
    data_structure : dict
        Dictionary containing all calculated indices
    """
    print(f"Processing seasonal climate indices for site {site}")
    
    # Base directory
    base_dir = Path(f'/Users/erappin/Documents/Mesonet/ClimateIndices/sitesTEST_CCindices/{site}')
    
    # Load threshold data
    try:
        thresh_file = base_dir / f"{site}_CLIthresh_daily.csv"
        thresh_data = pd.read_csv(thresh_file)
        print(f"Loaded threshold data from {thresh_file}")
    except Exception as e:
        print(f"Error loading threshold data: {e}")
        return None
    
    # Load seasonal data
    try:
        seasonal_file = base_dir / f"{site}_DATAinput_seasonal.csv"
        seasonal_data = pd.read_csv(seasonal_file)
        print(f"Loaded seasonal data from {seasonal_file}")
    except Exception as e:
        print(f"Error loading seasonal data: {e}")
        return None
    
    # Extract years
    start_year = datetime.strptime(start_date, '%Y-%m-%d').year
    end_year = datetime.strptime(end_date, '%Y-%m-%d').year
    years = list(range(start_year, end_year + 1))
    print(f"Processing data for years: {start_year} to {end_year}")
    
    # Define seasons
    seasons = ["Winter", "Spring", "Summer", "Fall"]
    
    # Initialize data structure
    data_structure = {
        'site': site,
        'years': years,
        'seasons': seasons,
        'data': [],
        'indices': {}
    }
    
    # Create season data structure
    season_data = {}
    for season_idx, season in enumerate(seasons):
        season_data[season] = {}
        for year in years:
            # Create start and end dates for each season
            if season == "Winter":
                start = datetime(year-1, 12, 1) if year > start_year else datetime(year, 1, 1)
                end = datetime(year, 2, 28)
            elif season == "Spring":
                start = datetime(year, 3, 1)
                end = datetime(year, 5, 31)
            elif season == "Summer":
                start = datetime(year, 6, 1)
                end = datetime(year, 8, 31)
            elif season == "Fall":
                start = datetime(year, 9, 1)
                end = datetime(year, 11, 30)
            
            # Filter data for this season and year
            season_data[season][year] = seasonal_data[
                (seasonal_data['date'] >= start.strftime('%Y-%m-%d')) & 
                (seasonal_data['date'] <= end.strftime('%Y-%m-%d'))
            ].copy()
            
            # Handle empty dataframes
            if len(season_data[season][year]) == 0:
                if year < end_year:
                    template = season_data[season][year + 1].copy() if year + 1 in season_data[season] else None
                else:
                    template = season_data[season][year - 1].copy() if year - 1 in season_data[season] else None
                
                if template is not None:
                    season_data[season][year] = template.copy()
                    season_data[season][year][:] = np.nan
                    season_data[season][year]['date'] = pd.date_range(start, end)
    
    # Initialize climate indices
    indices = {}
    
    # Define season lengths
    season_days = {"Winter": 90, "Spring": 92, "Summer": 92, "Fall": 91}
    
    # ======= CALCULATE CLIMATE INDICES =======
    
    # Growing Degree Days (GD4, GD10)
    GD4 = np.zeros((len(years), len(seasons)))
    GD10 = np.zeros((len(years), len(seasons)))
    
    for i, season in enumerate(seasons):
        for j, year in enumerate(years):
            if season in season_data and year in season_data[season]:
                df = season_data[season][year]
                if 'TAIR' in df.columns:
                    TAIR = df['TAIR'].values
                    GD4[j, i] = np.sum(np.maximum(TAIR - 4, 0))
                    GD10[j, i] = np.sum(np.maximum(TAIR - 10, 0))
    
    indices['GD4'] = GD4
    indices['GD10'] = GD10
    
    # Frost Days (FD)
    FD = np.zeros((len(years), len(seasons)))
    
    for i, season in enumerate(seasons):
        for j, year in enumerate(years):
            if season in season_data and year in season_data[season]:
                df = season_data[season][year]
                if 'TAIRn' in df.columns:
                    TAIRn = df['TAIRn'].values
                    FD[j, i] = np.sum(TAIRn < 0)
    
    indices['FD'] = FD
    
    # Consecutive Frost Days (CFD)
    CFD = np.zeros((len(years), len(seasons)))
    
    for i, season in enumerate(seasons):
        for j, year in enumerate(years):
            if season in season_data and year in season_data[season]:
                df = season_data[season][year]
                if 'TAIRn' in df.columns:
                    TAIRn = df['TAIRn'].values
                    B, N, BI = run_length(TAIRn < 0)
                    if len(N) > 0:
                        CFD[j, i] = np.max(N)
    
    indices['CFD'] = CFD
    
    # Heating Degree Days (HDD)
    HDD = np.zeros((len(years), len(seasons)))
    
    for i, season in enumerate(seasons):
        for j, year in enumerate(years):
            if season in season_data and year in season_data[season]:
                df = season_data[season][year]
                if 'TAIR' in df.columns:
                    TAIR = df['TAIR'].values
                    HDD[j, i] = np.sum(np.maximum(18.3 - TAIR, 0))
    
    indices['HDD'] = HDD
    
    # Ice Days (ID)
    ID = np.zeros((len(years), len(seasons)))
    
    for i, season in enumerate(seasons):
        for j, year in enumerate(years):
            if season in season_data and year in season_data[season]:
                df = season_data[season][year]
                if 'TAIRx' in df.columns:
                    TAIRx = df['TAIRx'].values
                    ID[j, i] = np.sum(TAIRx < 0)
    
    indices['ID'] = ID
    
    # Minimum Value of Daily Maximum Temperature (TXn)
    TXn = np.full((len(years), len(seasons)), np.nan)
    
    for i, season in enumerate(seasons):
        for j, year in enumerate(years):
            if season in season_data and year in season_data[season]:
                df = season_data[season][year]
                if 'TAIRx' in df.columns:
                    TAIRx = df['TAIRx'].values
                    if np.any(~np.isnan(TAIRx)):
                        TXn[j, i] = np.nanmin(TAIRx)
    
    indices['TXn'] = TXn
    
    # Minimum Value of Daily Minimum Temperature (TNn)
    TNn = np.full((len(years), len(seasons)), np.nan)
    
    for i, season in enumerate(seasons):
        for j, year in enumerate(years):
            if season in season_data and year in season_data[season]:
                df = season_data[season][year]
                if 'TAIRn' in df.columns:
                    TAIRn = df['TAIRn'].values
                    if np.any(~np.isnan(TAIRn)):
                        TNn[j, i] = np.nanmin(TAIRn)
    
    indices['TNn'] = TNn
    
    # Summer Days (SU)
    SU = np.zeros((len(years), len(seasons)))
    
    for i, season in enumerate(seasons):
        for j, year in enumerate(years):
            if season in season_data and year in season_data[season]:
                df = season_data[season][year]
                if 'TAIRx' in df.columns:
                    TAIRx = df['TAIRx'].values
                    SU[j, i] = np.sum(TAIRx > 25)
    
    indices['SU'] = SU
    
    # Maximum Number of Consecutive Summer Days (CSU)
    CSU = np.zeros((len(years), len(seasons)))
    
    for i, season in enumerate(seasons):
        for j, year in enumerate(years):
            if season in season_data and year in season_data[season]:
                df = season_data[season][year]
                if 'TAIRx' in df.columns:
                    TAIRx = df['TAIRx'].values
                    B, N, BI = run_length(TAIRx > 25)
                    if len(N) > 0:
                        CSU[j, i] = np.max(N)
    
    indices['CSU'] = CSU
    
    # Tropical Nights (TR)
    TR = np.zeros((len(years), len(seasons)))
    
    for i, season in enumerate(seasons):
        for j, year in enumerate(years):
            if season in season_data and year in season_data[season]:
                df = season_data[season][year]
                if 'TAIRn' in df.columns:
                    TAIRn = df['TAIRn'].values
                    TR[j, i] = np.sum(TAIRn > 20)
    
    indices['TR'] = TR
    
    # Maximum Value of Daily Maximum Temperature (TXx)
    TXx = np.full((len(years), len(seasons)), np.nan)
    
    for i, season in enumerate(seasons):
        for j, year in enumerate(years):
            if season in season_data and year in season_data[season]:
                df = season_data[season][year]
                if 'TAIRx' in df.columns:
                    TAIRx = df['TAIRx'].values
                    if np.any(~np.isnan(TAIRx)):
                        TXx[j, i] = np.nanmax(TAIRx)
    
    indices['TXx'] = TXx
    
    # Maximum Value of Daily Minimum Temperature (TNx)
    TNx = np.full((len(years), len(seasons)), np.nan)
    
    for i, season in enumerate(seasons):
        for j, year in enumerate(years):
            if season in season_data and year in season_data[season]:
                df = season_data[season][year]
                if 'TAIRn' in df.columns:
                    TAIRn = df['TAIRn'].values
                    if np.any(~np.isnan(TAIRn)):
                        TNx[j, i] = np.nanmax(TAIRn)
    
    indices['TNx'] = TNx
    
    # Mean Relative Humidity (RH)
    RH = np.full((len(years), len(seasons)), np.nan)
    
    for i, season in enumerate(seasons):
        for j, year in enumerate(years):
            if season in season_data and year in season_data[season]:
                df = season_data[season][year]
                if 'RELH' in df.columns:
                    RELH = df['RELH'].values
                    RH[j, i] = np.nanmean(RELH)
    
    indices['RH'] = RH
    
    # Precipitation Sum (RR)
    RR = np.full((len(years), len(seasons)), np.nan)
    
    for i, season in enumerate(seasons):
        for j, year in enumerate(years):
            if season in season_data and year in season_data[season]:
                df = season_data[season][year]
                if 'PRCP' in df.columns:
                    PRCP = df['PRCP'].values
                    RR[j, i] = np.nansum(PRCP)
    
    indices['RR'] = RR
    
    # Wet Days (RR1) - Days with precipitation >= 1mm
    RR1 = np.zeros((len(years), len(seasons)))
    
    for i, season in enumerate(seasons):
        for j, year in enumerate(years):
            if season in season_data and year in season_data[season]:
                df = season_data[season][year]
                if 'PRCP' in df.columns:
                    PRCP = df['PRCP'].values
                    RR1[j, i] = np.sum(PRCP >= 1)
    
    indices['RR1'] = RR1
    
    # Simple Daily Intensity Index (SDII)
    SDII = np.full((len(years), len(seasons)), np.nan)
    
    for i, season in enumerate(seasons):
        for j, year in enumerate(years):
            if season in season_data and year in season_data[season]:
                df = season_data[season][year]
                if 'PRCP' in df.columns:
                    PRCP = df['PRCP'].values
                    wet_days = PRCP[PRCP >= 1]
                    if len(wet_days) > 0:
                        SDII[j, i] = np.nansum(wet_days) / len(wet_days)
    
    indices['SDII'] = SDII
    
    # Maximum Number of Consecutive Wet Days (CWD)
    CWD = np.zeros((len(years), len(seasons)))
    
    for i, season in enumerate(seasons):
        for j, year in enumerate(years):
            if season in season_data and year in season_data[season]:
                df = season_data[season][year]
                if 'PRCP' in df.columns:
                    PRCP = df['PRCP'].values
                    B, N, BI = run_length(PRCP > 1)
                    if len(N) > 0:
                        CWD[j, i] = np.max(N)
    
    indices['CWD'] = CWD
    
    # Heavy Precipitation Days (RR10)
    RR10 = np.zeros((len(years), len(seasons)))
    
    for i, season in enumerate(seasons):
        for j, year in enumerate(years):
            if season in season_data and year in season_data[season]:
                df = season_data[season][year]
                if 'PRCP' in df.columns:
                    PRCP = df['PRCP'].values
                    RR10[j, i] = np.sum(PRCP > 10)
    
    indices['RR10'] = RR10
    
    # Very Heavy Precipitation Days (RR20)
    RR20 = np.zeros((len(years), len(seasons)))
    
    for i, season in enumerate(seasons):
        for j, year in enumerate(years):
            if season in season_data and year in season_data[season]:
                df = season_data[season][year]
                if 'PRCP' in df.columns:
                    PRCP = df['PRCP'].values
                    RR20[j, i] = np.sum(PRCP > 20)
    
    indices['RR20'] = RR20
    
    # Highest 1-Day Precipitation Amount (RX1day)
    RX1day = np.full((len(years), len(seasons)), np.nan)
    
    for i, season in enumerate(seasons):
        for j, year in enumerate(years):
            if season in season_data and year in season_data[season]:
                df = season_data[season][year]
                if 'PRCP' in df.columns:
                    PRCP = df['PRCP'].values
                    if np.any(~np.isnan(PRCP)):
                        RX1day[j, i] = np.nanmax(PRCP)
    
    indices['RX1day'] = RX1day
    
    # Mean of Daily Mean Air Temperature (TG)
    TG = np.full((len(years), len(seasons)), np.nan)
    
    for i, season in enumerate(seasons):
        for j, year in enumerate(years):
            if season in season_data and year in season_data[season]:
                df = season_data[season][year]
                if 'TAIR' in df.columns:
                    TAIR = df['TAIR'].values
                    TG[j, i] = np.nanmean(TAIR)
    
    indices['TG'] = TG
    
    # Mean of Daily Minimum Air Temperature (TN)
    TN = np.full((len(years), len(seasons)), np.nan)
    
    for i, season in enumerate(seasons):
        for j, year in enumerate(years):
            if season in season_data and year in season_data[season]:
                df = season_data[season][year]
                if 'TAIRn' in df.columns:
                    TAIRn = df['TAIRn'].values
                    TN[j, i] = np.nanmean(TAIRn)
    
    indices['TN'] = TN
    
    # Mean of Daily Maximum Air Temperature (TX)
    TX = np.full((len(years), len(seasons)), np.nan)
    
    for i, season in enumerate(seasons):
        for j, year in enumerate(years):
            if season in season_data and year in season_data[season]:
                df = season_data[season][year]
                if 'TAIRx' in df.columns:
                    TAIRx = df['TAIRx'].values
                    TX[j, i] = np.nanmean(TAIRx)
    
    indices['TX'] = TX
    
    # Mean of Diurnal Air Temperature Range (DTR)
    DTR = np.full((len(years), len(seasons)), np.nan)
    
    for i, season in enumerate(seasons):
        for j, year in enumerate(years):
            if season in season_data and year in season_data[season]:
                df = season_data[season][year]
                if 'TAIRx' in df.columns and 'TAIRn' in df.columns:
                    TAIRx = df['TAIRx'].values
                    TAIRn = df['TAIRn'].values
                    daily_range = TAIRx - TAIRn
                    DTR[j, i] = np.nanmean(daily_range)
    
    indices['DTR'] = DTR
    
    # Intra-Period Extreme Temperature Range (ETR)
    ETR = np.full((len(years), len(seasons)), np.nan)
    
    for i, season in enumerate(seasons):
        for j, year in enumerate(years):
            if season in season_data and year in season_data[season]:
                df = season_data[season][year]
                if 'TAIRx' in df.columns and 'TAIRn' in df.columns:
                    TAIRx = df['TAIRx'].values
                    TAIRn = df['TAIRn'].values
                    if np.any(~np.isnan(TAIRx)) and np.any(~np.isnan(TAIRn)):
                        ETR[j, i] = np.nanmax(TAIRx) - np.nanmin(TAIRn)
    
    indices['ETR'] = ETR
    
    # Maximum Daily Wind Speed Gust (FXx)
    FXx = np.full((len(years), len(seasons)), np.nan)
    
    for i, season in enumerate(seasons):
        for j, year in enumerate(years):
            if season in season_data and year in season_data[season]:
                df = season_data[season][year]
                if 'WSMX' in df.columns:
                    WSMX = df['WSMX'].values
                    if np.any(~np.isnan(WSMX)):
                        # Convert from m/s to mph
                        FXx[j, i] = np.nanmax(WSMX) * 2.23694
    
    indices['FXx'] = FXx
    
    # Mean of Daily Mean Wind Strength (FG)
    FG = np.full((len(years), len(seasons)), np.nan)
    
    for i, season in enumerate(seasons):
        for j, year in enumerate(years):
            if season in season_data and year in season_data[season]:
                df = season_data[season][year]
                if 'WSPD' in df.columns:
                    WSPD = df['WSPD'].values
                    FG[j, i] = np.nanmean(WSPD)
    
    indices['FG'] = FG
    
    # Wind Direction calculations
    DDsouth = np.zeros((len(years), len(seasons)))
    DDeast = np.zeros((len(years), len(seasons)))
    DDwest = np.zeros((len(years), len(seasons)))
    DDnorth = np.zeros((len(years), len(seasons)))
    
    for i, season in enumerate(seasons):
        for j, year in enumerate(years):
            if season in season_data and year in season_data[season]:
                df = season_data[season][year]
                if 'WDIR' in df.columns:
                    WDIR = df['WDIR'].values
                    # Days with Southerly Winds
                    DDsouth[j, i] = np.sum((WDIR <= 225) & (WDIR > 135))
                    # Days with Easterly Winds
                    DDeast[j, i] = np.sum((WDIR <= 135) & (WDIR > 45))
                    # Days with Westerly Winds
                    DDwest[j, i] = np.sum((WDIR <= 315) & (WDIR > 225))
                    # Days with Northerly Winds
                    DDnorth[j, i] = np.sum((WDIR <= 45) | (WDIR > 315))
    
    indices['DDsouth'] = DDsouth
    indices['DDeast'] = DDeast
    indices['DDwest'] = DDwest
    indices['DDnorth'] = DDnorth
    
    # Add indices to data structure
    data_structure['indices'] = indices
    
    # Save data structure
    output_dir = base_dir / 'output'
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{site}_seasonal_indices.csv"
    
    # Convert indices to a DataFrame for easy saving
    indices_df = pd.DataFrame()
    for idx_name, idx_values in indices.items():
        for season_idx, season in enumerate(seasons):
            col_name = f"{idx_name}_{season}"
            for year_idx, year in enumerate(years):
                if indices_df.empty:
                    indices_df = pd.DataFrame({'Year': years})
                indices_df.loc[year_idx, col_name] = idx_values[year_idx, season_idx]
    
    indices_df.to_csv(output_file, index=False)
    print(f"Saved seasonal climate indices to {output_file}")
    
    return data_structure

if __name__ == "__main__":
    # Example usage
    site = 'FARM'
    start_date = '2008-03-01'
    end_date = '2023-05-31'
    data = calculate_climate_indices(site, start_date, end_date)
