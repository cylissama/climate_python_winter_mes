#!/usr/bin/env python3
"""
Climate Indices Calculator

This script calculates various climate indices based on meteorological data.
It is a Python conversion of the MATLAB script 'driverCCI_seasonal.m'.

The climate indices are calculated according to the European Climate Assessment & Dataset
(ECA&D) project's Algorithm Theoretical Basis Document (ATBD).

Requirements:
- numpy
- pandas
- scipy

Author: Converted from MATLAB by Claude
Date: March 24, 2025
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats


def run_length(arr):
    """
    Calculates run lengths in a boolean array.

    Parameters:
    -----------
    arr : ndarray
        Boolean array

    Returns:
    --------
    B : ndarray
        Boolean array indicating where runs occur
    N : ndarray
        Length of each run
    BI : ndarray
        Indices where runs begin
    """
    # Make sure input is a numpy array
    arr = np.asarray(arr)

    # Find the start and end of runs
    diffs = np.diff(np.concatenate(([0], arr.astype(int), [0])))
    run_starts = np.where(diffs > 0)[0]
    run_ends = np.where(diffs < 0)[0]

    # Calculate run lengths
    N = run_ends - run_starts

    # Create output arrays
    B = np.zeros_like(arr, dtype=bool)
    for start, length in zip(run_starts, N):
        B[start:start + length] = True

    return B, N, run_starts


def pet(Ra, tmax, tmin, tmean):
    """
    Calculate Potential EvapoTranspiration using Penman-Monteith equation

    Parameters:
    -----------
    Ra : ndarray
        Net radiation at crop surface
    tmax : ndarray
        Maximum temperature
    tmin : ndarray
        Minimum temperature
    tmean : ndarray
        Mean temperature

    Returns:
    --------
    pet_values : ndarray
        Potential evapotranspiration values

    Notes:
    ------
    This is a simplified implementation based on the FAO Penman-Monteith equation
    as described in the ECA&D ATBD document.
    """
    # Constants
    gamma = 0.067  # psychrometric constant (kPa/°C)

    # Temperature-dependent parameters
    delta = 4098 * (0.6108 * np.exp((17.27 * tmean) / (tmean + 237.3))) / ((tmean + 237.3) ** 2)

    # Simplified calculation (would need more inputs for full equation)
    G = 0  # soil heat flux, assumed 0 for daily calculation
    u2 = 2  # wind speed at 2m height, assumed constant
    es = 0.6108 * np.exp((17.27 * tmean) / (tmean + 237.3))  # saturation vapor pressure
    ea = 0.6108 * np.exp((17.27 * tmin) / (tmin + 237.3))  # actual vapor pressure (simplified)

    pet_values = (0.408 * delta * (Ra - G) + gamma * (900 / (tmean + 273)) * u2 * (es - ea)) / (
                delta + gamma * (1 + 0.34 * u2))

    return pet_values


def load_data(site):
    """
    Load climate data for processing

    Parameters:
    -----------
    site : str
        Site identifier

    Returns:
    --------
    data : dict
        Dictionary containing the climate data
    """
    # This is a placeholder - in a real application, this would load actual data files
    # Instead, we'll create synthetic data for demonstration

    # Create years array
    years = np.arange(2000, 2010)
    num_years = len(years)

    # Create seasons
    seasons = ['winter', 'spring', 'summer', 'fall']
    num_seasons = len(seasons)

    # Days per season
    days_per_season = [90, 92, 92, 91]

    # Variables
    variables = [
        'TAIR',  # 1: Mean Air Temperature
        'DWPT',  # 2: Mean Dewpoint Temperature
        'TAIRx',  # 3: Maximum Air Temperature
        'DWPTx',  # 4: Maximum Dewpoint Temperature
        'TAIRn',  # 5: Minimum Air Temperature
        'DWPTn',  # 6: Minimum Dewpoint Temperature
        'PRCP',  # 7: Precipitation
        'RELH',  # 8: Relative Humidity
        'PRES',  # 9: Pressure (not used in original code)
        'SRAD',  # 10: Solar Radiation (not used in original code)
        'WDIR',  # 11: Wind Direction
        'WSPD',  # 12: Wind Speed
        'WSMX',  # 13: Maximum Wind Speed
        'Ra'  # 14: Net radiation
    ]
    num_vars = len(variables)

    # Create data structure
    data = {
        'year': years,
        'season': seasons,
        'var': variables,
        'days_per_season': days_per_season,
        'data': []
    }

    # Initialize data with synthetic values
    for s in range(num_seasons):
        season_data = []
        for y in range(num_years):
            n_days = days_per_season[s]

            # Create a dict to store variable data for this season/year
            var_data = {}

            # Fill with synthetic data
            var_data['TAIR'] = np.random.normal(15, 5, n_days)  # Mean temp around 15°C
            var_data['DWPT'] = var_data['TAIR'] - np.random.uniform(2, 5, n_days)  # Dewpoint lower than temp
            var_data['TAIRx'] = var_data['TAIR'] + np.random.uniform(3, 8, n_days)  # Daily max
            var_data['DWPTx'] = var_data['DWPT'] + np.random.uniform(1, 3, n_days)  # Daily max dewpoint
            var_data['TAIRn'] = var_data['TAIR'] - np.random.uniform(3, 8, n_days)  # Daily min
            var_data['DWPTn'] = var_data['DWPT'] - np.random.uniform(1, 3, n_days)  # Daily min dewpoint
            var_data['PRCP'] = np.random.exponential(5, n_days) * np.random.binomial(1, 0.3, n_days)  # Precipitation
            var_data['RELH'] = np.random.uniform(30, 90, n_days)  # Relative humidity
            var_data['WDIR'] = np.random.uniform(0, 360, n_days)  # Wind direction
            var_data['WSPD'] = np.random.weibull(2, n_days) * 3  # Wind speed
            var_data['WSMX'] = var_data['WSPD'] + np.random.weibull(2, n_days) * 2  # Max wind speed
            var_data['Ra'] = np.random.uniform(10, 25, n_days)  # Net radiation

            # Add to season data
            season_data.append(var_data)

        # Add to overall data
        data['data'].append(season_data)

    return data


def calculate_indices(data):
    """
    Calculate climate indices based on the input data.

    Parameters:
    -----------
    data : dict
        Dictionary containing climate data

    Returns:
    --------
    indices : dict
        Dictionary containing calculated climate indices
    """
    # Extract dimensions
    num_seasons = len(data['season'])
    num_years = len(data['year'])

    # Initialize dictionary to store indices
    indices = {}

    #########################
    # COLD INDICES
    #########################

    # GD4 - Growing Degree Days (sum of TG > 4°C)
    # As described in ECA&D ATBD section 5.3.2
    indices['GD4'] = np.zeros((num_seasons, num_years))
    for i in range(num_seasons):
        for j in range(num_years):
            tair = data['data'][i][j]['TAIR']
            indices['GD4'][i, j] = np.sum(np.maximum(tair - 4, 0))

    # GD10 - Growing Degree Days (sum of TG > 10°C)
    indices['GD10'] = np.zeros((num_seasons, num_years))
    for i in range(num_seasons):
        for j in range(num_years):
            tair = data['data'][i][j]['TAIR']
            indices['GD10'][i, j] = np.sum(np.maximum(tair - 10, 0))

    # FD - Frost Days (TN < 0°C)
    # As described in ECA&D ATBD section 5.3.2
    indices['FD'] = np.zeros((num_seasons, num_years))
    for i in range(num_seasons):
        for j in range(num_years):
            tn = data['data'][i][j]['TAIRn']
            indices['FD'][i, j] = np.sum(tn < 0)

    # CFD - Maximum number of consecutive frost days (TN < 0°C)
    # As described in ECA&D ATBD section 5.3.2
    indices['CFD'] = np.zeros((num_seasons, num_years))
    for i in range(num_seasons):
        for j in range(num_years):
            tn = data['data'][i][j]['TAIRn']
            B, N, _ = run_length(tn < 0)
            indices['CFD'][i, j] = np.max(N) if len(N) > 0 else 0

    # HDD - Heating Degree Days (sum of 18.3°C - TG)
    # Similar to HD17 in ECA&D ATBD section 5.3.2 but with 18.3°C baseline
    indices['HDD'] = np.zeros((num_seasons, num_years))
    for i in range(num_seasons):
        for j in range(num_years):
            tair = data['data'][i][j]['TAIR']
            indices['HDD'][i, j] = np.sum(18.3 - tair)

    # ID - Ice Days (TX < 0°C)
    # As described in ECA&D ATBD section 5.3.2
    indices['ID'] = np.zeros((num_seasons, num_years))
    for i in range(num_seasons):
        for j in range(num_years):
            tx = data['data'][i][j]['TAIRx']
            indices['ID'][i, j] = np.sum(tx < 0)

    # TXn - Minimum value of daily maximum temperature
    # As described in ECA&D ATBD section 5.3.2
    indices['TXn'] = np.zeros((num_seasons, num_years))
    for i in range(num_seasons):
        for j in range(num_years):
            tx = data['data'][i][j]['TAIRx']
            indices['TXn'][i, j] = np.nanmin(tx)

    # TNn - Minimum value of daily minimum temperature
    # As described in ECA&D ATBD section 5.3.2
    indices['TNn'] = np.zeros((num_seasons, num_years))
    for i in range(num_seasons):
        for j in range(num_years):
            tn = data['data'][i][j]['TAIRn']
            indices['TNn'][i, j] = np.nanmin(tn)

    #########################
    # DROUGHT INDICES
    #########################

    # PET - Potential Evapotranspiration
    # As described in ECA&D ATBD section 5.3.4
    indices['PET'] = np.zeros((num_seasons, num_years))
    for i in range(num_seasons):
        for j in range(num_years):
            ra = data['data'][i][j]['Ra']
            tmax = data['data'][i][j]['TAIRx']
            tmin = data['data'][i][j]['TAIRn']
            tmean = data['data'][i][j]['TAIR']
            indices['PET'][i, j] = np.nanmean(pet(ra, tmax, tmin, tmean))

    #########################
    # HEAT INDICES
    #########################

    # SU - Summer Days (TX > 25°C)
    # As described in ECA&D ATBD section 5.3.5
    indices['SU'] = np.zeros((num_seasons, num_years))
    for i in range(num_seasons):
        for j in range(num_years):
            tx = data['data'][i][j]['TAIRx']
            indices['SU'][i, j] = np.sum(tx > 25)

    # CSU - Maximum number of consecutive summer days (TX > 25°C)
    # As described in ECA&D ATBD section 5.3.5
    indices['CSU'] = np.zeros((num_seasons, num_years))
    for i in range(num_seasons):
        for j in range(num_years):
            tx = data['data'][i][j]['TAIRx']
            B, N, _ = run_length(tx > 25)
            indices['CSU'][i, j] = np.max(N) if len(N) > 0 else 0

    # TR - Tropical Nights (TN > 20°C)
    # As described in ECA&D ATBD section 5.3.5
    indices['TR'] = np.zeros((num_seasons, num_years))
    for i in range(num_seasons):
        for j in range(num_years):
            tn = data['data'][i][j]['TAIRn']
            indices['TR'][i, j] = np.sum(tn > 20)

    # TXx - Maximum value of daily maximum temperature
    # As described in ECA&D ATBD section 5.3.5
    indices['TXx'] = np.zeros((num_seasons, num_years))
    for i in range(num_seasons):
        for j in range(num_years):
            tx = data['data'][i][j]['TAIRx']
            indices['TXx'][i, j] = np.nanmax(tx)

    # TNx - Maximum value of daily minimum temperature
    # As described in ECA&D ATBD section 5.3.5
    indices['TNx'] = np.zeros((num_seasons, num_years))
    for i in range(num_seasons):
        for j in range(num_years):
            tn = data['data'][i][j]['TAIRn']
            indices['TNx'][i, j] = np.nanmax(tn)

    #########################
    # HUMIDITY INDICES
    #########################

    # RH - Mean of daily relative humidity
    # As described in ECA&D ATBD section 5.3.6
    indices['RH'] = np.zeros((num_seasons, num_years))
    for i in range(num_seasons):
        for j in range(num_years):
            relh = data['data'][i][j]['RELH']
            indices['RH'][i, j] = np.nanmean(relh)

    #########################
    # RAIN INDICES
    #########################

    # RR - Precipitation sum
    # As described in ECA&D ATBD section 5.3.8
    indices['RR'] = np.zeros((num_seasons, num_years))
    for i in range(num_seasons):
        for j in range(num_years):
            prcp = data['data'][i][j]['PRCP']
            indices['RR'][i, j] = np.nansum(prcp)

    # RR1 - Wet days (RR ≥ 1 mm)
    # As described in ECA&D ATBD section 5.3.8
    indices['RR1'] = np.zeros((num_seasons, num_years))
    for i in range(num_seasons):
        for j in range(num_years):
            prcp = data['data'][i][j]['PRCP']
            indices['RR1'][i, j] = np.sum(prcp >= 1)

    # SDII - Simple daily intensity index (mm/wet day)
    # As described in ECA&D ATBD section 5.3.8
    indices['SDII'] = np.zeros((num_seasons, num_years))
    for i in range(num_seasons):
        for j in range(num_years):
            prcp = data['data'][i][j]['PRCP']
            wet_days = prcp >= 1
            if np.sum(wet_days) > 0:
                indices['SDII'][i, j] = np.nansum(prcp[wet_days]) / np.sum(wet_days)
            else:
                indices['SDII'][i, j] = np.nan

    # CWD - Maximum number of consecutive wet days (RR ≥ 1 mm)
    # As described in ECA&D ATBD section 5.3.8
    indices['CWD'] = np.zeros((num_seasons, num_years))
    for i in range(num_seasons):
        for j in range(num_years):
            prcp = data['data'][i][j]['PRCP']
            B, N, _ = run_length(prcp >= 1)
            indices['CWD'][i, j] = np.max(N) if len(N) > 0 else 0

    # RR10 - Heavy precipitation days (precipitation ≥ 10 mm)
    # As described in ECA&D ATBD section 5.3.8
    indices['RR10'] = np.zeros((num_seasons, num_years))
    for i in range(num_seasons):
        for j in range(num_years):
            prcp = data['data'][i][j]['PRCP']
            indices['RR10'][i, j] = np.sum(prcp >= 10)

    # RR20 - Very heavy precipitation days (precipitation ≥ 20 mm)
    # As described in ECA&D ATBD section 5.3.8
    indices['RR20'] = np.zeros((num_seasons, num_years))
    for i in range(num_seasons):
        for j in range(num_years):
            prcp = data['data'][i][j]['PRCP']
            indices['RR20'][i, j] = np.sum(prcp >= 20)

    # RX1day - Highest 1-day precipitation amount
    # As described in ECA&D ATBD section 5.3.8
    indices['RX1day'] = np.zeros((num_seasons, num_years))
    for i in range(num_seasons):
        for j in range(num_years):
            prcp = data['data'][i][j]['PRCP']
            indices['RX1day'][i, j] = np.nanmax(prcp)

    #########################
    # TEMPERATURE INDICES
    #########################

    # TG - Mean of daily mean air temperature
    # As described in ECA&D ATBD section 5.3.11
    indices['TG'] = np.zeros((num_seasons, num_years))
    for i in range(num_seasons):
        for j in range(num_years):
            tair = data['data'][i][j]['TAIR']
            indices['TG'][i, j] = np.nanmean(tair)

    # TdG - Mean of daily mean dewpoint temperature
    indices['TdG'] = np.zeros((num_seasons, num_years))
    for i in range(num_seasons):
        for j in range(num_years):
            dwpt = data['data'][i][j]['DWPT']
            indices['TdG'][i, j] = np.nanmean(dwpt)

    # TN - Mean of daily minimum air temperature
    # As described in ECA&D ATBD section 5.3.11
    indices['TN'] = np.zeros((num_seasons, num_years))
    for i in range(num_seasons):
        for j in range(num_years):
            tn = data['data'][i][j]['TAIRn']
            indices['TN'][i, j] = np.nanmean(tn)

    # TdN - Mean of daily minimum dewpoint temperature
    indices['TdN'] = np.zeros((num_seasons, num_years))
    for i in range(num_seasons):
        for j in range(num_years):
            dwptn = data['data'][i][j]['DWPTn']
            indices['TdN'][i, j] = np.nanmean(dwptn)

    # TX - Mean of daily maximum air temperature
    # As described in ECA&D ATBD section 5.3.11
    indices['TX'] = np.zeros((num_seasons, num_years))
    for i in range(num_seasons):
        for j in range(num_years):
            tx = data['data'][i][j]['TAIRx']
            indices['TX'][i, j] = np.nanmean(tx)

    # TdX - Mean of daily maximum dewpoint temperature
    indices['TdX'] = np.zeros((num_seasons, num_years))
    for i in range(num_seasons):
        for j in range(num_years):
            dwptx = data['data'][i][j]['DWPTx']
            indices['TdX'][i, j] = np.nanmean(dwptx)

    # DTR - Mean of diurnal air temperature range
    # As described in ECA&D ATBD section 5.3.11
    indices['DTR'] = np.zeros((num_seasons, num_years))
    for i in range(num_seasons):
        for j in range(num_years):
            tn = data['data'][i][j]['TAIRn']
            tx = data['data'][i][j]['TAIRx']
            dtr = tx - tn
            indices['DTR'][i, j] = np.nanmean(dtr)

    # DTdR - Mean of diurnal dewpoint temperature range
    indices['DTdR'] = np.zeros((num_seasons, num_years))
    for i in range(num_seasons):
        for j in range(num_years):
            tn = data['data'][i][j]['DWPTn']
            tx = data['data'][i][j]['DWPTx']
            dtr = tx - tn
            indices['DTdR'][i, j] = np.nanmean(dtr)

    # ETR - Intra-period extreme temperature range
    # As described in ECA&D ATBD section 5.3.11
    indices['ETR'] = np.zeros((num_seasons, num_years))
    for i in range(num_seasons):
        for j in range(num_years):
            tn = data['data'][i][j]['TAIRn']
            tx = data['data'][i][j]['TAIRx']
            indices['ETR'][i, j] = np.nanmax(tx) - np.nanmin(tn)

    #########################
    # WIND INDICES
    #########################

    # FXx - Maximum daily wind speed gust
    # Similar to the description in ECA&D ATBD section 5.3.12
    indices['FXx'] = np.zeros((num_seasons, num_years))
    for i in range(num_seasons):
        for j in range(num_years):
            wsmx = data['data'][i][j]['WSMX']
            # Convert m/s to mph (simplified for demonstration)
            indices['FXx'][i, j] = np.nanmax(wsmx) * 2.237

    # FG - Mean of daily mean wind strength
    # As described in ECA&D ATBD section 5.3.12
    indices['FG'] = np.zeros((num_seasons, num_years))
    for i in range(num_seasons):
        for j in range(num_years):
            wspd = data['data'][i][j]['WSPD']
            indices['FG'][i, j] = np.nanmean(wspd)

    # Wind direction indices
    # As described in ECA&D ATBD section 5.3.12
    wind_indices = ['DDsouth', 'DDeast', 'DDwest', 'DDnorth']
    for index_name in wind_indices:
        indices[index_name] = np.zeros((num_seasons, num_years))

    for i in range(num_seasons):
        for j in range(num_years):
            wdir = data['data'][i][j]['WDIR']

            # Days with southerly winds (135° < DD ≤ 225°)
            south_mask = (wdir <= 225) & (wdir > 135)
            indices['DDsouth'][i, j] = np.sum(south_mask)

            # Days with easterly winds (45° < DD ≤ 135°)
            east_mask = (wdir <= 135) & (wdir > 45)
            indices['DDeast'][i, j] = np.sum(east_mask)

            # Days with westerly winds (225° < DD ≤ 315°)
            west_mask = (wdir <= 315) & (wdir > 225)
            indices['DDwest'][i, j] = np.sum(west_mask)

            # Days with northerly winds (DD ≤ 45° or DD > 315°)
            north_mask = (wdir <= 45) | (wdir > 315)
            indices['DDnorth'][i, j] = np.sum(north_mask)

    return indices


def main():
    """
    Main function to run the climate indices calculation.
    """
    # Set site name
    site = 'FARM'

    # Load data
    print(f"Loading data for site: {site}")
    data = load_data(site)

    # Calculate indices
    print("Calculating climate indices...")
    indices = calculate_indices(data)

    # Display results
    print("\nClimate Indices Results:")
    print("========================")
    for index_name, index_values in indices.items():
        season_means = np.nanmean(index_values, axis=1)
        print(f"{index_name}:")
        for s, season in enumerate(data['season']):
            print(f"  {season.capitalize()}: {season_means[s]:.2f}")

    print("\nDone!")


if __name__ == "__main__":
    main()