#!/usr/bin/env python3
"""
extract_climate_indices.py

Comprehensive script for calculating climate indices based on meteorological data.
This script implements the indices described in the European Climate Assessment & Dataset
(ECA&D) project's Algorithm Theoretical Basis Document (ATBD).

This script can be used on daily, monthly, or annual data to calculate various climate indices
including:
- Temperature indices (frost days, heat days, etc.)
- Precipitation indices (wet days, intensity, etc.)
- Humidity indices
- Wind indices
- Drought indices

Usage:
    python extract_climate_indices.py --site BMTN --start 2015-01-01 --end 2022-12-31 --period annual

Author: Climate Data Team
Date: 2025-03-24
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
from datetime import datetime
import h5py
import json

# Import custom modules
try:
    from run_length import find_runs, max_run_length
except ImportError:
    # Define fallback functions if module not available
    def find_runs(arr):
        """Find runs of consecutive True values"""
        # Convert to numpy array
        arr = np.asarray(arr, dtype=bool)
        
        # Handle empty array
        if arr.size == 0:
            return np.array([], dtype=int)
        
        # Find transitions
        padded = np.concatenate(([False], arr, [False]))
        diff = np.diff(padded.astype(int))
        
        # Start and end positions
        run_starts = np.where(diff > 0)[0]
        run_ends = np.where(diff < 0)[0]
        
        # Calculate run lengths
        run_lengths = run_ends - run_starts
        
        return run_lengths
    
    def max_run_length(arr):
        """Calculate maximum run length of True values"""
        run_lengths = find_runs(arr)
        return np.max(run_lengths) if run_lengths.size > 0 else 0

try:
    from pet import pet
except ImportError:
    # Define fallback pet function if module not available
    def pet(Ra, tmax, tmin, tmean):
        """
        Hargreaves-Samani Potential Evapotranspiration (PET) formula
        
        Parameters:
        -----------
        Ra : float or array-like
            Extraterrestrial radiation (MJ/m²/day)
        tmax : float or array-like
            Maximum temperature (°C)
        tmin : float or array-like
            Minimum temperature (°C)
        tmean : float or array-like
            Mean temperature (°C)
        
        Returns:
        --------
        PET : float or array-like
            Potential evapotranspiration (mm/day)
        """
        # Hargreaves-Samani coefficient (typically 0.0023)
        k = 0.0023
        
        # Calculate PET
        PET = k * Ra * np.sqrt(tmax - tmin) * (tmean + 17.8)
        
        return PET

try:
    from extraterrestrial_radiation import extraterrestrial_radiation
except ImportError:
    # Define fallback extraterrestrial_radiation function
    def extraterrestrial_radiation(doy, lat):
        """
        Calculate extraterrestrial radiation (Ra) using the FAO formula
        
        Parameters:
        -----------
        doy : int or array-like
            Day of the year (1-365)
        lat : float or array-like
            Latitude in degrees
        
        Returns:
        --------
        Ra : float or array-like
            Extraterrestrial radiation (MJ/m²/day)
        """
        # Convert latitude to radians
        phi = np.deg2rad(lat)
        
        # Inverse relative Earth-Sun distance
        dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
        
        # Solar declination
        delta = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
        
        # Sunset hour angle
        ws = np.arccos(-np.tan(phi) * np.tan(delta))
        
        # Extraterrestrial radiation
        Ra = 118.1 / np.pi * dr * (ws * np.sin(phi) * np.sin(delta) + 
                                  np.cos(phi) * np.cos(delta) * np.sin(ws))
        
        return Ra


class ClimateIndices:
    """
    Class for calculating climate indices from meteorological data
    
    Attributes:
    -----------
    data : pd.DataFrame
        Input meteorological data
    site : str
        Site identifier
    time_col : str
        Name of the timestamp column
    config : dict
        Configuration settings
    variable_map : dict
        Mapping of standard variable names to input data columns
    indices : dict
        Calculated climate indices
    """
    
    def __init__(self, site='BMTN', time_col='TimestampCollected', lat=None, lon=None):
        """
        Initialize the climate indices calculator
        
        Parameters:
        -----------
        site : str
            Site identifier
        time_col : str
            Name of the timestamp column
        lat : float, optional
            Latitude of the site (if known)
        lon : float, optional
            Longitude of the site (if known)
        """
        self.site = site
        self.time_col = time_col
        self.lat = lat
        self.lon = lon
        self.data = None
        self.indices = {}
        
        # Default variable mapping
        self.variable_map = {
            'temperature': 'TAIR',       # Mean temperature
            'temperature_max': 'TAIRx',  # Maximum temperature
            'temperature_min': 'TAIRn',  # Minimum temperature
            'precipitation': 'PRCP',     # Precipitation
            'humidity': 'RELH',          # Relative humidity
            'wind_speed': 'WSPD',        # Wind speed
            'wind_gust': 'WSMX',         # Maximum wind speed
            'wind_direction': 'WDIR',    # Wind direction
            'dewpoint': 'DWPT',          # Dewpoint temperature
            'dewpoint_max': 'DWPTx',     # Maximum dewpoint
            'dewpoint_min': 'DWPTn',     # Minimum dewpoint
            'pressure': 'PRES',          # Atmospheric pressure
            'solar_radiation': 'SRAD'    # Solar radiation
        }
        
        # Standard indices to calculate
        self.standard_indices = {
            # Temperature indices
            'FD': 'Frost days (days with minimum temperature < 0°C)',
            'SU': 'Summer days (days with maximum temperature > 25°C)',
            'ID': 'Ice days (days with maximum temperature < 0°C)',
            'TR': 'Tropical nights (days with minimum temperature > 20°C)',
            'GSL': 'Growing season length (days)',
            'TXx': 'Maximum value of daily maximum temperature',
            'TNn': 'Minimum value of daily minimum temperature',
            'TN10p': 'Percentage of days with minimum temperature < 10th percentile',
            'TX10p': 'Percentage of days with maximum temperature < 10th percentile',
            'TN90p': 'Percentage of days with minimum temperature > 90th percentile',
            'TX90p': 'Percentage of days with maximum temperature > 90th percentile',
            'DTR': 'Daily temperature range (mean difference between max and min temperature)',
            'ETR': 'Extreme temperature range (difference between max of max and min of min temperature)',
            'GD4': 'Growing degree days (base 4°C)',
            'GD10': 'Growing degree days (base 10°C)',
            'CFD': 'Consecutive frost days (max number of consecutive days with Tmin < 0°C)',
            'HDD': 'Heating degree days (sum of 18.3°C - Tmean, when Tmean < 18.3°C)',
            
            # Precipitation indices
            'RR': 'Total precipitation',
            'RR1': 'Number of wet days (precipitation >= 1mm)',
            'SDII': 'Simple daily intensity index (mean precipitation on wet days)',
            'R10': 'Number of days with precipitation >= 10mm',
            'R20': 'Number of days with precipitation >= 20mm',
            'CWD': 'Maximum number of consecutive wet days',
            'CDD': 'Maximum number of consecutive dry days',
            'R95p': 'Contribution from very wet days (precipitation > 95th percentile)',
            'PRCPTOT': 'Annual total precipitation',
            'RX1day': 'Maximum 1-day precipitation amount',
            'RX5day': 'Maximum 5-day precipitation amount',
            
            # Humidity indices
            'RH': 'Mean relative humidity',
            
            # Wind indices
            'FG': 'Mean wind speed',
            'FXx': 'Maximum wind gust',
            'DDsouth': 'Days with southerly winds (135° < DD ≤ 225°)',
            'DDeast': 'Days with easterly winds (45° < DD ≤ 135°)',
            'DDwest': 'Days with westerly winds (225° < DD ≤ 315°)',
            'DDnorth': 'Days with northerly winds (DD ≤ 45° or DD > 315°)',
            
            # Drought indices
            'PET': 'Potential evapotranspiration'
        }
    
    def load_data(self, data_file, start_date=None, end_date=None):
        """
        Load climate data from file
        
        Parameters:
        -----------
        data_file : str or Path
            Path to data file (CSV, MAT, or HDF5)
        start_date : str or datetime, optional
            Start date for filtering (format: YYYY-MM-DD)
        end_date : str or datetime, optional
            End date for filtering (format: YYYY-MM-DD)
            
        Returns:
        --------
        success : bool
            True if data loaded successfully
        """
        data_file = Path(data_file)
        
        try:
            # Load based on file extension
            if data_file.suffix.lower() == '.csv':
                self.data = pd.read_csv(data_file, parse_dates=[self.time_col])
            elif data_file.suffix.lower() == '.mat':
                from scipy.io import loadmat
                mat_data = loadmat(data_file)
                
                # Try to find the main data array
                main_data = None
                for key in mat_data.keys():
                    if key.startswith('__'):  # Skip MATLAB metadata
                        continue
                    
                    # Look for structured arrays with time data
                    if isinstance(mat_data[key], np.ndarray) and hasattr(mat_data[key], 'dtype'):
                        if hasattr(mat_data[key].dtype, 'names') and mat_data[key].dtype.names is not None:
                            if self.time_col in mat_data[key].dtype.names:
                                main_data = mat_data[key]
                                break
                
                if main_data is None:
                    raise ValueError(f"Could not find data array with '{self.time_col}' in MAT file")
                
                # Convert to DataFrame
                self.data = pd.DataFrame()
                for name in main_data.dtype.names:
                    values = main_data[name]
                    if values.ndim > 1:
                        values = values.flatten()
                    self.data[name] = values
                
                # Convert time column to datetime
                if self.time_col in self.data.columns:
                    self.data[self.time_col] = pd.to_datetime(self.data[self.time_col])
            
            elif data_file.suffix.lower() in ['.h5', '.hdf5']:
                with h5py.File(data_file, 'r') as f:
                    # Check if file has a simple structure
                    if all(isinstance(f[key], h5py.Dataset) for key in f.keys()):
                        data_dict = {key: f[key][()] for key in f.keys()}
                        self.data = pd.DataFrame(data_dict)
                    else:
                        # Try to find data in nested structure
                        for group_name, group in f.items():
                            if isinstance(group, h5py.Group) and 'data' in group_name.lower():
                                data_dict = {}
                                for key in group.keys():
                                    data_dict[key] = group[key][()]
                                self.data = pd.DataFrame(data_dict)
                                break
                        
                        if self.data is None:
                            raise ValueError("Could not find data in HDF5 file structure")
                
                # Convert time column to datetime if present
                if self.time_col in self.data.columns:
                    # If stored as byte strings, decode first
                    if isinstance(self.data[self.time_col].iloc[0], bytes):
                        self.data[self.time_col] = self.data[self.time_col].apply(
                            lambda x: x.decode('utf-8') if isinstance(x, bytes) else x
                        )
                    self.data[self.time_col] = pd.to_datetime(self.data[self.time_col])
            
            else:
                raise ValueError(f"Unsupported file format: {data_file.suffix}")
            
            # Filter by date range if specified
            if start_date is not None:
                if isinstance(start_date, str):
                    start_date = pd.to_datetime(start_date)
                self.data = self.data[self.data[self.time_col] >= start_date]
                
            if end_date is not None:
                if isinstance(end_date, str):
                    end_date = pd.to_datetime(end_date)
                self.data = self.data[self.data[self.time_col] <= end_date]
            
            # Set index to time column
            self.data = self.data.set_index(self.time_col)
            
            return True
        
        except Exception as e:
            warnings.warn(f"Error loading data: {e}")
            return False
    
    def set_variable_mapping(self, mapping=None):
        """
        Set mapping between standard variable names and input data columns
        
        Parameters:
        -----------
        mapping : dict, optional
            Dictionary mapping standard names to column names
            If None, try to auto-detect mapping
        """
        if mapping is not None:
            # Update provided mappings
            self.variable_map.update(mapping)
        else:
            # Try to auto-detect columns
            for standard_name, default_col in self.variable_map.items():
                # Check different variants of column names
                possible_cols = [
                    default_col,                                # Standard name
                    default_col.lower(),                        # Lowercase
                    default_col.upper(),                        # Uppercase
                    f"{default_col}_annual",                    # Annual suffix
                    f"{default_col}_month",                     # Monthly suffix
                    f"{default_col}_day",                       # Daily suffix
                    f"{default_col}_season",                    # Seasonal suffix
                    standard_name,                              # Use standard name directly
                    standard_name.replace('_', '')              # No underscores
                ]
                
                # Find first matching column
                for col in possible_cols:
                    if col in self.data.columns:
                        self.variable_map[standard_name] = col
                        break
    
    def _get_variable(self, name):
        """Get variable from data using mapping"""
        if name not in self.variable_map:
            return None
            
        col = self.variable_map[name]
        if col in self.data.columns:
            return self.data[col]
        else:
            return None
    
    def calculate_indices(self, period='annual', custom_indices=None):
        """
        Calculate climate indices
        
        Parameters:
        -----------
        period : str
            Time period for aggregation: 'daily', 'monthly', 'seasonal', or 'annual'
        custom_indices : list, optional
            List of specific indices to calculate (if None, calculate all applicable)
            
        Returns:
        --------
        indices : dict
            Dictionary containing calculated indices
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Set indices to calculate
        if custom_indices is not None:
            indices_to_calculate = custom_indices
        else:
            indices_to_calculate = list(self.standard_indices.keys())
        
        # Initialize results dictionary
        self.indices = {
            'site': self.site,
            'period': period,
            'metadata': {
                'start_date': self.data.index.min(),
                'end_date': self.data.index.max(),
                'variables': list(self.variable_map.values())
            }
        }
        
        # Calculate indices based on period
        if period == 'daily':
            self._calculate_daily_indices(indices_to_calculate)
        elif period == 'monthly':
            self._calculate_monthly_indices(indices_to_calculate)
        elif period == 'seasonal':
            self._calculate_seasonal_indices(indices_to_calculate)
        elif period == 'annual':
            self._calculate_annual_indices(indices_to_calculate)
        else:
            raise ValueError(f"Invalid period: {period}. Use 'daily', 'monthly', 'seasonal', or 'annual'.")
        
        return self.indices
    
    def _calculate_daily_indices(self, indices_list):
        """Calculate indices at daily time scale"""
        # Get relevant columns
        temp = self._get_variable('temperature')
        temp_max = self._get_variable('temperature_max')
        temp_min = self._get_variable('temperature_min')
        precip = self._get_variable('precipitation')
        humidity = self._get_variable('humidity')
        wind_speed = self._get_variable('wind_speed')
        wind_gust = self._get_variable('wind_gust')
        wind_dir = self._get_variable('wind_direction')
        
        # Create dataframe for daily indices
        daily_indices = pd.DataFrame(index=self.data.index)
        
        # Calculate daily indices
        if 'FD' in indices_list and temp_min is not None:
            daily_indices['FD'] = (temp_min < 0).astype(int)
        
        if 'SU' in indices_list and temp_max is not None:
            daily_indices['SU'] = (temp_max > 25).astype(int)
        
        if 'ID' in indices_list and temp_max is not None:
            daily_indices['ID'] = (temp_max < 0).astype(int)
        
        if 'TR' in indices_list and temp_min is not None:
            daily_indices['TR'] = (temp_min > 20).astype(int)
        
        if 'DTR' in indices_list and temp_max is not None and temp_min is not None:
            daily_indices['DTR'] = temp_max - temp_min
        
        if 'GD4' in indices_list and temp is not None:
            daily_indices['GD4'] = np.maximum(temp - 4, 0)
        
        if 'GD10' in indices_list and temp is not None:
            daily_indices['GD10'] = np.maximum(temp - 10, 0)
        
        if 'HDD' in indices_list and temp is not None:
            daily_indices['HDD'] = np.maximum(18.3 - temp, 0)
        
        if 'RR1' in indices_list and precip is not None:
            daily_indices['RR1'] = (precip >= 1).astype(int)
        
        if 'R10' in indices_list and precip is not None:
            daily_indices['R10'] = (precip >= 10).astype(int)
        
        if 'R20' in indices_list and precip is not None:
            daily_indices['R20'] = (precip >= 20).astype(int)
        
        if 'RH' in indices_list and humidity is not None:
            daily_indices['RH'] = humidity
        
        if 'FG' in indices_list and wind_speed is not None:
            daily_indices['FG'] = wind_speed
        
        if 'FXx' in indices_list and wind_gust is not None:
            daily_indices['FXx'] = wind_gust
        
        # Wind direction indices
        if wind_dir is not None:
            if 'DDsouth' in indices_list:
                daily_indices['DDsouth'] = ((wind_dir > 135) & (wind_dir <= 225)).astype(int)
            
            if 'DDeast' in indices_list:
                daily_indices['DDeast'] = ((wind_dir > 45) & (wind_dir <= 135)).astype(int)
            
            if 'DDwest' in indices_list:
                daily_indices['DDwest'] = ((wind_dir > 225) & (wind_dir <= 315)).astype(int)
            
            if 'DDnorth' in indices_list:
                daily_indices['DDnorth'] = ((wind_dir <= 45) | (wind_dir > 315)).astype(int)
        
        # Store results
        self.indices['daily'] = daily_indices
    
    def _calculate_monthly_indices(self, indices_list):
        """Calculate indices at monthly time scale"""
        # First calculate daily indices if needed
        try:
            daily_indices = self.indices['daily']
        except KeyError:
            self._calculate_daily_indices(indices_list)
            daily_indices = self.indices['daily']
        
        # Add month and year columns for grouping
        daily_indices['year'] = daily_indices.index.year
        daily_indices['month'] = daily_indices.index.month
        
        # Group by year and month
        monthly_groups = daily_indices.groupby(['year', 'month'])
        
        # Calculate monthly aggregations
        monthly_indices = {}
        
        # Temperature indices
        if 'FD' in indices_list:
            monthly_indices['FD'] = monthly_groups['FD'].sum()
        
        if 'SU' in indices_list:
            monthly_indices['SU'] = monthly_groups['SU'].sum()
        
        if 'ID' in indices_list:
            monthly_indices['ID'] = monthly_groups['ID'].sum()
        
        if 'TR' in indices_list:
            monthly_indices['TR'] = monthly_groups['TR'].sum()
        
        if 'DTR' in indices_list:
            monthly_indices['DTR'] = monthly_groups['DTR'].mean()
        
        if 'GD4' in indices_list:
            monthly_indices['GD4'] = monthly_groups['GD4'].sum()
        
        if 'GD10' in indices_list:
            monthly_indices['GD10'] = monthly_groups['GD10'].sum()
        
        if 'HDD' in indices_list:
            monthly_indices['HDD'] = monthly_groups['HDD'].sum()
        
        # Precipitation indices
        if 'RR1' in indices_list:
            monthly_indices['RR1'] = monthly_groups['RR1'].sum()
        
        if 'R10' in indices_list:
            monthly_indices['R10'] = monthly_groups['R10'].sum()
        
        if 'R20' in indices_list:
            monthly_indices['R20'] = monthly_groups['R20'].sum()
        
        # Wind and Humidity
        if 'RH' in indices_list and 'RH' in daily_indices.columns:
            monthly_indices['RH'] = monthly_groups['RH'].mean()
        
        if 'FG' in indices_list and 'FG' in daily_indices.columns:
            monthly_indices['FG'] = monthly_groups['FG'].mean()
        
        if 'FXx' in indices_list and 'FXx' in daily_indices.columns:
            monthly_indices['FXx'] = monthly_groups['FXx'].max()
        
        # Wind direction indices
        for wind_dir in ['DDsouth', 'DDeast', 'DDwest', 'DDnorth']:
            if wind_dir in indices_list and wind_dir in daily_indices.columns:
                monthly_indices[wind_dir] = monthly_groups[wind_dir].sum()
        
        # Get max and min temperatures
        temp_max = self._get_variable('temperature_max')
        temp_min = self._get_variable('temperature_min')
        
        if temp_max is not None and temp_min is not None:
            # Add to data with month and year
            self.data['year'] = self.data.index.year
            self.data['month'] = self.data.index.month
            
            # Group
            temp_groups = self.data.groupby(['year', 'month'])
            
            # Calculate
            if 'TXx' in indices_list:
                monthly_indices['TXx'] = temp_groups[self.variable_map['temperature_max']].max()
            
            if 'TNn' in indices_list:
                monthly_indices['TNn'] = temp_groups[self.variable_map['temperature_min']].min()
            
            if 'ETR' in indices_list:
                temp_range = pd.Series(index=temp_groups.groups.keys())
                for (year, month), group in temp_groups:
                    temp_range[(year, month)] = (
                        group[self.variable_map['temperature_max']].max() - 
                        group[self.variable_map['temperature_min']].min()
                    )
                monthly_indices['ETR'] = temp_range
        
        # Precipitation total and max
        precip = self._get_variable('precipitation')
        if precip is not None:
            self.data['year'] = self.data.index.year
            self.data['month'] = self.data.index.month
            
            # Group
            precip_groups = self.data.groupby(['year', 'month'])
            
            if 'RR' in indices_list:
                monthly_indices['RR'] = precip_groups[self.variable_map['precipitation']].sum()
            
            if 'RX1day' in indices_list:
                monthly_indices['RX1day'] = precip_groups[self.variable_map['precipitation']].max()
            
            if 'SDII' in indices_list:
                sdii = pd.Series(index=precip_groups.groups.keys())
                for (year, month), group in precip_groups:
                    wet_days = group[group[self.variable_map['precipitation']] >= 1][self.variable_map['precipitation']]
                    sdii[(year, month)] = wet_days.mean() if len(wet_days) > 0 else 0
                monthly_indices['SDII'] = sdii
        
        # Combine all indices into a DataFrame
        monthly_df = pd.DataFrame(monthly_indices)
        
        # Store results
        self.indices['monthly'] = monthly_df
    
    def _calculate_seasonal_indices(self, indices_list):
        """Calculate indices at seasonal time scale"""
        # Define seasons
        seasons = {
            'DJF': [12, 1, 2],    # Winter
            'MAM': [3, 4, 5],     # Spring
            'JJA': [6, 7, 8],     # Summer
            'SON': [9, 10, 11]    # Fall
        }
        
        # First calculate monthly indices if needed
        try:
            monthly_indices = self.indices['monthly']
        except KeyError:
            self._calculate_monthly_indices(indices_list)
            monthly_indices = self.indices['monthly']
        
        # Reset index to get year and month as columns
        monthly_data = monthly_indices.reset_index()
        
        # Add season column
        monthly_data['season'] = monthly_data['month'].apply(
            lambda m: next(season for season, months in seasons.items() if m in months)
        )
        
        # For winter (DJF), adjust year for December
        winter_dec = (monthly_data['season'] == 'DJF') & (monthly_data['month'] == 12)
        monthly_data.loc[winter_dec, 'year'] = monthly_data.loc[winter_dec, 'year'] + 1
        
        # Group by year and season
        seasonal_groups = monthly_data.groupby(['year', 'season'])
        
        # Calculate seasonal aggregations
        seasonal_indices = {}
        
        # Sum indices
        sum_indices = ['FD', 'SU', 'ID', 'TR', 'GD4', 'GD10', 'HDD', 
                      'RR1', 'R10', 'R20', 'DDsouth', 'DDeast', 'DDwest', 'DDnorth']
        
        for idx in sum_indices:
            if idx in indices_list and idx in monthly_indices.columns:
                seasonal_indices[idx] = seasonal_groups[idx].sum()
        
        # Mean indices
        mean_indices = ['DTR', 'RH', 'FG']
        
        for idx in mean_indices:
            if idx in indices_list and idx in monthly_indices.columns:
                seasonal_indices[idx] = seasonal_groups[idx].mean()
        
        # Max indices
        max_indices = ['TXx', 'FXx', 'RX1day']
        
        for idx in max_indices:
            if idx in indices_list and idx in monthly_indices.columns:
                seasonal_indices[idx] = seasonal_groups[idx].max()
        
        # Min indices
        min_indices = ['TNn']
        
        for idx in min_indices:
            if idx in indices_list and idx in monthly_indices.columns:
                seasonal_indices[idx] = seasonal_groups[idx].min()
        
        # Special handling for precipitation total
        if 'RR' in indices_list and 'RR' in monthly_indices.columns:
            seasonal_indices['RR'] = seasonal_groups['RR'].sum()
        
        # Combine all indices into a DataFrame
        seasonal_df = pd.DataFrame(seasonal_indices)
        
        # Store results
        self.indices['seasonal'] = seasonal_df
    
    def _calculate_annual_indices(self, indices_list):
        """Calculate indices at annual time scale"""
        # First calculate monthly indices if needed
        try:
            monthly_indices = self.indices['monthly']
        except KeyError:
            self._calculate_monthly_indices(indices_list)
            monthly_indices = self.indices['monthly']
        
        # Reset index to get year as column
        monthly_data = monthly_indices.reset_index()
        
        # Group by year
        annual_groups = monthly_data.groupby('year')
        
        # Calculate annual aggregations
        annual_indices = {}
        
        # Sum indices
        sum_indices = ['FD', 'SU', 'ID', 'TR', 'GD4', 'GD10', 'HDD', 
                       'RR1', 'R10', 'R20', 'DDsouth', 'DDeast', 'DDwest', 'DDnorth']
        
        for idx in sum_indices:
            if idx in indices_list and idx in monthly_indices.columns:
                annual_indices[idx] = annual_groups[idx].sum()

                # Mean indices
                mean_indices = ['DTR', 'RH', 'FG']

                for idx in mean_indices:
                    if idx in indices_list and idx in monthly_indices.columns:
                        annual_indices[idx] = annual_groups[idx].mean()

                # Max indices
                max_indices = ['TXx', 'ETR', 'FXx', 'RX1day']

                for idx in max_indices:
                    if idx in indices_list and idx in monthly_indices.columns:
                        annual_indices[idx] = annual_groups[idx].max()

                # Min indices
                min_indices = ['TNn']

                for idx in min_indices:
                    if idx in indices_list and idx in monthly_indices.columns:
                        annual_indices[idx] = annual_groups[idx].min()

                # Special handling for precipitation total
                if 'RR' in indices_list and 'RR' in monthly_indices.columns:
                    annual_indices['RR'] = annual_groups['RR'].sum()

                # Calculate GSL - Growing Season Length
                if 'GSL' in indices_list:
                    # This requires daily data to calculate properly
                    self.data['year'] = self.data.index.year

                    # Get temperature data
                    temp = self._get_variable('temperature')

                    if temp is not None:
                        gsl = pd.Series(index=np.unique(self.data.index.year))

                        for year, group in self.data.groupby('year'):
                            if len(group) < 365:  # Skip incomplete years
                                continue

                            # Sort by day of year
                            group = group.sort_index()

                            # Find growing season start/end
                            growing_start = None
                            growing_end = None

                            # Start: First 6-day period with daily mean temperature > 5°C
                            for i in range(len(group) - 5):
                                if all(group[self.variable_map['temperature']].iloc[i:i + 6].values > 5):
                                    growing_start = group.index[i]
                                    break

                            # End: First frost after July 1
                            if growing_start is not None:
                                july_first = pd.Timestamp(year=year, month=7, day=1)
                                after_july = group.index >= july_first

                                if temp_min is not None:
                                    for date, value in group.loc[
                                        after_july, self.variable_map['temperature_min']].items():
                                        if value < 0:
                                            growing_end = date
                                            break
                                else:
                                    # If no min temperature, assume first day below 5°C after July 1
                                    for date, value in group.loc[after_july, self.variable_map['temperature']].items():
                                        if value < 5:
                                            growing_end = date
                                            break

                            # Calculate GSL
                            if growing_start is not None and growing_end is not None:
                                gsl[year] = (growing_end - growing_start).days
                            else:
                                gsl[year] = np.nan

                        annual_indices['GSL'] = gsl

                # Combine all indices into a DataFrame
                annual_df = pd.DataFrame(annual_indices)

                # Store results
                self.indices['annual'] = annual_df

            def save_indices(self, output_file=None, formats=None):
                """
                Save calculated indices to file(s)

                Parameters:
                -----------
                output_file : str or Path, optional
                    Base path for output file (without extension)
                    If None, uses '{site}_{period}_indices'
                formats : list, optional
                    List of output formats: 'csv', 'json', 'h5'
                    If None, defaults to ['csv']

                Returns:
                --------
                output_files : list
                    List of saved output file paths
                """
                if not self.indices:
                    raise ValueError("No indices calculated. Call calculate_indices() first.")

                # Set default output file
                if output_file is None:
                    output_file = f"{self.site}_{self.indices['period']}_indices"

                # Set default formats
                if formats is None:
                    formats = ['csv']

                # Ensure output_file is a Path object
                output_file = Path(output_file)

                # Create parent directory if it doesn't exist
                output_file.parent.mkdir(parents=True, exist_ok=True)

                # Get period-specific indices
                period = self.indices['period']
                indices_df = self.indices.get(period)

                if indices_df is None:
                    raise ValueError(f"No indices found for period: {period}")

                # Initialize output files list
                output_files = []

                # Save in each format
                for fmt in formats:
                    if fmt.lower() == 'csv':
                        # For CSV, save with index
                        csv_file = output_file.with_suffix('.csv')
                        indices_df.to_csv(csv_file)
                        output_files.append(csv_file)

                    elif fmt.lower() == 'json':
                        # For JSON, include metadata
                        json_file = output_file.with_suffix('.json')

                        # Create JSON structure
                        json_data = {
                            'metadata': self.indices['metadata'],
                            'site': self.site,
                            'period': period,
                            'indices': indices_df.reset_index().to_dict(orient='records')
                        }

                        # Save to file
                        with open(json_file, 'w') as f:
                            json.dump(json_data, f, indent=2, default=str)

                        output_files.append(json_file)

                    elif fmt.lower() in ['h5', 'hdf5']:
                        # For HDF5, include all periods and metadata
                        h5_file = output_file.with_suffix('.h5')

                        with h5py.File(h5_file, 'w') as f:
                            # Add metadata
                            meta_group = f.create_group('metadata')
                            meta_group.attrs['site'] = self.site
                            meta_group.attrs['period'] = period
                            meta_group.attrs['start_date'] = str(self.indices['metadata']['start_date'])
                            meta_group.attrs['end_date'] = str(self.indices['metadata']['end_date'])

                            # Add indices for all calculated periods
                            for p, df in self.indices.items():
                                if isinstance(df, pd.DataFrame):
                                    period_group = f.create_group(p)

                                    # Convert index to datasets
                                    if isinstance(df.index, pd.MultiIndex):
                                        for i, name in enumerate(df.index.names):
                                            # Convert index level to string if needed
                                            idx_values = df.index.get_level_values(i)
                                            if pd.api.types.is_object_dtype(idx_values):
                                                idx_values = idx_values.astype(str)
                                            period_group.create_dataset(f'index_{name}', data=idx_values)
                                    else:
                                        # Single index
                                        idx_values = df.index.values
                                        if pd.api.types.is_object_dtype(idx_values):
                                            idx_values = np.array([str(x) for x in idx_values])
                                        period_group.create_dataset('index', data=idx_values)

                                    # Add each column as a dataset
                                    for col in df.columns:
                                        period_group.create_dataset(col, data=df[col].values)

                        output_files.append(h5_file)

                    else:
                        warnings.warn(f"Unsupported output format: {fmt}")

                return output_files

            def main():
                """Run climate indices calculation from command line"""
                parser = argparse.ArgumentParser(
                    description="Calculate climate indices from meteorological data"
                )

                parser.add_argument('--site', type=str, default='BMTN',
                                    help="Site identifier")
                parser.add_argument('--file', type=str, required=True,
                                    help="Input data file (CSV, MAT, or HDF5)")
                parser.add_argument('--start', type=str,
                                    help="Start date (YYYY-MM-DD)")
                parser.add_argument('--end', type=str,
                                    help="End date (YYYY-MM-DD)")
                parser.add_argument('--period', type=str, default='annual',
                                    choices=['daily', 'monthly', 'seasonal', 'annual'],
                                    help="Time period for aggregation")
                parser.add_argument('--output', type=str,
                                    help="Output file base name (without extension)")
                parser.add_argument('--formats', type=str, default='csv',
                                    help="Output formats, comma-separated (csv,json,h5)")
                parser.add_argument('--lat', type=float,
                                    help="Site latitude (for radiation calculations)")
                parser.add_argument('--lon', type=float,
                                    help="Site longitude (optional)")
                parser.add_argument('--indices', type=str,
                                    help="Specific indices to calculate, comma-separated")
                parser.add_argument('--time-col', type=str, default='TimestampCollected',
                                    help="Name of timestamp column in input data")

                args = parser.parse_args()

                # Initialize calculator
                calculator = ClimateIndices(
                    site=args.site,
                    time_col=args.time_col,
                    lat=args.lat,
                    lon=args.lon
                )

                # Load data
                print(f"Loading data from {args.file}...")
                success = calculator.load_data(
                    data_file=args.file,
                    start_date=args.start,
                    end_date=args.end
                )

                if not success:
                    print("Error loading data. Exiting.")
                    sys.exit(1)

                # Set variable mapping
                print("Detecting variables in data...")
                calculator.set_variable_mapping()

                # Calculate indices
                print(f"Calculating {args.period} climate indices...")
                custom_indices = args.indices.split(',') if args.indices else None
                indices = calculator.calculate_indices(
                    period=args.period,
                    custom_indices=custom_indices
                )

                # Save results
                formats = args.formats.split(',')
                print(f"Saving results in formats: {formats}")
                output_files = calculator.save_indices(
                    output_file=args.output,
                    formats=formats
                )

                print(f"Saved indices to: {', '.join(str(f) for f in output_files)}")

            if __name__ == "__main__":
                main()