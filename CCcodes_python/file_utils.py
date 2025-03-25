"""
file_utils.py

Utility functions for loading and saving climate data files in different formats.
Provides standardized interfaces for file operations across the codebase.

Functions:
- load_data: Loads data from various file formats (CSV, MATLAB, HDF5, etc.)
- save_data: Saves data to various file formats
- convert_file_format: Converts between different file formats
- list_files: Lists files in a directory matching a pattern
- get_site_metadata: Gets metadata for a specific site

Author: Climate Data Team
Date: 2025-03-24
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import h5py
import warnings
import json
import glob
from datetime import datetime


def load_data(file_path, file_type=None, time_col='TimestampCollected', parse_dates=True, **kwargs):
    """
    Load climate data from various file formats with automatic format detection

    Parameters:
    -----------
    file_path : str or Path
        Path to data file
    file_type : str, optional
        File type override ('csv', 'mat', 'h5', 'pkl', 'npz')
        If None, determined from file extension
    time_col : str, optional
        Column name containing timestamps for parsing
    parse_dates : bool, optional
        Whether to convert time columns to datetime objects
    **kwargs : dict
        Additional parameters passed to the specific loader function

    Returns:
    --------
    data : pandas DataFrame, numpy array, or dict
        Loaded data structure

    Raises:
    -------
    FileNotFoundError: If file doesn't exist
    ValueError: If file format is unknown or unsupported

    Examples:
    ---------
    >>> # Load CSV data
    >>> df = load_data('BMTN_daily_data.csv')
    >>>
    >>> # Load MATLAB file with specific options
    >>> data = load_data('BMTN_data.mat', time_col='Date')
    >>>
    >>> # Load HDF5 file without date parsing
    >>> data = load_data('climate_indices.h5', parse_dates=False)
    """
    file_path = Path(file_path)

    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Determine file type if not provided
    if file_type is None:
        file_type = file_path.suffix.lower().lstrip('.')

    # Load based on file type
    try:
        if file_type in ['csv', 'txt']:
            # CSV/text file loading
            date_cols = [time_col] if parse_dates and time_col else None
            df = pd.read_csv(file_path, parse_dates=date_cols, **kwargs)

            # Parse date columns if not done during loading
            if parse_dates and time_col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

            return df

        elif file_type == 'mat':
            # MATLAB .mat file loading
            import scipy.io as sio
            try:
                # Try loading the file with scipy.io
                mat_data = sio.loadmat(str(file_path), **kwargs)

                # Check for common structures in MATLAB climate data
                if 'TT_dailyMES' in mat_data:
                    return _process_matlab_daily(mat_data, time_col, parse_dates)
                elif 'TT_hourly' in mat_data:
                    return _process_matlab_hourly(mat_data, time_col, parse_dates)
                elif 'DATAannual' in mat_data:
                    return _process_matlab_annual(mat_data, time_col, parse_dates)
                elif 'DATAmonthly' in mat_data:
                    return _process_matlab_monthly(mat_data, time_col, parse_dates)
                else:
                    # Return the raw data if no known structure is found
                    return mat_data
            except Exception as e:
                warnings.warn(f"Error loading MATLAB file: {e}. Trying h5py...")
                # If scipy.io fails, try h5py for MATLAB HDF5 file
                with h5py.File(file_path, 'r') as f:
                    # Create a dictionary with arrays for each variable
                    data_dict = {key: np.array(f[key]) for key in f.keys()}
                    return data_dict

        elif file_type in ['h5', 'hdf5']:
            # HDF5 file loading
            with h5py.File(file_path, 'r') as f:
                # If the file has a specific structure, process accordingly
                if 'data' in f:
                    # Complex HDF5 structure
                    data_dict = {}
                    # Extract metadata
                    if 'years' in f:
                        data_dict['years'] = np.array(f['years'])
                    if 'months' in f:
                        data_dict['months'] = np.array(f['months'])
                    if 'vars' in f:
                        data_dict['vars'] = np.array(f['vars'])

                    # Extract data groups
                    data_dict['data'] = {}
                    for key in f['data'].keys():
                        data_dict['data'][key] = {}
                        for subkey in f['data'][key].keys():
                            data_dict['data'][key][subkey] = np.array(f['data'][key][subkey])

                    return data_dict
                else:
                    # Simple HDF5 structure - convert to DataFrame if possible
                    data_dict = {key: np.array(f[key]) for key in f.keys()}

                    # Try to convert to DataFrame if all arrays have the same length
                    lengths = [len(arr) if hasattr(arr, '__len__') else 1 for arr in data_dict.values()]
                    if len(set(lengths)) <= 1:  # At most one unique length
                        try:
                            df = pd.DataFrame(data_dict)
                            if parse_dates and time_col in df.columns:
                                # Convert timestamp strings to datetime objects
                                if isinstance(df[time_col].iloc[0], (bytes, str)):
                                    if isinstance(df[time_col].iloc[0], bytes):
                                        df[time_col] = df[time_col].apply(
                                            lambda x: x.decode('utf-8') if isinstance(x, bytes) else x
                                        )
                                    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                            return df
                        except:
                            # Return dictionary if DataFrame conversion fails
                            return data_dict
                    else:
                        # Return dictionary for arrays of different lengths
                        return data_dict

        elif file_type == 'pkl':
            # Pickle file loading
            return pd.read_pickle(file_path, **kwargs)

        elif file_type == 'npz':
            # NumPy compressed file loading
            with np.load(file_path, allow_pickle=True, **kwargs) as data:
                return {key: data[key] for key in data.files}

        elif file_type == 'json':
            # JSON file loading
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Try to convert to DataFrame if the structure is appropriate
            if isinstance(data, dict) and 'indices' in data and isinstance(data['indices'], list):
                df = pd.DataFrame(data['indices'])
                if parse_dates and time_col in df.columns:
                    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                return df
            else:
                return data

        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    except Exception as e:
        raise ValueError(f"Error loading file {file_path}: {e}")


def _process_matlab_daily(mat_data, time_col='TimestampCollected', parse_dates=True):
    """Helper function to process MATLAB daily data structure"""
    data = mat_data['TT_dailyMES']

    # Convert to DataFrame based on structure
    if isinstance(data, np.ndarray) and hasattr(data, 'dtype') and data.dtype.names:
        df = pd.DataFrame()
        for name in data.dtype.names:
            # Handle different array dimensions
            if data[name].ndim > 1:
                df[name] = data[name].flatten()
            else:
                df[name] = data[name]

        # Parse date column if needed
        if parse_dates and time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

        return df
    else:
        # If structure doesn't match expectations, return raw data
        return mat_data


def _process_matlab_hourly(mat_data, time_col='TimestampCollected', parse_dates=True):
    """Helper function to process MATLAB hourly data structure"""
    if 'TT_hourly' in mat_data:
        data = mat_data['TT_hourly']
    else:
        # Try to find an appropriate hourly data structure
        for key in mat_data.keys():
            if 'hourly' in key.lower() and isinstance(mat_data[key], np.ndarray):
                data = mat_data[key]
                break
        else:
            # Return raw data if no appropriate structure found
            return mat_data

    # Convert to DataFrame similar to daily data
    if isinstance(data, np.ndarray) and hasattr(data, 'dtype') and data.dtype.names:
        df = pd.DataFrame()
        for name in data.dtype.names:
            # Handle different array dimensions
            if data[name].ndim > 1:
                df[name] = data[name].flatten()
            else:
                df[name] = data[name]

        # Parse date column if needed
        if parse_dates and time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

        return df
    else:
        # If structure doesn't match expectations, return raw data
        return mat_data


def _process_matlab_annual(mat_data, time_col=None, parse_dates=True):
    """Helper function to process MATLAB annual data structure"""
    if 'DATAannual' not in mat_data:
        # Try to find an appropriate annual data structure
        for key in mat_data.keys():
            if 'annual' in key.lower() and isinstance(mat_data[key], np.ndarray):
                data = mat_data[key]
                break
        else:
            # Return raw data if no appropriate structure found
            return mat_data
    else:
        data = mat_data['DATAannual']

    # Extract fields from the structure
    result = {}

    # Try to extract year, var, and data fields
    if hasattr(data, 'dtype') and data.dtype.names:
        for field in data.dtype.names:
            field_data = data[field]
            if field_data.ndim > 1:
                field_data = field_data.flatten()
            result[field] = field_data[0] if field_data.size == 1 else field_data

    return result


def _process_matlab_monthly(mat_data, time_col=None, parse_dates=True):
    """Helper function to process MATLAB monthly data structure"""
    if 'DATAmonthly' not in mat_data:
        # Try to find an appropriate monthly data structure
        for key in mat_data.keys():
            if 'monthly' in key.lower() and isinstance(mat_data[key], np.ndarray):
                data = mat_data[key]
                break
        else:
            # Return raw data if no appropriate structure found
            return mat_data
    else:
        data = mat_data['DATAmonthly']

    # Extract fields from the structure (similar to annual)
    result = {}

    # Try to extract year, month, var, and data fields
    if hasattr(data, 'dtype') and data.dtype.names:
        for field in data.dtype.names:
            field_data = data[field]
            if field_data.ndim > 1:
                field_data = field_data.flatten()
            result[field] = field_data[0] if field_data.size == 1 else field_data

    return result


def save_data(data, file_path, file_type=None, **kwargs):
    """
    Save climate data to various file formats

    Parameters:
    -----------
    data : pandas DataFrame, numpy array, or dict
        Data to save
    file_path : str or Path
        Path to save file
    file_type : str, optional
        File type ('csv', 'mat', 'h5', 'pkl', 'npz')
        If None, determined from file extension
    **kwargs : dict
        Additional parameters passed to the specific saver function

    Returns:
    --------
    success : bool
        True if save was successful

    Raises:
    -------
    ValueError: If file format is unknown or unsupported

    Examples:
    ---------
    >>> # Save DataFrame to CSV
    >>> df = pd.DataFrame({'Temperature': [25.3, 26.1, 24.8]})
    >>> save_data(df, 'temperature_data.csv')
    >>>
    >>> # Save data dictionary to HDF5 with compression
    >>> data_dict = {'years': [2020, 2021, 2022], 'values': [10, 12, 9]}
    >>> save_data(data_dict, 'data.h5', compression='gzip')
    """
    file_path = Path(file_path)

    # Create directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine file type if not provided
    if file_type is None:
        file_type = file_path.suffix.lower().lstrip('.')

    try:
        if file_type in ['csv', 'txt']:
            # Handle DataFrame
            if isinstance(data, pd.DataFrame):
                data.to_csv(file_path, **kwargs)
            # Handle numpy array
            elif isinstance(data, np.ndarray):
                np.savetxt(file_path, data, delimiter=',', **kwargs)
            # Handle dict
            elif isinstance(data, dict):
                pd.DataFrame(data).to_csv(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported data type for CSV: {type(data)}")

        elif file_type == 'mat':
            # MATLAB .mat file saving
            import scipy.io as sio
            if isinstance(data, pd.DataFrame):
                # Convert DataFrame to dict of arrays
                data_dict = {col: data[col].values for col in data.columns}
                sio.savemat(file_path, data_dict, **kwargs)
            elif isinstance(data, dict):
                sio.savemat(file_path, data, **kwargs)
            else:
                raise ValueError(f"Unsupported data type for MAT: {type(data)}")

        elif file_type in ['h5', 'hdf5']:
            # HDF5 file saving
            with h5py.File(file_path, 'w') as f:
                if isinstance(data, pd.DataFrame):
                    # Save each column as a dataset
                    for col in data.columns:
                        # Convert datetime columns to strings
                        if pd.api.types.is_datetime64_any_dtype(data[col]):
                            f.create_dataset(col, data=np.array(data[col].astype(str)), **kwargs)
                        else:
                            f.create_dataset(col, data=np.array(data[col]), **kwargs)
                elif isinstance(data, dict):
                    # Handle nested dictionary structure
                    def add_dict_to_hdf5(group, d):
                        for key, value in d.items():
                            if isinstance(value, dict):
                                subgroup = group.create_group(str(key))
                                add_dict_to_hdf5(subgroup, value)
                            else:
                                # Handle different types of arrays/values
                                if isinstance(value, (pd.DataFrame, pd.Series)):
                                    group.create_dataset(str(key), data=value.values, **kwargs)
                                elif isinstance(value, (list, np.ndarray, int, float, str)):
                                    if isinstance(value, list):
                                        value = np.array(value)
                                    # Convert string arrays correctly
                                    if isinstance(value, np.ndarray) and value.dtype.kind in ['U', 'S']:
                                        value = np.array(value, dtype='S')
                                    group.create_dataset(str(key), data=value, **kwargs)

                    # Start the recursion
                    add_dict_to_hdf5(f, data)
                else:
                    raise ValueError(f"Unsupported data type for HDF5: {type(data)}")

        elif file_type == 'pkl':
            # Pickle file saving
            if isinstance(data, pd.DataFrame):
                data.to_pickle(file_path, **kwargs)
            else:
                import pickle
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f, **kwargs)

        elif file_type == 'npz':
            # NumPy compressed file saving
            if isinstance(data, dict):
                np.savez_compressed(file_path, **data)
            elif isinstance(data, pd.DataFrame):
                np.savez_compressed(file_path, **{col: data[col].values for col in data.columns})
            else:
                raise ValueError(f"Unsupported data type for NPZ: {type(data)}")

        elif file_type == 'json':
            # JSON file saving
            if isinstance(data, pd.DataFrame):
                # Convert to records format
                records = data.to_dict(orient='records')
                with open(file_path, 'w') as f:
                    json.dump({'indices': records}, f, indent=2, default=str)
            elif isinstance(data, dict):
                # Handle datetime objects
                def json_serial(obj):
                    if isinstance(obj, (datetime, np.datetime64)):
                        return obj.isoformat()
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        return float(obj)
                    raise TypeError(f"Type {type(obj)} not serializable")

                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2, default=json_serial)
            else:
                raise ValueError(f"Unsupported data type for JSON: {type(data)}")

        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        return True

    except Exception as e:
        warnings.warn(f"Error saving file {file_path}: {e}")
        return False


def convert_file_format(input_file, output_file, **kwargs):
    """
    Convert climate data between different file formats

    Parameters:
    -----------
    input_file : str or Path
        Path to input file
    output_file : str or Path
        Path to output file
    **kwargs : dict
        Additional parameters passed to load_data and save_data

    Returns:
    --------
    success : bool
        True if conversion was successful

    Examples:
    ---------
    >>> # Convert MATLAB file to CSV
    >>> convert_file_format('BMTN_data.mat', 'BMTN_data.csv')
    >>>
    >>> # Convert CSV to HDF5 with specific time column
    >>> convert_file_format('station_data.csv', 'station_data.h5', time_col='Date')
    """
    try:
        # Load data from input file
        data = load_data(input_file, **kwargs)

        # Save data to output file
        return save_data(data, output_file, **kwargs)

    except Exception as e:
        warnings.warn(f"Error converting file format: {e}")
        return False


def list_files(directory, pattern="*", recursive=False, sort=True):
    """
    List all climate data files in a directory that match a pattern

    Parameters:
    -----------
    directory : str or Path
        Directory to search
    pattern : str, optional
        File pattern to match (e.g., "*.csv", "BMTN_*.mat")
    recursive : bool, optional
        Whether to search recursively in subdirectories
    sort : bool, optional
        Whether to sort the files alphabetically

    Returns:
    --------
    files : list
        List of Path objects for matching files

    Examples:
    ---------
    >>> # List all CSV files in the current directory
    >>> csv_files = list_files('.', '*.csv')
    >>>
    >>> # Find all data files for a specific site across subdirectories
    >>> site_files = list_files('data_dir', 'BMTN_*.*', recursive=True)
    """
    directory = Path(directory)

    # Find all files matching the pattern
    if recursive:
        # Use glob with recursive pattern
        if '**' not in str(pattern):
            # Add ** for recursive search if not already present
            pattern = f"**/{pattern}"
        files = list(directory.glob(pattern))
    else:
        files = list(directory.glob(pattern))

    # Sort if requested
    if sort:
        files = sorted(files)

    return files


def get_site_metadata(site_code, metadata_file=None):
    """
    Get metadata for a specific site

    Parameters:
    -----------
    site_code : str
        Site identifier code
    metadata_file : str or Path, optional
        Path to metadata file (if None, searches in common locations)

    Returns:
    --------
    metadata : dict
        Dictionary containing site metadata

    Examples:
    ---------
    >>> # Get metadata for a specific site
    >>> bmtn_meta = get_site_metadata('BMTN')
    >>> print(f"Site location: {bmtn_meta['latitude']}, {bmtn_meta['longitude']}")
    """
    # If metadata file is provided, use it
    if metadata_file is not None:
        metadata_path = Path(metadata_file)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    else:
        # Search in common locations
        search_paths = [
            Path.cwd() / 'metadata.csv',
            Path.cwd() / 'data' / 'metadata.csv',
            Path.cwd() / 'sites.csv',
            Path.cwd() / 'station_info.csv',
            Path.cwd().parent / 'metadata' / 'sites.csv'
        ]

        # Find first existing file
        for path in search_paths:
            if path.exists():
                metadata_path = path
                break
        else:
            raise FileNotFoundError("No metadata file found in common locations")

    # Load metadata file
    try:
        meta_df = pd.read_csv(metadata_path)
    except Exception as e:
        raise ValueError(f"Error loading metadata file: {e}")

    # Find the site in the metadata
    site_col = None
    for col in meta_df.columns:
        if 'site' in col.lower() or 'code' in col.lower() or 'id' in col.lower():
            if site_code in meta_df[col].values:
                site_col = col
                break

    if site_col is None:
        raise ValueError(f"Site {site_code} not found in metadata file")

    # Extract metadata for the site
    site_row = meta_df[meta_df[site_col] == site_code].iloc[0]

    # Convert to dictionary
    metadata = site_row.to_dict()

    return metadata


def test_file_utils():
    """Test the file utility functions with sample data"""
    # Create sample data
    sample_data = pd.DataFrame({
        'Date': pd.date_range(start='2023-01-01', periods=10),
        'Temperature': np.random.normal(20, 5, 10),
        'Precipitation': np.random.exponential(scale=5, size=10)
    })

    # Test directory
    test_dir = Path('temp_test_files')
    test_dir.mkdir(exist_ok=True)

    try:
        # Test save_data with different formats
        formats = ['csv', 'pkl', 'h5', 'npz']
        saved_files = []

        for fmt in formats:
            file_path = test_dir / f"test_data.{fmt}"
            success = save_data(sample_data, file_path)

            if success:
                saved_files.append(file_path)
                print(f"Successfully saved data to {file_path}")

        # Test load_data with each saved file
        for file_path in saved_files:
            data = load_data(file_path)

            if isinstance(data, pd.DataFrame):
                print(f"Successfully loaded {file_path}, shape: {data.shape}")
            else:
                print(f"Loaded {file_path}, type: {type(data)}")

        # Test file conversion
        convert_file_format(saved_files[0], test_dir / "converted_data.h5")
        print(f"Converted {saved_files[0]} to HDF5")

        # Test list_files
        files = list_files(test_dir, "*.csv")
        print(f"Found {len(files)} CSV files in test directory")

    finally:
        # Clean up test files
        for file_path in test_dir.glob("*"):
            file_path.unlink()

        test_dir.rmdir()
        print("Cleaned up test files")


if __name__ == "__main__":
    test_file_utils()