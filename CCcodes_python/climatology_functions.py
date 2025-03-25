"""
climatology_functions.py

Functions for climate data analysis including percentile calculations, trend analysis,
and climate indices calculations. These implementations follow the methodologies 
described in the European Climate Assessment & Dataset (ECA&D) project.

References:
- Klein Tank, A.M.G., et al., 2009: Daily dataset of 20th-century surface air temperature 
  and precipitation series for the European Climate Assessment. Int. J. Climatol., 29, 1141-1453.
- Zhang, X., et al., 2005: Avoiding inhomogeneity in percentile-based indices of temperature 
  extremes. Journal of Climate, 18, 1641-1651.

Author: Climate Data Team
Date: 2025-03-24
"""

import numpy as np
import pandas as pd
from datetime import datetime
import scipy.stats as stats
import warnings


def calculate_percentiles(data, base_period_start=1961, base_period_end=1990,
                          percentiles=[10, 25, 50, 75, 90, 95, 99]):
    """
    Calculate percentiles according to the method described in ECA&D document.
    Uses a 5-day window centered on each calendar day and a bootstrap approach
    to avoid inhomogeneities at the boundaries of the base period.

    Parameters:
    -----------
    data : pandas DataFrame
        DataFrame containing the climate data
    base_period_start : int
        Start year of the base period
    base_period_end : int
        End year of the base period
    percentiles : list of int
        Percentiles to calculate (1-99)

    Returns:
    --------
    percentiles_dict : dict
        Dictionary containing the requested percentiles for each calendar day

    Examples:
    ---------
    >>> # Calculate 10th and 90th percentiles for temperature data
    >>> df = pd.read_csv('station_data.csv', parse_dates=['date'])
    >>> percentiles = calculate_percentiles(df, 1981, 2010, [10, 90])
    >>> 
    >>> # Use percentiles to identify extreme temperature days
    >>> df['is_extreme_cold'] = df.apply(
    ...     lambda x: x['temperature'] < percentiles['10th'][(x['date'].strftime('%m-%d'), 'temperature')],
    ...     axis=1
    ... )
    """
    # Validate inputs
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame")

    if base_period_end <= base_period_start:
        raise ValueError("End year must be greater than start year")

    if not all(1 <= p <= 99 for p in percentiles):
        raise ValueError("Percentiles must be between 1 and 99")

    # Create date column if it doesn't exist
    if 'date' not in data.columns:
        if isinstance(data.index, pd.DatetimeIndex):
            data = data.reset_index().rename(columns={'index': 'date'})
        else:
            raise ValueError("Data must have a 'date' column or a datetime index")

    # Ensure date is datetime
    if not pd.api.types.is_datetime64_dtype(data['date']):
        try:
            data['date'] = pd.to_datetime(data['date'])
        except:
            raise ValueError("Could not convert 'date' column to datetime")

    # Filter data for base period
    base_period = data[(data['date'].dt.year >= base_period_start) &
                       (data['date'].dt.year <= base_period_end)]

    if len(base_period) == 0:
        raise ValueError(f"No data found in base period {base_period_start}-{base_period_end}")

    # Create day-of-year identifier (ignoring year)
    base_period['day_of_year'] = base_period['date'].dt.strftime('%m-%d')

    # Initialize percentile dictionaries
    percentiles_dict = {f'{p}th': {} for p in percentiles}

    # Get unique days of year
    unique_days = np.unique(base_period['day_of_year'])

    # Process each calendar day
    for day in unique_days:
        # For each calendar day, extract 5-day window
        month, day_num = map(int, day.split('-'))

        # Create 5-day window masks centered on the calendar day
        window_masks = []
        for offset in range(-2, 3):
            # Calculate the date with offset (handling month boundaries)
            try:
                # Use a leap year to handle Feb 29
                target_date = pd.Timestamp(year=2000, month=month, day=day_num) + pd.Timedelta(days=offset)
                target_day = target_date.strftime('%m-%d')
                day_mask = base_period['day_of_year'] == target_day
                window_masks.append(day_mask)
            except ValueError:
                # Skip invalid dates
                continue

        # Combine masks with OR
        if window_masks:
            window_mask = np.logical_or.reduce(window_masks)
            window_data = base_period[window_mask]
        else:
            # Use just the current day if window creation failed
            window_data = base_period[base_period['day_of_year'] == day]

        # Following the method from Zhang et al. (2005)
        # For each year in the base period, calculate percentiles excluding that year
        for var in data.columns:
            if var in ['date', 'day_of_year', 'year', 'month', 'day']:
                continue

            years = np.unique(window_data['date'].dt.year)
            percentile_estimates = {p: [] for p in percentiles}

            for year in years:
                # Construct a block by excluding the target year
                year_mask = window_data['date'].dt.year == year
                out_of_base_data = window_data[year_mask][var].values
                in_base_data = window_data[~year_mask]

                if len(in_base_data) < 5:  # Need minimum data for estimation
                    continue

                # Calculate percentiles from in-base data
                for p in percentiles:
                    p_value = np.percentile(in_base_data[var].values, p)

                    # Count exceedances in out-of-base data
                    if p < 50:
                        # For lower percentiles, count values below threshold
                        exceed = np.sum(out_of_base_data < p_value) / len(out_of_base_data)
                    else:
                        # For upper percentiles, count values above threshold
                        exceed = np.sum(out_of_base_data > p_value) / len(out_of_base_data)

                    percentile_estimates[p].append(exceed)

            # Average the estimates for each percentile
            for p in percentiles:
                if percentile_estimates[p]:
                    percentile_key = f'{p}th'
                    percentiles_dict[percentile_key][(day, var)] = np.mean(percentile_estimates[p])

    return percentiles_dict


def lowess_smoother(x, y, f=None, n_iter=3):
    """
    Implement LOWESS (locally weighted scatterplot smoothing) as described in ECA&D.
    This is a robust, non-parametric regression method that combines multiple
    linear regression models in a k-nearest-neighbor-based meta-model.

    Parameters:
    -----------
    x : array-like
        Independent variable values
    y : array-like
        Dependent variable values
    f : float
        Smoothing span (fraction of points to use). If None, calculated as 30/len(x).
    n_iter : int
        Number of robustifying iterations

    Returns:
    --------
    y_smooth : array-like
        Smoothed values

    Examples:
    ---------
    >>> import numpy as np
    >>> x = np.linspace(0, 2*np.pi, 100)
    >>> y = np.sin(x) + np.random.normal(0, 0.2, 100)  # Noisy sine wave
    >>> y_smooth = lowess_smoother(x, y, f=0.2)
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) != len(y):
        raise ValueError("Input arrays must have the same length")

    # Handle NaN values
    valid = ~(np.isnan(x) | np.isnan(y))
    if not np.all(valid):
        x = x[valid]
        y = y[valid]
        if len(x) < 3:
            return np.full_like(y, np.nan)

    if len(x) < 3:
        warnings.warn("Not enough valid data points for smoothing")
        return np.full_like(y, np.nan)

    if f is None:
        f = min(1.0, 30.0 / len(x))

    n = len(x)
    r = int(np.ceil(f * n))

    # Initial fit
    y_smooth = np.zeros_like(y)

    # Use vectorized operations where possible
    for i in range(n):
        distances = np.abs(x - x[i])
        sorted_idx = np.argsort(distances)
        neighbor_idx = sorted_idx[:r]
        max_dist = distances[sorted_idx[r - 1]]

        # Compute weights using tricube function
        weights = np.zeros(n)
        dist_ratio = distances[neighbor_idx] / max_dist
        weights[neighbor_idx] = np.where(
            dist_ratio < 1,
            (1 - dist_ratio ** 3) ** 3,
            0
        )

        # Weighted linear regression
        sum_weight = np.sum(weights)
        if sum_weight > 0:
            weighted_x = np.sum(weights * x) / sum_weight
            weighted_y = np.sum(weights * y) / sum_weight
            weighted_xy = np.sum(weights * x * y) / sum_weight
            weighted_xx = np.sum(weights * x * x) / sum_weight

            try:
                b1 = (weighted_xy - weighted_x * weighted_y) / (weighted_xx - weighted_x ** 2)
                b0 = weighted_y - b1 * weighted_x
                y_smooth[i] = b0 + b1 * x[i]
            except ZeroDivisionError:
                y_smooth[i] = weighted_y
        else:
            y_smooth[i] = y[i]

    # Robustifying iterations
    for iteration in range(n_iter):
        residuals = y - y_smooth
        s = np.median(np.abs(residuals))

        if s == 0:  # Perfect fit or not enough variation
            break

        # Bisquare function for robust weights
        robust_weights = np.zeros_like(residuals)
        u = residuals / (6.0 * s)
        mask = np.abs(u) < 1
        robust_weights[mask] = (1 - u[mask] ** 2) ** 2

        # Refit with robust weights
        for i in range(n):
            distances = np.abs(x - x[i])
            sorted_idx = np.argsort(distances)
            neighbor_idx = sorted_idx[:r]
            max_dist = distances[sorted_idx[r - 1]]

            # Compute weights using tricube function and robust weights
            weights = np.zeros(n)
            for j in neighbor_idx:
                dist_ratio = distances[j] / max_dist
                weights[j] = ((1 - dist_ratio ** 3) ** 3 if dist_ratio < 1 else 0) * robust_weights[j]

            # Weighted linear regression
            sum_weight = np.sum(weights)
            if sum_weight > 0:
                weighted_x = np.sum(weights * x) / sum_weight
                weighted_y = np.sum(weights * y) / sum_weight
                weighted_xy = np.sum(weights * x * y) / sum_weight
                weighted_xx = np.sum(weights * x * x) / sum_weight

                try:
                    if (weighted_xx - weighted_x ** 2) != 0:
                        b1 = (weighted_xy - weighted_x * weighted_y) / (weighted_xx - weighted_x ** 2)
                        b0 = weighted_y - b1 * weighted_x
                        y_smooth[i] = b0 + b1 * x[i]
                    else:
                        y_smooth[i] = weighted_y
                except:
                    y_smooth[i] = weighted_y
            else:
                y_smooth[i] = y[i]

    return y_smooth


def calculate_trend(data, auto_correlation=True, time_col=None, value_col=None):
    """
    Calculate trend value using least squares and adjust significance for autocorrelation.
    This implementation accounts for autocorrelation in the residuals which can
    artificially inflate the significance of trends in climate time series.

    Parameters:
    -----------
    data : pandas DataFrame or Series
        DataFrame with time as index and values to analyze, or Series of values
    auto_correlation : bool
        Whether to adjust for autocorrelation in the significance calculation
    time_col : str, optional
        Column name for time values if using specific columns from DataFrame
    value_col : str, optional
        Column name for data values if using specific columns from DataFrame

    Returns:
    --------
    result : dict
        Dictionary containing trend value, significance, and other statistics

    Examples:
    ---------
    >>> # Calculate trend in annual temperature data
    >>> df = pd.DataFrame({
    ...     'year': range(1980, 2021),
    ...     'temperature': [15 + 0.02*i + np.random.normal(0, 0.5) for i in range(41)]
    ... })
    >>> trend = calculate_trend(df, time_col='year', value_col='temperature')
    >>> print(f"Temperature trend: {trend['trend']:.3f}°C/year (p={trend['p_value']:.3f})")
    """
    # Extract x (time) and y (values) based on input type
    if isinstance(data, pd.DataFrame):
        if time_col is not None and value_col is not None:
            x = np.array(data[time_col])
            y = np.array(data[value_col])
            time_values = data[time_col].values
        elif isinstance(data.index, pd.DatetimeIndex):
            # Use index as time and specified column as values
            if value_col is not None:
                y = data[value_col].values
            else:
                # Assume single column of values
                y = data.iloc[:, 0].values

            # Convert datetime index to numerical values for regression
            x = np.arange(len(data))
            time_values = data.index.to_numpy()
        else:
            raise ValueError("With DataFrame input, must specify time_col and value_col, or use DatetimeIndex")
    elif isinstance(data, pd.Series):
        y = data.values
        x = np.arange(len(data))
        time_values = data.index.to_numpy()
    else:
        raise TypeError("Input must be a pandas DataFrame or Series")

    # Filter out NaN values
    mask = ~np.isnan(y)
    x_filtered = x[mask]
    y_filtered = y[mask]
    time_filtered = time_values[mask] if len(time_values) == len(mask) else x_filtered

    if len(x_filtered) < 3:
        return {
            'trend': np.nan,
            'significance': np.nan,
            'p_value': np.nan,
            'equivalent_sample_size': np.nan,
            'r_squared': np.nan,
            'std_err': np.nan,
            'time_values': time_filtered
        }

    # Calculate linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_filtered, y_filtered)

    # Calculate residuals
    y_fit = intercept + slope * x_filtered
    residuals = y_filtered - y_fit

    # Calculate R-squared
    r_squared = r_value ** 2

    # Calculate squared sum of errors
    sigma_e_squared = np.sum(residuals ** 2) / (len(x_filtered) - 2)

    # Calculate SXX
    sxx = np.sum((x_filtered - np.mean(x_filtered)) ** 2)

    # Calculate SSR (sum of squares regression)
    ssr = np.sum((y_fit - np.mean(y_filtered)) ** 2)

    # Calculate F-statistic
    f_statistic = ssr / sigma_e_squared

    if auto_correlation:
        # Adjust for autocorrelation using equivalent sample size
        # Calculate autocorrelation function
        acf = np.correlate(residuals, residuals, mode='full')
        acf = acf[len(acf) // 2:] / acf[len(acf) // 2]  # Normalize

        # Calculate equivalent sample size
        n = len(x_filtered)
        n_eff = n

        # The sum term in the effective sample size calculation
        # Sum of rk * (1 - k/n) for k=1 to min(n-1, max_lag)
        max_lag = min(n - 1, 20)  # Use up to 20 lags or n-1, whichever is smaller
        sum_term = 0
        for k in range(1, max_lag + 1):
            # Apply the weighting function (1 - k/n)
            weight = 1 - k / n
            sum_term += 2 * weight * acf[k]

        # Calculate effective sample size
        n_eff = n / (1 + sum_term)

        # Adjust p-value for effective sample size
        if n_eff > 2:
            t_stat = slope / (np.sqrt(sigma_e_squared / sxx))
            p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), int(n_eff) - 2))
        else:
            n_eff = np.nan
            p_value = np.nan
    else:
        n_eff = len(x_filtered)

    return {
        'trend': slope,
        'intercept': intercept,
        'significance': p_value < 0.05,
        'p_value': p_value,
        'equivalent_sample_size': n_eff,
        'r_squared': r_squared,
        'std_err': std_err,
        'time_values': time_filtered
    }


def find_consecutive_days(condition):
    """
    Find runs of consecutive True values in a Series or array

    Parameters:
    -----------
    condition : pandas Series or array-like
        Boolean series to find runs in

    Returns:
    --------
    run_lengths : array
        Array containing lengths of all runs

    Examples:
    ---------
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Find runs of rainy days
    >>> rain = pd.Series([0, 2, 5, 0, 0, 1, 2, 3, 0, 1])
    >>> rainy_days = rain > 0
    >>> runs = find_consecutive_days(rainy_days)
    >>> print(f"Run lengths: {runs}")
    >>> print(f"Maximum consecutive rainy days: {runs.max() if len(runs) > 0 else 0}")
    """
    # Convert to numpy array for processing
    if isinstance(condition, pd.Series):
        condition_arr = condition.values
    else:
        condition_arr = np.asarray(condition, dtype=bool)

    # Handle empty or all-False arrays
    if len(condition_arr) == 0 or not np.any(condition_arr):
        return np.array([], dtype=int)

    # Find transitions
    transitions = np.diff(np.concatenate(([False], condition_arr, [False])))

    # Start positions
    run_starts = np.where(transitions == 1)[0]

    # End positions
    run_ends = np.where(transitions == -1)[0]

    # Calculate run lengths
    run_lengths = run_ends - run_starts

    return run_lengths


def count_exceedances(data, threshold, mode='above'):
    """
    Count days exceeding a threshold

    Parameters:
    -----------
    data : pandas Series or array-like
        Data to check
    threshold : float
        Threshold value
    mode : str
        'above' to count values > threshold, 'below' for values < threshold

    Returns:
    --------
    count : int
        Number of days exceeding threshold

    Examples:
    ---------
    >>> # Count hot days (> 30°C)
    >>> temperature = pd.Series([28, 31, 32, 29, 33, 30, 27])
    >>> hot_days = count_exceedances(temperature, 30, 'above')
    >>> print(f"Hot days: {hot_days}")
    """
    if isinstance(data, pd.Series):
        data_arr = data.values
    else:
        data_arr = np.asarray(data)

    # Remove NaNs
    data_arr = data_arr[~np.isnan(data_arr)]

    if mode == 'above':
        return np.sum(data_arr > threshold)
    else:
        return np.sum(data_arr < threshold)


def calculate_climate_indices(data, temp_var='TAIR', precip_var='PRCP', wind_var='WSPD',
                              thresholds=None, base_period=None):
    """
    Calculate climate indices as described in ECA&D document.
    This function calculates a wide range of climate indices related to temperature,
    precipitation, wind, and other meteorological variables.

    Parameters:
    -----------
    data : pandas DataFrame
        DataFrame containing daily climate data
    temp_var : str
        Name of temperature variable in data
    precip_var : str
        Name of precipitation variable in data
    wind_var : str
        Name of wind speed variable in data
    thresholds : dict, optional
        Dictionary containing threshold values for percentile-based indices
    base_period : tuple, optional
        (start_year, end_year) for calculating thresholds if not provided

    Returns:
    --------
    indices : dict
        Dictionary containing calculated climate indices

    Examples:
    ---------
    >>> # Calculate climate indices for a weather station
    >>> df = pd.read_csv('station_data.csv', parse_dates=['date'])
    >>> df = df.set_index('date')
    >>> indices = calculate_climate_indices(df, temp_var='temp', precip_var='prcp')
    >>> print(f"Frost days: {indices['FD']}")
    >>> print(f"Annual precipitation: {indices['PRCPTOT']:.1f} mm")
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame")

    # Ensure data has a datetime index
    if not isinstance(data.index, pd.DatetimeIndex):
        if 'date' in data.columns:
            data = data.set_index('date')
        else:
            raise ValueError("Data must have a datetime index or a 'date' column")

    # Initialize indices dictionary
    indices = {}

    # Calculate thresholds if not provided
    if thresholds is None and base_period is not None:
        start_year, end_year = base_period
        base_data = data[(data.index.year >= start_year) & (data.index.year <= end_year)]
        if len(base_data) > 0:
            # Create a copy with 'date' column for percentile calculation
            base_data_copy = base_data.reset_index()
            base_data_copy.rename(columns={base_data_copy.columns[0]: 'date'}, inplace=True)
            thresholds = calculate_percentiles(base_data_copy, start_year, end_year)

    # Check variable presence
    vars_present = {
        'temp': temp_var in data.columns,
        'temp_max': f"{temp_var}x" in data.columns,
        'temp_min': f"{temp_var}n" in data.columns,
        'precip': precip_var in data.columns,
        'wind': wind_var in data.columns,
        'wind_max': f"{wind_var}x" in data.columns
    }

    # Temperature indices
    if vars_present['temp']:
        # GSL - Growing season length (days)
        temp_season_var = f"{temp_var}_season" if f"{temp_var}_season" in data.columns else temp_var
        temp = data[temp_season_var]

        # Find periods with at least 6 consecutive days > 5°C
        years = np.unique(data.index.year)
        gsl_values = []

        for year in years:
            year_data = data[data.index.year == year]
            if len(year_data) < 100:  # Skip incomplete years
                continue

            # Use numpy for faster processing
            dates = year_data.index
            temps = year_data[temp_season_var].values

            # Find first 6-day spell with temps > 5°C
            growing_start = None
            for i in range(len(temps) - 5):
                if all(temps[i:i + 6] > 5):
                    growing_start = dates[i]
                    break

            # Find first frost (or temp < 5°C) after July 1
            growing_end = None
            if growing_start is not None:
                july_first = pd.Timestamp(year=year, month=7, day=1)
                after_july = dates >= july_first

                if vars_present['temp_min']:
                    for j, date in enumerate(dates[after_july]):
                        idx = np.where(after_july)[0][j]
                        if year_data[f"{temp_var}n"].iloc[idx] < 0:
                            growing_end = date
                            break
                else:
                    # If min temp not available, use daily mean < 5°C
                    for j, date in enumerate(dates[after_july]):
                        idx = np.where(after_july)[0][j]
                        if temps[idx] < 5:
                            growing_end = date
                            break

            # Calculate GSL
            if growing_start and growing_end:
                gsl_values.append((year, (growing_end - growing_start).days))

        if gsl_values:
            indices['GSL'] = dict(gsl_values)

        # SU - Summer days (Tmax > 25°C)
        if vars_present['temp_max']:
            tmax = data[f"{temp_var}x"]
            indices['SU'] = count_exceedances(tmax, 25, 'above')

            # TXx - Maximum value of daily maximum temperature
            indices['TXx'] = tmax.max()

            # Ice days (Tmax < 0°C)
            indices['ID'] = count_exceedances(tmax, 0, 'below')

            if vars_present['temp_min']:
                tmin = data[f"{temp_var}n"]

                # DTR - Mean diurnal temperature range
                dtr = tmax - tmin
                indices['DTR'] = dtr.mean()

                # ETR - Extreme temperature range
                indices['ETR'] = tmax.max() - tmin.min()

        # Frost days (Tmin < 0°C)
        if vars_present['temp_min']:
            tmin = data[f"{temp_var}n"]
            indices['FD'] = count_exceedances(tmin, 0, 'below')

            # TNn - Minimum value of daily minimum temperature
            indices['TNn'] = tmin.min()

            # TR - Tropical nights (Tmin > 20°C)
            indices['TR'] = count_exceedances(tmin, 20, 'above')

            # Percentile-based indices
            if thresholds is not None:
                # Find calendar day key format (e.g., '01-01')
                sample_day = data.index[0].strftime('%m-%d')

                # Check if appropriate thresholds exist
                if '10th' in thresholds:
                    has_tn10p = any((sample_day, f"{temp_var}n") in thresholds['10th'] for sample_day in
                                    [data.index[0].strftime('%m-%d')])
                    if has_tn10p:
                        # TN10p - Cold nights (Tmin < 10th percentile)
                        cold_nights = 0
                        for date, temp in zip(data.index, data[f"{temp_var}n"]):
                            day = date.strftime('%m-%d')
                            threshold_key = (day, f"{temp_var}n")
                            if threshold_key in thresholds['10th'] and temp < thresholds['10th'][threshold_key]:
                                cold_nights += 1
                        indices['TN10p'] = cold_nights

                if '90th' in thresholds:
                    has_tn90p = any((sample_day, f"{temp_var}n") in thresholds['90th'] for sample_day in
                                    [data.index[0].strftime('%m-%d')])
                    if has_tn90p:
                        # TN90p - Warm nights (Tmin > 90th percentile)
                        warm_nights = 0
                        for date, temp in zip(data.index, data[f"{temp_var}n"]):
                            day = date.strftime('%m-%d')
                            threshold_key = (day, f"{temp_var}n")
                            if threshold_key in thresholds['90th'] and temp > thresholds['90th'][threshold_key]:
                                warm_nights += 1
                        indices['TN90p'] = warm_nights

        # Growing degree days
        indices['GD4'] = np.maximum(data[temp_var] - 4, 0).sum()
        indices['GD10'] = np.maximum(data[temp_var] - 10, 0).sum()

        # Heating degree days
        indices['HDD'] = np.maximum(18.3 - data[temp_var], 0).sum()

    # Precipitation indices
    if vars_present['precip']:
        precip = data[precip_var]

        # RR - Total precipitation
        indices['RR'] = precip.sum()

        # RR1 - Number of wet days (precip >= 1mm)
        wet_days = precip >= 1
        indices['RR1'] = wet_days.sum()

        # SDII - Simple daily intensity index (mean precip on wet days)
        if wet_days.sum() > 0:
            indices['SDII'] = precip[wet_days].mean()
        else:
            indices['SDII'] = 0

        # R10 - Number of days with precipitation >= 10mm
        indices['R10'] = count_exceedances(precip, 10, 'above')

        # R20 - Number of days with precipitation >= 20mm
        indices['R20'] = count_exceedances(precip, 20, 'above')

        # CWD - Maximum number of consecutive wet days
        runs = find_consecutive_days(wet_days)
        indices['CWD'] = runs.max() if len(runs) > 0 else 0

        # CDD - Maximum number of consecutive dry days
        dry_runs = find_consecutive_days(~wet_days)
        indices['CDD'] = dry_runs.max() if len(dry_runs) > 0 else 0

        # RX1day - Maximum 1-day precipitation amount
        indices['RX1day'] = precip.max()

        # RX5day - Maximum 5-day precipitation amount
        if len(precip) >= 5:
            rolling_sum = pd.Series(precip).rolling(window=5).sum()
            indices['RX5day'] = rolling_sum.max()

        # R95p - Contribution from very wet days (> 95th percentile)
        if thresholds is not None and '95th' in thresholds:
            # Check if appropriate thresholds exist
            sample_day = data.index[0].strftime('%m-%d')
            has_r95p = any(
                (sample_day, precip_var) in thresholds['95th'] for sample_day in [data.index[0].strftime('%m-%d')])

            if has_r95p:
                r95p_sum = 0
                for date, p in zip(data.index, precip):
                    day = date.strftime('%m-%d')
                    threshold_key = (day, precip_var)
                    if threshold_key in thresholds['95th'] and p > thresholds['95th'][threshold_key]:
                        r95p_sum += p
                indices['R95p'] = r95p_sum

        # PRCPTOT - Annual total precipitation
        indices['PRCPTOT'] = precip.sum()

        # Wind indices
    if vars_present['wind']:
        wind = data[wind_var]

        # FG - Mean wind speed
        indices['FG'] = wind.mean()

        # Wind direction indices
        if 'WDIR' in data.columns:
            wdir = data['WDIR']

            # Calculate days with winds from different directions
            south_wind = (wdir > 135) & (wdir <= 225)
            east_wind = (wdir > 45) & (wdir <= 135)
            west_wind = (wdir > 225) & (wdir <= 315)
            north_wind = (wdir <= 45) | (wdir > 315)

            indices['DDsouth'] = south_wind.sum()
            indices['DDeast'] = east_wind.sum()
            indices['DDwest'] = west_wind.sum()
            indices['DDnorth'] = north_wind.sum()

        # Maximum wind gust
        if vars_present['wind_max']:
            wind_max = data[f"{wind_var}x"]
            indices['FXx'] = wind_max.max()

        # Calculate consecutive frost days if we have minimum temperature
    if vars_present['temp_min']:
        tmin = data[f"{temp_var}n"]
        frost_days = tmin < 0
        frost_runs = find_consecutive_days(frost_days)
        indices['CFD'] = frost_runs.max() if len(frost_runs) > 0 else 0

        # Add basic metadata
    indices['start_date'] = data.index.min()
    indices['end_date'] = data.index.max()
    indices['days_count'] = len(data)

    return indices

def test_climatology_functions():
    """Test the climatology functions with sample data"""
    # Create sample temperature data
    dates = pd.date_range(start='2020-01-01', end='2021-12-31')
    np.random.seed(42)  # For reproducibility

    # Create seasonal temperatures with random variation
    base_temp = 10  # Base temperature
    seasonal_cycle = 15 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
    random_variation = np.random.normal(0, 2, len(dates))
    warming_trend = 0.001 * np.arange(len(dates))  # Slight warming trend

    tmin = base_temp + seasonal_cycle - 5 + random_variation + warming_trend
    tmean = base_temp + seasonal_cycle + random_variation + warming_trend
    tmax = base_temp + seasonal_cycle + 5 + random_variation + warming_trend

    # Create precipitation data (more in summer, less in winter)
    prcp_seasonal = 5 + 10 * np.maximum(0, np.sin(2 * np.pi * np.arange(len(dates)) / 365))
    prcp_random = np.random.exponential(scale=1.0, size=len(dates))
    prcp_occurrence = np.random.binomial(1, 0.3, len(dates))  # 30% chance of rain
    prcp = prcp_occurrence * (prcp_seasonal + prcp_random)

    # Create wind data
    wind_speed = 3 + 2 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) + np.random.normal(0, 1, len(dates))
    wind_dir = np.random.uniform(0, 360, len(dates))
    wind_gust = wind_speed + np.random.exponential(scale=2.0, size=len(dates))

    # Create DataFrame
    df = pd.DataFrame({
        'TAIR': tmean,
        'TAIRx': tmax,
        'TAIRn': tmin,
        'PRCP': prcp,
        'WSPD': wind_speed,
        'WSPDx': wind_gust,
        'WDIR': wind_dir
    }, index=dates)

    # Test percentile calculation
    print("Testing percentile calculation:")
    df_reset = df.reset_index()
    df_reset.rename(columns={'index': 'date'}, inplace=True)
    percentiles = calculate_percentiles(df_reset, 2020, 2021, [10, 90])
    print(
        f"Number of unique calendar days with 10th percentiles: {len(set(k[0] for k in percentiles['10th'].keys()))}")

    # Test trend calculation
    print("\nTesting trend calculation:")
    annual_temp = df.resample('Y').mean()
    trend_result = calculate_trend(annual_temp['TAIR'])
    print(f"Temperature trend: {trend_result['trend']:.6f} °C/year (p={trend_result['p_value']:.4f})")

    # Test climate indices calculation
    print("\nTesting climate indices calculation:")
    indices = calculate_climate_indices(df)

    # Print selected indices
    print(f"Frost Days (FD): {indices['FD']}")
    print(f"Summer Days (SU): {indices['SU']}")
    print(f"Growing Degree Days (GD4): {indices['GD4']:.1f}")
    print(f"Total Precipitation: {indices['PRCPTOT']:.1f} mm")
    print(f"Wet Days (RR1): {indices['RR1']}")
    print(f"Mean Wind Speed: {indices['FG']:.2f} m/s")

    return df, percentiles, indices

if __name__ == "__main__":
    test_climatology_functions()