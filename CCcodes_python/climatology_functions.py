import numpy as np
import pandas as pd
from datetime import datetime
import scipy.stats as stats


def calculate_percentiles(data, base_period_start=1961, base_period_end=1990):
    """
    Calculate percentiles according to the method described in ECA&D document.

    Parameters:
    -----------
    data : pandas DataFrame
        DataFrame containing the climate data
    base_period_start : int
        Start year of the base period
    base_period_end : int
        End year of the base period

    Returns:
    --------
    percentiles : dict
        Dictionary containing the 10th, 90th, 95th, 99th percentiles for each calendar day
    """
    # Filter data for base period
    if 'date' not in data.columns:
        data['date'] = pd.to_datetime(data.index)

    base_period = data[(data['date'].dt.year >= base_period_start) &
                       (data['date'].dt.year <= base_period_end)]

    # Create day-of-year identifier (ignoring year)
    base_period['day_of_year'] = base_period['date'].dt.strftime('%m-%d')

    # Initialize percentile dictionaries
    percentiles = {
        '10th': {},
        '90th': {},
        '95th': {},
        '99th': {}
    }

    # Get unique days of year
    unique_days = np.unique(base_period['day_of_year'])

    for day in unique_days:
        # For each calendar day, extract 5-day window
        month, day_num = map(int, day.split('-'))

        # Create 5-day window masks centered on the calendar day
        window_masks = []
        for offset in range(-2, 3):
            # Calculate the date with offset
            # Note: This is simplified and doesn't handle month/year boundaries correctly
            target_day = pd.Timestamp(2000, month, day_num) + pd.Timedelta(days=offset)
            day_mask = base_period['day_of_year'] == target_day.strftime('%m-%d')
            window_masks.append(day_mask)

        # Combine masks with OR
        window_mask = np.logical_or.reduce(window_masks)
        window_data = base_period[window_mask]

        # Following the method from Zhang et al. (2005) as described in the document
        # For each year in the base period, calculate percentiles excluding that year
        for var in data.columns:
            if var in ['date', 'day_of_year']:
                continue

            years = np.unique(window_data['date'].dt.year)
            percentile_estimates = []

            for year in years:
                # Construct 30-yr block by excluding the target year and replicating another year
                year_mask = window_data['date'].dt.year == year
                out_of_base_data = window_data[year_mask][var].values

                # Exclude current year and replicate another year from the base period
                in_base_data = window_data[~year_mask]

                if len(in_base_data) < 29 * 5:  # Need enough data for 29 years with 5 days each
                    continue

                # Replicate data from first year that's not the target year
                replicate_year = [y for y in years if y != year][0]
                replicate_mask = window_data['date'].dt.year == replicate_year
                replicated_data = window_data[replicate_mask][var].values

                # Combine data for 30-yr block
                combined_data = np.concatenate([in_base_data[var].values, replicated_data])

                # Calculate percentiles
                p10 = np.percentile(combined_data, 10)
                p90 = np.percentile(combined_data, 90)
                p95 = np.percentile(combined_data, 95)
                p99 = np.percentile(combined_data, 99)

                # Count exceedances for this year
                p10_exceed = np.sum(out_of_base_data < p10) / len(out_of_base_data)
                p90_exceed = np.sum(out_of_base_data > p90) / len(out_of_base_data)
                p95_exceed = np.sum(out_of_base_data > p95) / len(out_of_base_data)
                p99_exceed = np.sum(out_of_base_data > p99) / len(out_of_base_data)

                percentile_estimates.append({
                    '10th': p10_exceed,
                    '90th': p90_exceed,
                    '95th': p95_exceed,
                    '99th': p99_exceed
                })

            # Average the estimates
            if percentile_estimates:
                percentiles['10th'][(day, var)] = np.mean([est['10th'] for est in percentile_estimates])
                percentiles['90th'][(day, var)] = np.mean([est['90th'] for est in percentile_estimates])
                percentiles['95th'][(day, var)] = np.mean([est['95th'] for est in percentile_estimates])
                percentiles['99th'][(day, var)] = np.mean([est['99th'] for est in percentile_estimates])

    return percentiles


def lowess_smoother(x, y, f=None, n_iter=3):
    """
    Implement LOWESS (locally weighted scatterplot smoothing) as described in ECA&D.

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
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if f is None:
        f = min(1.0, 30.0 / len(x))

    n = len(x)
    r = int(np.ceil(f * n))

    # Initial fit
    y_smooth = np.zeros_like(y)
    for i in range(n):
        weights = np.zeros_like(x, dtype=np.float64)

        # Calculate distances
        distances = np.abs(x - x[i])

        # Sort distances and get r nearest points
        sorted_idx = np.argsort(distances)
        neighbor_idx = sorted_idx[:r]

        # Max distance among included points
        max_dist = distances[sorted_idx[r - 1]]

        # Compute weights using tricube function
        for j in neighbor_idx:
            dist_ratio = distances[j] / max_dist
            weights[j] = (1 - dist_ratio ** 3) ** 3 if dist_ratio < 1 else 0

        # Weighted linear regression
        sum_weight = np.sum(weights)
        if sum_weight > 0:
            weighted_x = np.sum(weights * x) / sum_weight
            weighted_y = np.sum(weights * y) / sum_weight
            weighted_xy = np.sum(weights * x * y) / sum_weight
            weighted_xx = np.sum(weights * x * x) / sum_weight

            b1 = (weighted_xy - weighted_x * weighted_y) / (weighted_xx - weighted_x ** 2)
            b0 = weighted_y - b1 * weighted_x

            y_smooth[i] = b0 + b1 * x[i]
        else:
            y_smooth[i] = y[i]

    # Robustifying iterations
    for iteration in range(n_iter):
        residuals = y - y_smooth
        s = np.median(np.abs(residuals))

        robust_weights = np.zeros_like(residuals)
        for i in range(n):
            # Bisquare weight function
            u = residuals[i] / (6.0 * s)
            if np.abs(u) < 1:
                robust_weights[i] = (1 - u ** 2) ** 2
            else:
                robust_weights[i] = 0

        # Refit with robust weights
        for i in range(n):
            weights = np.zeros_like(x, dtype=np.float64)

            # Calculate distances
            distances = np.abs(x - x[i])

            # Sort distances and get r nearest points
            sorted_idx = np.argsort(distances)
            neighbor_idx = sorted_idx[:r]

            # Max distance among included points
            max_dist = distances[sorted_idx[r - 1]]

            # Compute weights using tricube function and robust weights
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

                if (weighted_xx - weighted_x ** 2) != 0:
                    b1 = (weighted_xy - weighted_x * weighted_y) / (weighted_xx - weighted_x ** 2)
                    b0 = weighted_y - b1 * weighted_x

                    y_smooth[i] = b0 + b1 * x[i]
                else:
                    y_smooth[i] = weighted_y
            else:
                y_smooth[i] = y[i]

    return y_smooth


def calculate_trend(data, auto_correlation=True):
    """
    Calculate trend value using least squares and adjust significance for autocorrelation.

    Parameters:
    -----------
    data : pandas DataFrame
        DataFrame with time as index and values to analyze
    auto_correlation : bool
        Whether to adjust for autocorrelation in the significance calculation

    Returns:
    --------
    result : dict
        Dictionary containing trend value, significance, and other statistics
    """
    # Extract x (time) and y (values)
    if isinstance(data.index, pd.DatetimeIndex):
        x = np.arange(len(data))
        years = data.index.year.values
    else:
        x = np.arange(len(data))
        years = x

    y = data.values

    # Filter out NaN values
    mask = ~np.isnan(y)
    x_filtered = x[mask]
    y_filtered = y[mask]

    if len(x_filtered) < 2:
        return {
            'trend': np.nan,
            'significance': np.nan,
            'p_value': np.nan,
            'equivalent_sample_size': np.nan
        }

    # Calculate linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_filtered, y_filtered)

    # Calculate residuals
    y_fit = intercept + slope * x_filtered
    residuals = y_filtered - y_fit

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

        sum_term = 0
        for k in range(1, min(n - 1, 20)):  # Use up to 20 lags
            weight = 1 - k / n
            sum_term += 2 * weight * acf[k]

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
        'years': years[mask]
    }


def calculate_climate_indices(data, temp_var='TAIR', precip_var='PRCP', wind_var='WSPD'):
    """
    Calculate climate indices as described in ECA&D document.

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

    Returns:
    --------
    indices : dict
        Dictionary containing calculated climate indices
    """
    indices = {}

    # Temperature indices
    if temp_var in data.columns:
        # GSL - Growing season length (days)
        if f'{temp_var}_season' in data.columns:
            temp = data[f'{temp_var}_season']
        else:
            temp = data[temp_var]

        # Find periods with at least 6 consecutive days > 5°C
        consec_days = 0
        growing_start = None

        for date, value in temp.items():
            if pd.isna(value):
                consec_days = 0
                continue

            if value > 5:
                consec