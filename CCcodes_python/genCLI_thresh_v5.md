# Climate Threshold Calculator

This script calculates climate threshold values based on historical meteorological data for multiple sites. It processes 30 years of hourly data, converts it to daily values, and computes percentile thresholds for various climate variables.

## Overview

The script performs the following steps:
1. Loads site metadata from a CSV file
2. For each site, loads 30 years of hourly meteorological data
3. Converts hourly data to daily values (mean, min, max, sum as appropriate)
4. Handles leap days by removing February 29
5. Creates "in-base" and "out-of-base" datasets for cross-validation
6. Applies smoothing with two window sizes (5-day and 25-day)
7. Calculates percentiles for precipitation and temperature variables
8. Averages the percentiles across all years
9. Saves the results for each site

## Input Requirements

- `site.csv`: A CSV file containing site information with columns for site ID, abbreviation, latitude, longitude, and elevation
- Hourly meteorological data files in MATLAB .mat format, organized by site and year

## Output

For each site, the script generates a `.npz` file containing:
- Site metadata (location, elevation)
- Calculated percentiles for precipitation and temperature variables
- Both raw and smoothed percentiles (using 5-day and 25-day windows)

## Key Functions

### rlowess_smooth(data, window_size)
Applies robust local regression smoothing to a time series, similar to MATLAB's smoothdata function with the 'rlowess' method.

### calculate_climate_thresholds(site_csv, output_dir)
Main function that processes site data and calculates climate thresholds.

## Percentile Calculation

The script calculates the following percentiles:
- Precipitation: 20, 25, 33.3, 40, 50, 60, 66.6, 75, 80, 90, 95, 99, 99.9
- Temperature: 10, 25, 50, 75, 90

## Cross-Validation Approach

To ensure robust thresholds, the script uses a leave-one-year-out approach:
1. For each year in the 30-year period, the data from that year is excluded
2. The remaining 29 years' data is used to calculate percentiles
3. A random year's data is used as a replacement for the excluded year
4. This process is repeated for all 30 years
5. The final thresholds are averaged across all iterations

## Dependencies

- numpy
- pandas
- scipy
- pathlib