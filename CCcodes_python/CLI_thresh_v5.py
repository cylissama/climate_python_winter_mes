import numpy as np
import pandas as pd
from pathlib import Path
import scipy.io as sio
from scipy.signal import savgol_filter


def rlowess_smooth(data, window_size):
    """
    Apply robust lowess smoothing similar to MATLAB's smoothdata function.

    Args:
        data: 1D numpy array of values to smooth
        window_size: Size of the smoothing window

    Returns:
        Smoothed data array
    """
    # For a simple approximation, we'll use Savitzky-Golay filter
    # which performs local polynomial regression
    # The window length must be odd and greater than polyorder
    if window_size % 2 == 0:
        window_size += 1

    # The polynomial order (typically 1-3)
    polyorder = min(3, window_size - 1)

    # Apply filter and return smoothed data
    return savgol_filter(data, window_size, polyorder)


def calculate_climate_thresholds(site_csv='site.csv', output_dir=None):
    """
    Calculate climate thresholds for precipitation and temperature variables.

    Args:
        site_csv: Path to CSV file containing site information
        output_dir: Directory to save output files
    """
    # Define thresholds to calculate
    PRCP_thresh = np.array([20, 25, 33.3, 40, 50, 60, 66.6, 75, 80, 90, 95, 99, 99.9])
    TAIR_thresh = np.array([10, 25, 50, 75, 90])

    # Number of years in climatology
    nc = 30

    # Load site metadata
    non_temporal = pd.read_csv(site_csv)
    sites = non_temporal.iloc[:, 1].values.astype(str)
    nsM = len(sites)
    latM = non_temporal.iloc[:, 2].values
    lonM = non_temporal.iloc[:, 3].values
    elvM = non_temporal.iloc[:, 4].values * 0.3048  # Convert feet to meters

    # Directory setup
    if output_dir is None:
        output_dir = Path.cwd() / 'output'
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Process each site
    for i in range(nsM):
        site = sites[i]
        print(f"Processing site {site} ({i + 1}/{nsM})")

        # Initialize data arrays
        prcp_ib1 = np.zeros((nc, 365))
        tair_ib1 = np.zeros((nc, 365))
        tairx_ib1 = np.zeros((nc, 365))
        tairn_ib1 = np.zeros((nc, 365))

        # Load hourly data for 30 years (1980-2010)
        site_dir = Path(f'/Users/erappin/Documents/Mesonet/ClimateIndices/cliSITES/{site}')

        # Load first year to initialize
        file_path = site_dir / f'1980_{site}_hourly.mat'
        try:
            data = sio.loadmat(str(file_path))
            tt_hourly = data['TT_hourly']
        except:
            print(f"Warning: Could not load data for site {site}")
            continue

        # Load remaining years and concatenate
        for j in range(1, nc):
            year = 1980 + j
            file_path = site_dir / f'{year}_{site}_hourly.mat'
            try:
                data = sio.loadmat(str(file_path))
                # Assuming TT_hourly is a structured array with columns
                tt_hourly = np.vstack((tt_hourly, data['TT_hourly']))
            except:
                print(f"Warning: Could not load data for site {site}, year {year}")

        # Convert hourly to daily values
        # Assuming tt_hourly has a timestamp column and the required variables
        df_hourly = pd.DataFrame(tt_hourly)
        df_hourly['timestamp'] = pd.to_datetime(df_hourly.iloc[:, 0])
        df_hourly.set_index('timestamp', inplace=True)

        # Resample to daily values
        df_daily = df_hourly.resample('D').mean()
        df_daily_min = df_hourly.resample('D').min()
        df_daily_max = df_hourly.resample('D').max()
        df_daily_sum = df_hourly.resample('D').sum()

        # Extract variables (assuming column order matches MATLAB)
        df_daily['TAIR'] = df_daily.iloc[:, 0]
        df_daily['TAIRx'] = df_daily_max.iloc[:, 0]
        df_daily['TAIRn'] = df_daily_min.iloc[:, 0]
        df_daily['PRCP'] = df_daily_sum.iloc[:, 1]  # Assuming PRCP is column 1

        # Remove leap days
        df_daily = df_daily[~((df_daily.index.month == 2) & (df_daily.index.day == 29))]

        # Group by year
        grouped = df_daily.groupby(df_daily.index.year)
        years = list(grouped.groups.keys())

        # Calculate in-base and out-of-base datasets
        for j in range(nc):
            year_data = grouped.get_group(years[j])

            # Extract data for the current year (out-of-base)
            if len(year_data) == 365:
                prcp_ob = year_data['PRCP'].values
                tair_ob = year_data['TAIR'].values
                tairx_ob = year_data['TAIRx'].values
                tairn_ob = year_data['TAIRn'].values

            # For in-base data, replace with a random different year
            valid_years = [y for y in years if y != years[j]]
            replacement_year = np.random.choice(valid_years)
            replacement_data = grouped.get_group(replacement_year)

            if len(replacement_data) == 365:
                prcp_ib1[j] = replacement_data['PRCP'].values
                tair_ib1[j] = replacement_data['TAIR'].values
                tairx_ib1[j] = replacement_data['TAIRx'].values
                tairn_ib1[j] = replacement_data['TAIRn'].values

        # Apply smoothing
        prcp_ib5 = np.zeros_like(prcp_ib1)
        tair_ib5 = np.zeros_like(tair_ib1)
        tairx_ib5 = np.zeros_like(tairx_ib1)
        tairn_ib5 = np.zeros_like(tairn_ib1)

        prcp_ib25 = np.zeros_like(prcp_ib1)
        tair_ib25 = np.zeros_like(tair_ib1)
        tairx_ib25 = np.zeros_like(tairx_ib1)
        tairn_ib25 = np.zeros_like(tairn_ib1)

        for j in range(nc):
            prcp_ib5[j] = rlowess_smooth(prcp_ib1[j], 5)
            tair_ib5[j] = rlowess_smooth(tair_ib1[j], 5)
            tairx_ib5[j] = rlowess_smooth(tairx_ib1[j], 5)
            tairn_ib5[j] = rlowess_smooth(tairn_ib1[j], 5)

            prcp_ib25[j] = rlowess_smooth(prcp_ib1[j], 25)
            tair_ib25[j] = rlowess_smooth(tair_ib1[j], 25)
            tairx_ib25[j] = rlowess_smooth(tairx_ib1[j], 25)
            tairn_ib25[j] = rlowess_smooth(tairn_ib1[j], 25)

        # Calculate percentiles
        prcp_perc1 = np.zeros((nc, len(PRCP_thresh), 365))
        tair_perc1 = np.zeros((nc, len(TAIR_thresh), 365))
        tairx_perc1 = np.zeros((nc, len(TAIR_thresh), 365))
        tairn_perc1 = np.zeros((nc, len(TAIR_thresh), 365))

        prcp_perc5 = np.zeros((nc, len(PRCP_thresh), 365))
        tair_perc5 = np.zeros((nc, len(TAIR_thresh), 365))
        tairx_perc5 = np.zeros((nc, len(TAIR_thresh), 365))
        tairn_perc5 = np.zeros((nc, len(TAIR_thresh), 365))

        prcp_perc25 = np.zeros((nc, len(PRCP_thresh), 365))
        tair_perc25 = np.zeros((nc, len(TAIR_thresh), 365))
        tairx_perc25 = np.zeros((nc, len(TAIR_thresh), 365))
        tairn_perc25 = np.zeros((nc, len(TAIR_thresh), 365))

        for j in range(nc):
            for d in range(365):
                # Get data for this day across all years
                prcp_day = prcp_ib1[:, d]
                tair_day = tair_ib1[:, d]
                tairx_day = tairx_ib1[:, d]
                tairn_day = tairn_ib1[:, d]

                # Calculate percentiles
                prcp_perc1[j, :, d] = np.percentile(prcp_day, PRCP_thresh)
                tair_perc1[j, :, d] = np.percentile(tair_day, TAIR_thresh)
                tairx_perc1[j, :, d] = np.percentile(tairx_day, TAIR_thresh)
                tairn_perc1[j, :, d] = np.percentile(tairn_day, TAIR_thresh)

                # Smoothed data (5-day window)
                prcp_day = prcp_ib5[:, d]
                tair_day = tair_ib5[:, d]
                tairx_day = tairx_ib5[:, d]
                tairn_day = tairn_ib5[:, d]

                prcp_perc5[j, :, d] = np.percentile(prcp_day, PRCP_thresh)
                tair_perc5[j, :, d] = np.percentile(tair_day, TAIR_thresh)
                tairx_perc5[j, :, d] = np.percentile(tairx_day, TAIR_thresh)
                tairn_perc5[j, :, d] = np.percentile(tairn_day, TAIR_thresh)

                # Smoothed data (25-day window)
                prcp_day = prcp_ib25[:, d]
                tair_day = tair_ib25[:, d]
                tairx_day = tairx_ib25[:, d]
                tairn_day = tairn_ib25[:, d]

                prcp_perc25[j, :, d] = np.percentile(prcp_day, PRCP_thresh)
                tair_perc25[j, :, d] = np.percentile(tair_day, TAIR_thresh)
                tairx_perc25[j, :, d] = np.percentile(tairx_day, TAIR_thresh)
                tairn_perc25[j, :, d] = np.percentile(tairn_day, TAIR_thresh)

        # Calculate mean percentiles across years
        prcp_meanpercs1 = np.mean(prcp_perc1, axis=0)
        tair_meanpercs1 = np.mean(tair_perc1, axis=0)
        tairx_meanpercs1 = np.mean(tairx_perc1, axis=0)
        tairn_meanpercs1 = np.mean(tairn_perc1, axis=0)

        prcp_meanpercs5 = np.mean(prcp_perc5, axis=0)
        tair_meanpercs5 = np.mean(tair_perc5, axis=0)
        tairx_meanpercs5 = np.mean(tairx_perc5, axis=0)
        tairn_meanpercs5 = np.mean(tairn_perc5, axis=0)

        prcp_meanpercs25 = np.mean(prcp_perc25, axis=0)
        tair_meanpercs25 = np.mean(tair_perc25, axis=0)
        tairx_meanpercs25 = np.mean(tairx_perc25, axis=0)
        tairn_meanpercs25 = np.mean(tairn_perc25, axis=0)

        # Save results
        output_file = output_dir / f"{site}_CLIthresh_daily.npz"
        np.savez(
            output_file,
            sites=sites,
            nsM=nsM,
            latM=latM,
            lonM=lonM,
            elvM=elvM,
            PRCP_meanpercs1=prcp_meanpercs1,
            TAIR_meanpercs1=tair_meanpercs1,
            TAIRx_meanpercs1=tairx_meanpercs1,
            TAIRn_meanpercs1=tairn_meanpercs1,
            PRCP_meanpercs5=prcp_meanpercs5,
            TAIR_meanpercs5=tair_meanpercs5,
            TAIRx_meanpercs5=tairx_meanpercs5,
            TAIRn_meanpercs5=tairn_meanpercs5,
            PRCP_meanpercs25=prcp_meanpercs25,
            TAIR_meanpercs25=tair_meanpercs25,
            TAIRx_meanpercs25=tairx_meanpercs25,
            TAIRn_meanpercs25=tairn_meanpercs25,
            PRCP_thresh=PRCP_thresh,
            TAIR_thresh=TAIR_thresh,
        )

        print(f"Saved threshold data for site {site}")


if __name__ == "__main__":
    calculate_climate_thresholds()