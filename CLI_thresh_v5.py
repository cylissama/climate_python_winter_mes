import os
import numpy as np
import pandas as pd
import scipy.io
from statsmodels.nonparametric.smoothers_lowess import lowess

# Define a helper function to replicate rlowess smoothing
def rlowess_smooth(data_1d, window):
    """
    Replicates the 'smoothdata(..., "rlowess", window)' concept from MATLAB
    using statsmodels.nonparametric.smoothers_lowess.lowess.
    
    data_1d: 1D numpy array of daily values
    window: integer smoothing window (approx # of days)
    """
    if len(data_1d) == 0:
        return data_1d
    
    # Create an x-axis for the data
    x = np.arange(len(data_1d))
    
    # fraction determines how large the smoothing window is relative to data length
    # For a rough approximation: fraction ~ window / length_of_series
    frac = min(1.0, window / float(len(data_1d)))
    
    # LOWESS returns an Nx2 array of smoothed [x, yhat]
    smoothed = lowess(data_1d, x, frac=frac, return_sorted=True)
    
    # The second column of `smoothed` is the fitted values
    return smoothed[:, 1]

def main():
    # --- 1. Define thresholds ---
    PRCP_thresh = [20, 25, 33.3, 40, 50, 60, 66.6, 75, 80, 90, 95, 99, 99.9]
    TAIR_thresh = [10, 25, 50, 75, 90]
    
    nc = 30  # number of years in the climatology (e.g., 1980-2009)
    
    # --- 2. Load station data (site.csv) ---
    # 'site.csv' expected to have columns: e.g. [ID, SiteName, Lat, Lon, Elev(ft)]
    non_temporal = pd.read_csv('site.csv', na_values='NaN')
    # Adjust as needed if CSV structure differs
    # Example indexing: site name is col 2, lat is col 3, lon col 4, elev col 5
    sites = non_temporal.iloc[:, 1].astype(str).values  # second column
    nsM = len(sites)
    latM = non_temporal.iloc[:, 2].values
    lonM = non_temporal.iloc[:, 3].values
    elvM = non_temporal.iloc[:, 4].values * 0.3048  # convert ft to m
    
    # Preallocate arrays. We expect ~365 days x 30 yrs for each station
    # The final shape might be (nsM, nc, 365) if each year has 365 days (minus leap days).
    # We won't know the exact daily dimension up front, so let's store as list-of-arrays.
    PRCP_ib1  = np.full((nsM, nc, 365), np.nan)
    TAIR_ib1  = np.full((nsM, nc, 365), np.nan)
    TAIRx_ib1 = np.full((nsM, nc, 365), np.nan)
    TAIRn_ib1 = np.full((nsM, nc, 365), np.nan)
    
    # We'll also store the "out-of-base" data
    PRCP_ob  = np.full((nsM, 365), np.nan)
    TAIR_ob  = np.full((nsM, 365), np.nan)
    TAIRx_ob = np.full((nsM, 365), np.nan)
    TAIRn_ob = np.full((nsM, 365), np.nan)
    
    # Similarly for smoothed versions
    PRCP_ib5  = np.full((nsM, nc, 365), np.nan)
    TAIR_ib5  = np.full((nsM, nc, 365), np.nan)
    TAIRx_ib5 = np.full((nsM, nc, 365), np.nan)
    TAIRn_ib5 = np.full((nsM, nc, 365), np.nan)
    
    PRCP_ib25  = np.full((nsM, nc, 365), np.nan)
    TAIR_ib25  = np.full((nsM, nc, 365), np.nan)
    TAIRx_ib25 = np.full((nsM, nc, 365), np.nan)
    TAIRn_ib25 = np.full((nsM, nc, 365), np.nan)
    
    # Arrays to hold final daily percentile results
    # For each site, year, percentile, day. For PRCP_thresh, shape ~ (nsM, nc, len(PRCP_thresh), 365)
    PRCP_perc1  = np.full((nsM, nc, len(PRCP_thresh), 365), np.nan)
    TAIR_perc1  = np.full((nsM, nc, len(TAIR_thresh), 365), np.nan)
    TAIRx_perc1 = np.full((nsM, nc, len(TAIR_thresh), 365), np.nan)
    TAIRn_perc1 = np.full((nsM, nc, len(TAIR_thresh), 365), np.nan)
    
    PRCP_perc5  = np.full((nsM, nc, len(PRCP_thresh), 365), np.nan)
    TAIR_perc5  = np.full((nsM, nc, len(TAIR_thresh), 365), np.nan)
    TAIRx_perc5 = np.full((nsM, nc, len(TAIR_thresh), 365), np.nan)
    TAIRn_perc5 = np.full((nsM, nc, len(TAIR_thresh), 365), np.nan)
    
    PRCP_perc25  = np.full((nsM, nc, len(PRCP_thresh), 365), np.nan)
    TAIR_perc25  = np.full((nsM, nc, len(TAIR_thresh), 365), np.nan)
    TAIRx_perc25 = np.full((nsM, nc, len(TAIR_thresh), 365), np.nan)
    TAIRn_perc25 = np.full((nsM, nc, len(TAIR_thresh), 365), np.nan)
    
    #dirin1 = '/Users/erappin/Documents/Mesonet/ClimateIndices/cliSITES/'
    dirin1 = '/Volumes/Mesonet/cliSITES/'
    
    # Loop over each station
    for i in range(nsM):
        station_id = sites[i]
        dirin = os.path.join(dirin1, station_id)
        
        # Initialize a container for all hourly data
        TT_hourlyF = None
        
        # Loop over 30 years: 1980..1980+nc-1
        for j in range(nc):
            year_str = str(1980 + j)
            filein_year = os.path.join(dirin, f"{year_str}_{station_id}_hourly.mat")

            # Try loading the MAT file
            try:
                matdata = scipy.io.loadmat(filein_year)
                for key in matdata.keys():
                    print(key)

            except FileNotFoundError:
                #print(f"Warning: File not found: {filein_year}. Skipping this year.")
                continue
            
            if "TT_hourly" not in matdata:
                print(f"Warning: 'TT_hourly' variable missing in {filein_year}. Skipping.")
                continue
            
            TT_hourly_j = matdata['TT_hourly']  # NxM array or structured data
            if TT_hourlyF is None:
                TT_hourlyF = TT_hourly_j
            else:
                TT_hourlyF = np.concatenate([TT_hourlyF, TT_hourly_j], axis=0)
        
        # If no data was loaded at all (all files missing?), skip this station
        if TT_hourlyF is None or TT_hourlyF.size == 0:
            print(f"No data loaded for station {station_id}. Skipping station.")
            continue
        
        # At this point, TT_hourlyF is a concatenation of hourly data for station i (all 30 years).
        # Next steps: convert to a pandas DataFrame with a DateTime index so we can resample.
        # Suppose TT_hourly has columns: TIMESTAMP, TAIR, PRCP, PRES, etc.
        # Let’s assume TT_hourlyF is Nx4 or Nx5 and has a first column for timestamps.
        
        # In reality, you'd need to map the actual structure. For demonstration:
        df_hourly = pd.DataFrame(TT_hourlyF, columns=['TIMESTAMP','TAIR','PRCP','PRES'])
        # Convert TIMESTAMP from numeric to datetime if needed:
        # df_hourly['TIMESTAMP'] = pd.to_datetime(df_hourly['TIMESTAMP'], unit='s') or some format
        
        # Set DateTimeIndex
        df_hourly.set_index('TIMESTAMP', inplace=True)
        
        # --- 4. Convert hourly -> daily mean/min/max/sum ---
        df_daily_mean = df_hourly.resample('D').mean()
        df_daily_max  = df_hourly.resample('D').max()
        df_daily_min  = df_hourly.resample('D').min()
        df_daily_sum  = df_hourly.resample('D').sum()  # for precipitation
        
        # Remove the old PRCP from df_daily_mean, then add custom columns
        if 'PRCP' in df_daily_mean.columns:
            df_daily_mean.drop(columns=['PRCP'], inplace=True)
        
        df_daily_mean['TAIRx'] = df_daily_max['TAIR']
        df_daily_mean['TAIRn'] = df_daily_min['TAIR']
        df_daily_mean['PRCP']  = df_daily_sum['PRCP']
        
        # --- 5. Exclude the "current year" (like the MATLAB code that excludes YEAR) ---
        # In MATLAB, they do: TT_daily(y==YEAR,:) = []
        # But the code actually loops per-year and excludes that year from the base dataset.
        # If we strictly replicate that logic, we need to do the exclude for each year j in a loop.
        
        # For simplicity, let's gather the daily data once. Then we do the in-base logic below.
        # Also remove leap days (Feb 29):
        df_daily_mean = df_daily_mean[~((df_daily_mean.index.month == 2) & (df_daily_mean.index.day == 29))]
        
        # We now have a DataFrame of daily data for all 30 years (minus leap days).
        # The final shape should be ~ 30*365 = 10950 rows (if no data is missing).
        
        # The MATLAB code indexes data by year and day. We can reorganize by year-day in Python.
        # But MATLAB is effectively doing: for j in 1:30, exclude that jth year. Then store data in PRCP_ib1(i,j,:).
        
        # Let’s reorganize daily data by year:
        df_daily_mean['YEAR'] = df_daily_mean.index.year
        # For each year 1980...1980+nc-1, we collect the daily data except that year. 
        # Then store in PRCP_ib1. 
        # This is the "leave-one-year-out" approach.
        
        # We assume the full base period is 1980..1980+nc-1 = 1980..2009
        years = range(1980, 1980+nc)
        
        # We'll convert daily data into an array for easier indexing:
        # For each year, we keep only that year's daily data (365 days after leap-day removal).
        for j_idx, year_j in enumerate(years):
            df_that_year = df_daily_mean[df_daily_mean['YEAR'] == year_j]
            
            # Exclude that year from 'in-base' data
            df_inbase = df_daily_mean[df_daily_mean['YEAR'] != year_j]
            
            # If the code is storing daily data for that year as 'out-of-base'
            # we take the original daily data for that year:
            # But in the MATLAB code, they do a random year replacement for the in-base dataset.
            
            # Store out-of-base (ob) data:
            if len(df_that_year) == 365:
                PRCP_ob[i, :]  = df_that_year['PRCP'].values
                TAIR_ob[i, :]  = df_that_year['TAIR'].values
                TAIRx_ob[i, :] = df_that_year['TAIRx'].values
                TAIRn_ob[i, :] = df_that_year['TAIRn'].values
            
            # Now, pick a random valid year (not year_j) for the in-base data
            valid_vals = [yr for yr in years if yr != year_j]
            result_year = np.random.choice(valid_vals)
            df_result_year = df_daily_mean[df_daily_mean['YEAR'] == result_year]
            
            if len(df_result_year) == 365:
                PRCP_ib1[i, j_idx, :]  = df_result_year['PRCP'].values
                TAIR_ib1[i, j_idx, :]  = df_result_year['TAIR'].values
                TAIRx_ib1[i, j_idx, :] = df_result_year['TAIRx'].values
                TAIRn_ib1[i, j_idx, :] = df_result_year['TAIRn'].values
            
            # --- 6. Smooth (rlowess) with windows of 5 and 25 ---
            # replicate "smoothdata(..., 'rlowess', window)"
            if len(df_result_year) == 365:
                PRCP_ib5[i, j_idx, :]  = rlowess_smooth(PRCP_ib1[i, j_idx, :], 5)
                TAIR_ib5[i, j_idx, :]  = rlowess_smooth(TAIR_ib1[i, j_idx, :], 5)
                TAIRx_ib5[i, j_idx, :] = rlowess_smooth(TAIRx_ib1[i, j_idx, :], 5)
                TAIRn_ib5[i, j_idx, :] = rlowess_smooth(TAIRn_ib1[i, j_idx, :], 5)
                
                PRCP_ib25[i, j_idx, :]  = rlowess_smooth(PRCP_ib1[i, j_idx, :], 25)
                TAIR_ib25[i, j_idx, :]  = rlowess_smooth(TAIR_ib1[i, j_idx, :], 25)
                TAIRx_ib25[i, j_idx, :] = rlowess_smooth(TAIRx_ib1[i, j_idx, :], 25)
                TAIRn_ib25[i, j_idx, :] = rlowess_smooth(TAIRn_ib1[i, j_idx, :], 25)
        
        # --- 7. Calculate percentiles for each day across the 30 in-base sets ---
        # In MATLAB: prctile(PRCP_ib1(i,:,:), PRCP_thresh, 2)
        # Means: across the year dimension, compute percentile. 
        # Shape (nc, 365) -> percentile result shape: (len(PRCP_thresh), 365)
        
        for j_idx in range(nc):
            # Non-smoothed:
            daily_data_prcp = PRCP_ib1[i, :, :]  # shape (nc, 365)
            daily_data_tair = TAIR_ib1[i, :, :]
            daily_data_tx   = TAIRx_ib1[i, :, :]
            daily_data_tn   = TAIRn_ib1[i, :, :]
            
            # We want the percentile for each day across the dimension=0 (the year dimension).
            # So for day d in [0..364], PRCP_perc1(i,j_idx,:,d) = percentiles across the nc array?
            # But the code also has a loop for j in 1:nc, then calls prctile(...).
            
            PRCP_perc1[i, j_idx, :, :]  = np.percentile(daily_data_prcp, PRCP_thresh, axis=0)
            TAIR_perc1[i, j_idx, :, :]  = np.percentile(daily_data_tair, TAIR_thresh, axis=0)
            TAIRx_perc1[i, j_idx, :, :] = np.percentile(daily_data_tx,   TAIR_thresh, axis=0)
            TAIRn_perc1[i, j_idx, :, :] = np.percentile(daily_data_tn,   TAIR_thresh, axis=0)
            
            # Smoothed, 5-day window:
            daily_data_prcp_5 = PRCP_ib5[i, :, :]
            daily_data_tair_5 = TAIR_ib5[i, :, :]
            daily_data_tx_5   = TAIRx_ib5[i, :, :]
            daily_data_tn_5   = TAIRn_ib5[i, :, :]
            
            PRCP_perc5[i, j_idx, :, :]  = np.percentile(daily_data_prcp_5, PRCP_thresh, axis=0)
            TAIR_perc5[i, j_idx, :, :]  = np.percentile(daily_data_tair_5, TAIR_thresh, axis=0)
            TAIRx_perc5[i, j_idx, :, :] = np.percentile(daily_data_tx_5,   TAIR_thresh, axis=0)
            TAIRn_perc5[i, j_idx, :, :] = np.percentile(daily_data_tn_5,   TAIR_thresh, axis=0)
            
            # Smoothed, 25-day window:
            daily_data_prcp_25 = PRCP_ib25[i, :, :]
            daily_data_tair_25 = TAIR_ib25[i, :, :]
            daily_data_tx_25   = TAIRx_ib25[i, :, :]
            daily_data_tn_25   = TAIRn_ib25[i, :, :]
            
            PRCP_perc25[i, j_idx, :, :]  = np.percentile(daily_data_prcp_25, PRCP_thresh, axis=0)
            TAIR_perc25[i, j_idx, :, :]  = np.percentile(daily_data_tair_25, TAIR_thresh, axis=0)
            TAIRx_perc25[i, j_idx, :, :] = np.percentile(daily_data_tx_25,   TAIR_thresh, axis=0)
            TAIRn_perc25[i, j_idx, :, :] = np.percentile(daily_data_tn_25,   TAIR_thresh, axis=0)
        
        # --- 8. Compute the mean percentile across all 30 base years ---
        # The MATLAB code does something like: mean(PRCP_perc1(i,:,:,:),2)
        # This means take the mean over dimension=2 (the 'year' dimension).
        # Our dimension ordering is (nsM, nc, num_thresh, 365). The 'year' dimension is index=1.
        PRCP_meanpercs1  = np.mean(PRCP_perc1[i, :, :, :], axis=0)  # shape (num_thresh, 365)
        TAIR_meanpercs1  = np.mean(TAIR_perc1[i, :, :, :], axis=0)
        TAIRx_meanpercs1 = np.mean(TAIRx_perc1[i, :, :, :], axis=0)
        TAIRn_meanpercs1 = np.mean(TAIRn_perc1[i, :, :, :], axis=0)
        
        PRCP_meanpercs5  = np.mean(PRCP_perc5[i, :, :, :], axis=0)
        TAIR_meanpercs5  = np.mean(TAIR_perc5[i, :, :, :], axis=0)
        TAIRx_meanpercs5 = np.mean(TAIRx_perc5[i, :, :, :], axis=0)
        TAIRn_meanpercs5 = np.mean(TAIRn_perc5[i, :, :, :], axis=0)
        
        PRCP_meanpercs25  = np.mean(PRCP_perc25[i, :, :, :], axis=0)
        TAIR_meanpercs25  = np.mean(TAIR_perc25[i, :, :, :], axis=0)
        TAIRx_meanpercs25 = np.mean(TAIRx_perc25[i, :, :, :], axis=0)
        TAIRn_meanpercs25 = np.mean(TAIRn_perc25[i, :, :, :], axis=0)
        
        # --- 9. Save the results to a .mat file (or other format) ---
        dirout = os.path.join(dirin1, station_id)
        fileout = os.path.join(dirout, f"{station_id}_CLIthresh_daily.mat")
        
        mdict = {
            'sites': sites,
            'nsM': nsM,
            'latM': latM,
            'lonM': lonM,
            'elvM': elvM,
            'PRCP_meanpercs1': PRCP_meanpercs1,
            'TAIR_meanpercs1': TAIR_meanpercs1,
            'TAIRx_meanpercs1': TAIRx_meanpercs1,
            'TAIRn_meanpercs1': TAIRn_meanpercs1,
            
            'PRCP_meanpercs5': PRCP_meanpercs5,
            'TAIR_meanpercs5': TAIR_meanpercs5,
            'TAIRx_meanpercs5': TAIRx_meanpercs5,
            'TAIRn_meanpercs5': TAIRn_meanpercs5,
            
            'PRCP_meanpercs25': PRCP_meanpercs25,
            'TAIR_meanpercs25': TAIR_meanpercs25,
            'TAIRx_meanpercs25': TAIRx_meanpercs25,
            'TAIRn_meanpercs25': TAIRn_meanpercs25,
            
            'PRCP_thresh': PRCP_thresh,
            'TAIR_thresh': TAIR_thresh,
            
            # Optionally save some sample data frames or arrays
            'TT_hourlyF': TT_hourlyF,  # This will just be a NumPy array
            # 'TT_daily': df_daily_mean.to_numpy(),  # or store as needed
        }
        
        savemat(fileout, mdict)
        print(f"Saved results to {fileout}")

if __name__ == '__main__':
    main()