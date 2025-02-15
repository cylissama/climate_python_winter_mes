import numpy as np
import pandas as pd
from pathlib import Path

def calculate_climate_indices(site='HCKM'):
    # ======================
    # 1. Data Loading Setup
    # ======================
    base_dir = Path('/Users/erappin/Documents/Mesonet/ClimateIndices/sitesTEST_CCindices/')
    data_dir = base_dir / site
    
    # Load threshold data (assuming .npy or .csv format)
    thresh_path = data_dir / f"{site}_CLIthresh_daily.npy"
    clim_thresh = np.load(thresh_path, allow_pickle=True).item()
    
    # Load monthly data (assuming pickle format)
    input_path = data_dir / f"{site}_DATAinput_monthly.pkl"
    data_month = pd.read_pickle(input_path)

    # ======================
    # 2. Data Preparation
    # ======================
    ny = len(data_month['year'])    # Number of years
    nm = len(data_month['month'])   # Number of months
    nv = len(data_month['var'])     # Number of variables
    
    # Initialize data structure
    data_all = [[[] for _ in range(ny)] for _ in range(nm)]
    
    # Fill data structure with DataFrames
    for i in range(nm):
        for j in range(ny):
            if not data_month['data'][i][j]:
                # Handle missing data by copying from adjacent year
                if j < ny-1:
                    data_all[i][j] = data_month['data'][i][j+1].copy()
                else:
                    data_all[i][j] = data_month['data'][i][j-1].copy()
                # Fill with NaN values
                data_all[i][j].iloc[:] = np.nan
                # Fix year in timestamps
                data_all[i][j]['TimestampCollected'] = data_all[i][j]['TimestampCollected'].apply(
                    lambda x: x.replace(year=data_month['year'][j]))
            else:
                data_all[i][j] = data_month['data'][i][j]

    # ======================
    # 3. Climate Indices Calculation
    # ======================
    
    # Initialize result matrices
    results = {
        'FD': np.zeros((ny, nm)),     # Frost Days
        'HDD': np.zeros((ny, nm)),    # Heating Degree Days
        'GD4': np.zeros((ny, nm)),    # Growing Degree Days (4°C base)
        'GD10': np.zeros((ny, nm)),   # Growing Degree Days (10°C base)
        'TXx': np.zeros((ny, nm)),    # Max Tmax
        'TNn': np.zeros((ny, nm)),    # Min Tmin
    }

    for i in range(nm):  # Month loop
        for j in range(ny):  # Year loop
            df = data_all[i][j]
            
            # Frost Days (TMIN < 0°C)
            results['FD'][j, i] = (df['TMIN'] < 0).sum()
            
            # Heating Degree Days (18.3°C - TAVG)
            hdd = 18.3 - df['TAVG']
            results['HDD'][j, i] = hdd[hdd > 0].sum()
            
            # Growing Degree Days
            results['GD4'][j, i] = (df['TAVG'] - 4).clip(lower=0).sum()
            results['GD10'][j, i] = (df['TAVG'] - 10).clip(lower=0).sum()
            
            # Extreme temperatures
            results['TXx'][j, i] = df['TMAX'].max()
            results['TNn'][j, i] = df['TMIN'].min()

    # ======================
    # 4. Post-processing
    # ======================
    # Convert results to DataFrame with multi-index
    years = data_month['year']
    months = data_month['month']
    index = pd.MultiIndex.from_product([years, months], names=['Year', 'Month'])
    
    result_df = pd.DataFrame(
        {k: v.ravel() for k, v in results.items()},
        index=index
    )

    return result_df

# Execute the function
if __name__ == '__main__':
    climate_indices = calculate_climate_indices()
    print(climate_indices.head())