"""
driverCCI_annual.py

This script processes climate data from Mesonet and calculates annual climate indices.
It performs the following steps:
1. Loads daily climate data from .mat files.
2. Converts the .mat files to .csv files using a MATLAB function.
3. Loads the .csv files into pandas DataFrames.
4. Processes the data to extract annual values for various climate variables.
5. Saves the results to a new .mat file.

The script ensures that the operations and logic are consistent with the original MATLAB code.

Cy Dixon
Created: 2025-01-06
"""

import os
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from datetime import datetime
from statsmodels.nonparametric.smoothers_lowess import lowess
import matlab.engine

def load_data(file_path):
    data = loadmat(file_path)
    return data

def get_data_annual(TT, iy, TIME):
    TIME_annual = TIME[iy]

    TAIR_A = TT['TAIR']
    TAIR_annual = TAIR_A[iy]
    TT_TAIR_an = pd.DataFrame({'TIME': TIME_annual, 'TAIR': TAIR_annual})

    DWPT_A = TT['DWPT']
    DWPT_annual = DWPT_A[iy]
    TT_DWPT_an = pd.DataFrame({'TIME': TIME_annual, 'DWPT': DWPT_annual})

    TAIRx_A = TT['TAIRx']
    TAIRx_annual = TAIRx_A[iy]
    TTx_TAIR_an = pd.DataFrame({'TIME': TIME_annual, 'TAIRx': TAIRx_annual})

    TAIRn_A = TT['TAIRn']
    TAIRn_annual = TAIRn_A[iy]
    TTn_TAIR_an = pd.DataFrame({'TIME': TIME_annual, 'TAIRn': TAIRn_annual})

    PRCP_A = TT['PRCP']
    PRCP_annual = PRCP_A[iy]
    TT_PRCP_an = pd.DataFrame({'TIME': TIME_annual, 'PRCP': PRCP_annual})

    RELH_A = TT['RELH']
    RELH_annual = RELH_A[iy]
    TT_RELH_an = pd.DataFrame({'TIME': TIME_annual, 'RELH': RELH_annual})

    PRES_A = TT['PRES']
    PRES_annual = PRES_A[iy]
    TT_PRES_an = pd.DataFrame({'TIME': TIME_annual, 'PRES': PRES_annual})

    SM02_A = TT['SM02']
    SM02_annual = SM02_A[iy]
    TT_SM02_an = pd.DataFrame({'TIME': TIME_annual, 'SM02': SM02_annual})

    WDIR_A = TT['WDIR']
    WDIR_annual = WDIR_A[iy]
    TT_WDIR_an = pd.DataFrame({'TIME': TIME_annual, 'WDIR': WDIR_annual})

    WSPD_A = TT['WSPD']
    WSPD_annual = WSPD_A[iy]
    TT_WSPD_an = pd.DataFrame({'TIME': TIME_annual, 'WSPD': WSPD_annual})

    WSMX_A = TT['WSMX']
    WSMX_annual = WSMX_A[iy]
    TT_WSMX_an = pd.DataFrame({'TIME': TIME_annual, 'WSMX': WSMX_annual})

    SRAD_A = TT['SRAD']
    SRAD_annual = SRAD_A[iy]
    TT_SRAD_an = pd.DataFrame({'TIME': TIME_annual, 'SRAD': SRAD_annual})

    TT_annual = pd.concat([TT_TAIR_an, TT_DWPT_an, TTx_TAIR_an, TTn_TAIR_an,
                           TT_PRCP_an, TT_RELH_an, TT_PRES_an, TT_SM02_an,
                           TT_WDIR_an, TT_WSPD_an, TT_WSMX_an, TT_SRAD_an], axis=1)
    return TT_annual

def main():
    site = 'BMTN'
    # Load Mesonet, Climate Data and Thresholds

    # set paths to data files
    dataDir = '/Volumes/Mesonet/winter_break/CCdata/'
    dailyFilePath = 'BMTN/01-Mar-2014_01-Aug-2023_BMTN_daily.mat'
    dataSiteDir = os.path.join(dataDir, site)
    filethresh = os.path.join(dataSiteDir, f"{site}_CLIthresh_daily.mat")
    filemes = os.path.join(dataDir, dailyFilePath)
    
    # matfile = loadmat(filethresh)
    # count = 0
    # print("\nVariables in threshold file:")
    # print("------------------------")
    # for key in matfile.keys():
    #     if not key.startswith('__'):  # Skip MATLAB metadata keys
    #         count += 1
    #         print(f"Variable: {key}")
    #         print(f"Type: {type(matfile[key])}")
    #         print(f"Size: {matfile[key].shape}")
    #         print("------------------------")
    # print(f"\nTotal variables: {count}")

    # Start MATLAB engine
    eng = matlab.engine.start_matlab()

    # Convert .mat file to .csv using MATLAB function
    # We need to get both the daily data from each file
    eng.addpath(r'/Volumes/Mesonet/winter_break/utils', nargout=0)
    # thresh_csv = eng.convert_mat_to_csv(filemes, nargout=1)
    thresh_csv = eng.convert_mat_to_csv(filethresh, "daily")
    mes_csv = eng.convert_mat_to_csv(filemes, "dailyMES")

    if not thresh_csv:
        print(f"Failed to convert to CSV.")
        return

    # Load both CSV files into pandas DataFrames
    try:
        thresh_df = pd.read_csv(thresh_csv)
        mes_df = pd.read_csv(mes_csv)
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return

    # Merge DataFrames on TIMESTAMP column
    combined_df = pd.merge(thresh_df, mes_df, 
                          on='TIMESTAMP', 
                          how='outer',
                          suffixes=('_thresh', '_mes'))
    
    # Sort by timestamp
    combined_df = combined_df.sort_values('TIMESTAMP')
    
    # Convert TIMESTAMP to datetime
    combined_df['TIMESTAMP'] = pd.to_datetime(combined_df['TIMESTAMP'])
    
    # Use combined DataFrame for further processing
    TT_dailyMES = combined_df
    TIME_full = TT_dailyMES['TIMESTAMP']
    #TIME_full = pd.to_datetime(TT_dailyMES['TIMESTAMP'])
    sTIME_full = TIME_full.iloc[0]
    eTIME_full = TIME_full.iloc[-1]
    YEAR_full = np.unique(TIME_full.dt.year)
    sYf, sMf, sDf = sTIME_full.year, sTIME_full.month, sTIME_full.day
    eYf, eMf, eDf = eTIME_full.year, eTIME_full.month, eTIME_full.day

    dateS = datetime(sYf + 1, 1, 1)
    if dateS >= sTIME_full:
        isD = TIME_full[TIME_full == dateS].index[0]
    dateE = datetime(eYf - 1, 12, 31)
    if dateE <= eTIME_full:
        ieD = TIME_full[TIME_full == dateE].index[0]

    TT = TT_dailyMES.iloc[isD:ieD]
    TIME = pd.to_datetime(TT['TIMESTAMP'])

    VAR = ["TAIR_month", "DWPT_month", "TAIRx_month", "TAIRn_month", "PRCP_month", "RELH_month",
           "PRES_month", "SM02_month", "WDIR_month", "WSPD_month", "WSMX_month", "SRAD_month"]
    YEAR = np.unique(TIME.dt.year)

    myStructA = {'var': VAR, 'year': YEAR}

    ny = len(YEAR)
    nv = len(VAR)

    Md = [[None for _ in range(nv)] for _ in range(ny)]
    My = [[None for _ in range(nv)] for _ in range(ny)]

    yoD = TIME.dt.year
    moD = TIME.dt.month

    for i in range(ny):
        k = 2009 + i
        iy = (yoD == k)
        TIME_annual = TIME[iy]
        for j in range(nv):
            TTnew = get_data_annual(TT, iy, TIME)
            TTcal = TIME
            TTyears = TTcal.dt.year
            TTindYears = np.unique(TTyears, return_index=True)[1]
            Vartt = TTnew.iloc[:, j]
            TTy = [TTnew.iloc[TTyears == y, :] for y in TTindYears]
            Ad = TTy
            Ay = TTindYears
            Md[i][j] = Ad
            My[i][j] = Ay

    myStructA['year'] = Ay
    myStructA['data'] = Ad

    DATA = myStructA['data']

    IndicesY = np.ones(ny, dtype=int)
    for i in range(ny):
        yearT = datetime(int(YEAR[i]), 1, 1)
        IndicesY[i] = TIME_full[TIME_full == yearT].index[0]

    # Ensure output directory exists
    output_dir = 'output_data'
    os.makedirs(output_dir, exist_ok=True)

    # Save the results to a .mat file
    fileout = os.path.join(output_dir, f"{site}_DATAinput_annual.mat")
    savemat(fileout, {'myStructA': myStructA})
    print(f"Saved results to {fileout}")

if __name__ == '__main__':
    main()