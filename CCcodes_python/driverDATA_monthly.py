import numpy as np
import pandas as pd
import datetime
import scipy.io as sio
import os


def driver_data_monthly(site='HCKM',
                        file_mes1='01-Nov-2009_01-Aug-2023_HCKM_daily.mat',
                        date_start=datetime.datetime(2009, 11, 1),
                        date_end=datetime.datetime(2023, 7, 31)):
    """
    Processes daily weather station data into monthly datasets

    Parameters:
    -----------
    site : str
        Site identifier
    file_mes1 : str
        Filename of the daily data .mat file
    date_start : datetime
        Start date for processing
    date_end : datetime
        End date for processing

    Returns:
    --------
    data_month : dict
        Dictionary containing the monthly organized data
    """

    # Setup paths and load data
    dircli1 = '/Users/erappin/Documents/Mesonet/ClimateIndices/sitesTEST_CCindices/'
    dircli = os.path.join(dircli1, site, '')
    file_thresh = os.path.join(dircli, f"{site}_CLIthresh_daily.mat")
    file_mes = os.path.join(dircli, file_mes1)

    # Load threshold data
    thresh_data = sio.loadmat(file_thresh)

    # Load station data
    station_data = sio.loadmat(file_mes)
    tt_daily_mes = pd.DataFrame(station_data['TT_dailyMES'])

    # Extract timestamp column and convert to datetime
    time_full = pd.to_datetime(tt_daily_mes['TimestampCollected'].iloc[:, 0])

    # Filter data based on date range
    mask = (time_full >= date_start) & (time_full <= date_end)
    tt = tt_daily_mes.loc[mask]

    # Rename variables to include "_month" suffix
    var_old = ["TAIR", "TAIRx", "TAIRn", "DWPT", "PRCP", "PRES", "RELH", "SM02", "SRAD", "WDIR", "WSPD", "WSMX"]
    var_new = [f"{v}_month" for v in var_old]

    # Remove unnecessary variables
    vars_to_remove = ['SM20', 'SM04', 'SM40', 'SM08', 'ST02', 'ST20', 'ST04', 'ST40', 'ST08', 'WSMN']
    for var in vars_to_remove:
        if var in tt.columns:
            tt = tt.drop(var, axis=1)

    # Rename variables
    for old, new in zip(var_old, var_new):
        if old in tt.columns:
            tt = tt.rename(columns={old: new})

    # Extract time information
    time = pd.to_datetime(tt['TimestampCollected'].iloc[:, 0])

    # Get unique years and month names
    years = np.unique(time.year)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Initialize data_month dictionary
    data_month = {
        'year': years,
        'month': months,
        'var': var_new,
        'data': {}
    }

    # Initialize month data lists
    month_data_lists = [[] for _ in range(12)]

    # Process data by month
    for k, year in enumerate(years):
        for m in range(12):
            month_num = m + 1
            # Create date range for the month
            lower = datetime.datetime(year, month_num, 1)
            if month_num in [4, 6, 9, 11]:  # 30-day months
                upper = datetime.datetime(year, month_num, 30)
            elif month_num == 2:  # February
                if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):  # Leap year
                    upper = datetime.datetime(year, month_num, 29)
                else:
                    upper = datetime.datetime(year, month_num, 28)
            else:  # 31-day months
                upper = datetime.datetime(year, month_num, 31)

            # Filter data for this month and year
            month_mask = (time >= lower) & (time <= upper)
            month_data = tt.loc[month_mask]

            # Store month data
            if not month_data.empty:
                month_data_lists[m].append((year, month_data))
            else:
                month_data_lists[m].append((year, None))

    # Save month data to data_month
    for m in range(12):
        data_month['data'][months[m]] = month_data_lists[m]

    return data_month