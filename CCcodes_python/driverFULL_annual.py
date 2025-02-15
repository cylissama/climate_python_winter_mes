import numpy as np
import pandas as pd
import scipy.io
import os

# Set display format for numpy arrays
np.set_printoptions(precision=15, floatmode='maxprec')

# Helper Functions ============================================================
def getData_annual(TT, iy, TIME):
    """Extract annual data for all variables and return as DataFrame"""
    TIME_annual = pd.DatetimeIndex(TIME[iy])
    annual_data = {}
    
    vars_to_process = [
        ('TAIR', 'TAIR_annual'), ('DWPT', 'DWPT_annual'),
        ('TAIRx', 'TAIRx_annual'), ('TAIRn', 'TAIRn_annual'),
        ('PRCP', 'PRCP_annual'), ('RELH', 'RELH_annual'),
        ('PRES', 'PRES_annual'), ('SM02', 'SM02_annual'),
        ('WDIR', 'WDIR_annual'), ('WSPD', 'WSPD_annual'),
        ('WSMX', 'WSMX_annual'), ('SRAD', 'SRAD_annual')
    ]
    
    for var_src, var_dest in vars_to_process:
        try:
            var_array = TT[var_src][iy] if TT[var_src].ndim > 1 else TT[var_src][iy]
            annual_data[var_dest] = var_array.flatten()
        except KeyError:
            raise ValueError(f"Missing variable {var_src} in dataset")

    return pd.DataFrame({
        'TIME_annual': TIME_annual,
        **annual_data
    }).set_index('TIME_annual').rename(columns=lambda x: x.replace('_annual', ''))

def getData_month(TT, mo, TIME):
    """Extract monthly data for all variables and return as DataFrame"""
    TIME_month = pd.DatetimeIndex(TIME[mo])
    
    vars_to_process = [
        ('TAIR', 'TAIR_month'), ('DWPT', 'DWPT_month'),
        ('TAIRx', 'TAIRx_month'), ('TAIRn', 'TAIRn_month'),
        ('PRCP', 'PRCP_month'), ('RELH', 'RELH_month'),
        ('PRES', 'PRES_month'), ('SM02', 'SM02_month'),
        ('WDIR', 'WDIR_month'), ('WSPD', 'WSPD_month'),
        ('WSMX', 'WSMX_month'), ('SRAD', 'SRAD_month')
    ]
    
    monthly_data = {}
    for var_src, var_dest in vars_to_process:
        try:
            var_array = TT[var_src][mo] if TT[var_src].ndim > 1 else TT[var_src][mo]
            monthly_data[var_dest] = var_array.flatten()
        except KeyError:
            raise ValueError(f"Missing required variable {var_src} in TT dataset")
    
    return pd.DataFrame({
        'TIME_month': TIME_month,
        **monthly_data
    }).set_index('TIME_month').rename(columns=lambda x: x.replace('_month', ''))

def calculate_run_length(arr):
    """Calculate maximum consecutive True values in a boolean array"""
    max_count = count = 0
    for val in arr:
        count = count + 1 if val else 0
        max_count = max(max_count, count)
    return max_count

def calculate_gdd(data, years, time_index, isD, ieD):
    """Calculate growing degree days for GD4 and GD10"""
    nm = 12
    gdd4 = np.full((len(years), nm), np.nan)
    gdd10 = np.full((len(years), nm), np.nan)
    
    for y_idx, year in enumerate(years):
        year_mask = (time_index.year == year)
        for m_idx in range(nm):
            mask = year_mask & (time_index.month == m_idx+1)
            if not mask.any(): continue
                
            try:
                tair = data['TAIR'][isD:ieD+1][mask]
                gdd4[y_idx, m_idx] = np.nansum(np.maximum(tair - 4, 0))
                gdd10[y_idx, m_idx] = np.nansum(np.maximum(tair - 10, 0))
            except KeyError:
                print(f"Missing TAIR data for {year}-{m_idx+1}")
                
    return {'GD4': gdd4, 'GD10': gdd10}

# Main Processing Function ====================================================
def main_processing():
    # Configuration
    site = 'BMNT'
    data_root = '/Volumes/Mesonet/winter_break/CCdata/'
    
    try:
        # Load data
        mesonet = scipy.io.loadmat(os.path.join(data_root, 'BMTN/01-Mar-2014_01-Aug-2023_BMTN_daily.mat'))
        time_data = pd.DatetimeIndex(mesonet['TimestampCollected'].flatten())
        
        # Date range calculations
        date_start = time_data[0] + pd.DateOffset(years=1)
        date_end = time_data[-1] - pd.DateOffset(years=1)
        isD = max(np.searchsorted(time_data, date_start.normalize(), side='left'), 0)
        ieD = min(np.searchsorted(time_data, date_end.normalize(), side='right'), len(time_data)-1)

        # Data validation
        try:
            TT = mesonet['TT_dailyMES'][isD:ieD+1]
        except KeyError:
            raise RuntimeError("Required dataset TT_dailyMES not found in MAT file")

        # Initialize output structure
        results = {
            'metadata': {
                'site': site,
                'start_date': time_data[isD],
                'end_date': time_data[ieD],
                'variables': [
                    'TAIR', 'DWPT', 'TAIRx', 'TAIRn', 'PRCP', 'RELH',
                    'PRES', 'SM02', 'WDIR', 'WSPD', 'WSMX', 'SRAD'
                ]
            },
            'years': np.unique(time_data[isD:ieD+1].year),
            'monthly': None,
            'annual': [],
            'indices': {},
            'temp_indices': {},
            'gdd': None
        }

        ny = len(results['years'])
        nm = 12

        # Initialize matrices
        FDm = np.zeros((ny, nm))
        CFDm = np.zeros((ny, nm))
        HDDm = np.zeros((ny, nm))
        IDm = np.zeros((ny, nm))
        TXnm = np.full((ny, nm), np.nan)
        TNnm = np.full((ny, nm), np.nan)

        # Process annual and monthly data
        monthly_data = np.full((ny, nm, len(results['metadata']['variables'])), np.nan)
        
        for y_idx, year in enumerate(results['years']):
            year_mask = (time_data[isD:ieD+1].year == year)
            
            try:
                # Annual data processing
                df_year = getData_annual(TT, year_mask, time_data)
                results['annual'].append(df_year)
                
                # Monthly aggregates
                monthly_means = df_year.groupby(df_year.index.month).mean()
                monthly_data[y_idx] = monthly_means.values.T

                # Temperature indices calculation
                for m_idx in range(nm):
                    time_mask = year_mask & (time_data[isD:ieD+1].month == m_idx+1)
                    
                    TAIR = TT['TAIR'][time_mask]
                    TAIRn = TT['TAIRn'][time_mask]
                    TAIRx = TT['TAIRx'][time_mask]

                    # Temperature extremes
                    TXnm[y_idx, m_idx] = np.nanmin(TAIRx)
                    TNnm[y_idx, m_idx] = np.nanmin(TAIRn)
                    
                    # Thermal indices
                    FDm[y_idx, m_idx] = np.nansum(TAIRn < 0)
                    frost_mask = TAIRn < 0
                    CFDm[y_idx, m_idx] = calculate_run_length(frost_mask) if np.any(frost_mask) else 0
                    HDDm[y_idx, m_idx] = np.nansum(18.3 - TAIR)
                    IDm[y_idx, m_idx] = np.nansum(TAIRx < 0)

            except Exception as e:
                print(f"Error processing {year}: {str(e)}")
                continue

        # Store results
        results['monthly'] = monthly_data
        results['gdd'] = calculate_gdd(TT, results['years'], time_data[isD:ieD+1], isD, ieD)
        results['temp_indices'] = {
            'FD': FDm, 'CFD': CFDm, 'HDD': HDDm, 'ID': IDm,
            'TXn': TXnm, 'TNn': TNnm,
            'years': results['years'], 'months': np.arange(1, 13)
        }

        print(f"Processing completed for {len(results['years'])} years")
        return results

    except Exception as e:
        print(f"Critical error in main processing: {str(e)}")
        raise

if __name__ == "__main__":
    analysis_results = main_processing()