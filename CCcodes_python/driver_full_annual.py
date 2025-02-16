import numpy as np
import pandas as pd
import os
from pathlib import Path

# Set display format for numpy arrays
np.set_printoptions(precision=15, floatmode='maxprec')

# Helper Functions ============================================================
def calculate_run_length(arr):
    """Calculate maximum consecutive True values in a boolean array"""
    max_count = count = 0
    for val in arr:
        count = count + 1 if val else 0
        max_count = max(max_count, count)
    return max_count

def calculate_gdd(data, years, time_index):
    """Calculate growing degree days for GD4 and GD10"""
    nm = 12  # Number of months
    # Initialize DataFrames with proper structure
    gdd4 = pd.DataFrame(
        np.full((len(years), nm), np.nan),
        index=years,
        columns=range(1, nm+1)
    )
    gdd10 = pd.DataFrame(
        np.full((len(years), nm), np.nan),
        index=years,
        columns=range(1, nm+1)
    )
    
    for y_idx, year in enumerate(years):
        for m_idx in range(nm):
            try:
                # Create mask for year-month combination
                mask = (time_index.year == year) & (time_index.month == m_idx+1)
                tair = data.loc[mask, 'TAIR'].values
                gdd4.iloc[y_idx, m_idx] = np.nansum(np.maximum(tair - 4, 0))
                gdd10.iloc[y_idx, m_idx] = np.nansum(np.maximum(tair - 10, 0))
            except KeyError:
                print(f"Missing TAIR data for {year}-{m_idx+1}")
                
    return {'GD4': gdd4, 'GD10': gdd10}

def save_results(results, output_dir="results"):
    """Save all results to CSV files in organized directory structure"""
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        pd.DataFrame.from_dict(results['metadata'], orient='index').to_csv(
            os.path.join(output_dir, 'metadata.csv'), header=False
        )

        # Save resampled data
        results['statistics']['monthly'].to_csv(
            os.path.join(output_dir, 'monthly_statistics.csv')
        )
        results['statistics']['annual'].to_csv(
            os.path.join(output_dir, 'annual_statistics.csv')
        )

        # Save GDD data
        gdd_data = results['statistics']['gdd']['GD4'].stack().reset_index()
        gdd_data.columns = ['Year', 'Month', 'GD4']
        gdd_data['GD10'] = results['statistics']['gdd']['GD10'].stack().values
        gdd_data.to_csv(
            os.path.join(output_dir, 'growing_degree_days.csv'), index=False
        )

        # Save indices
        results['statistics']['indices'].to_csv(
            os.path.join(output_dir, 'temperature_indices.csv')
        )

        # Save daily data
        results['statistics']['daily'].to_csv(
            os.path.join(output_dir, 'daily_data.csv')
        )

        print(f"Results saved to {output_dir} directory")
    except Exception as e:
        print(f"Error saving results: {str(e)}")
        raise

# Main Processing Function ====================================================
def main_processing():
    # Configuration
    data_path = '/Volumes/Mesonet/winter_break/output_data/01-Mar-2014_01-Aug-2023_BMTN_daily.csv'
    output_dir = '/Volumes/Mesonet/winter_break/CCdata/results'

    try:
        # Load and prepare data
        df = pd.read_csv(
            data_path,
            parse_dates=['TimestampCollected'],
            index_col='TimestampCollected'
        )
        df = df[[
            'TAIR', 'DWPT', 'TAIRx', 'TAIRn', 'PRCP', 
            'RELH', 'PRES', 'SM02', 'WDIR', 'WSPD', 'WSMX', 'SRAD'
        ]]
        
        # Filter to valid date range
        time_data = df.index
        valid_mask = (time_data >= time_data[0] + pd.DateOffset(years=1)) & \
                     (time_data <= time_data[-1] - pd.DateOffset(years=1))
        TT = df[valid_mask]
        
        # Initialize results structure
        results = {
            'metadata': {
                'start_date': TT.index[0],
                'end_date': TT.index[-1],
                'variables': TT.columns.tolist()
            },
            'statistics': {
                'monthly': TT.resample('ME').mean(),
                'annual': TT.resample('YE').mean(),
                'gdd': None,
                'indices': None,
                'daily': TT
            }
        }

        # Calculate indices
        years = TT.index.year.unique().tolist()
        results['statistics']['gdd'] = calculate_gdd(TT, years, TT.index)
        
        indices = pd.DataFrame(index=TT.index)
        indices['FD'] = (TT['TAIRn'] < 0).astype(int)
        indices['CFD'] = indices.groupby(indices.index.year)['FD'].transform(calculate_run_length)
        indices['HDD'] = 18.3 - TT['TAIR']
        indices['ID'] = (TT['TAIRx'] < 0).astype(int)
        results['statistics']['indices'] = indices

        save_results(results, output_dir)
        print(f"Processed {len(TT)} days from {TT.index[0]} to {TT.index[-1]}")
        return results

    except Exception as e:
        print(f"Error processing data: {str(e)}")
        raise

if __name__ == "__main__":
    results = main_processing()