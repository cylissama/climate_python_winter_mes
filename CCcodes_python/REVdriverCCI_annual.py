"""
REVdriverCCI_annual.py

Python implementation of MATLAB code for processing climate data.
Handles data loading and analysis of climate indices.

Author: Cy Dixon
Date: 2024-01-06
"""

import os
import numpy as np
from scipy.io import loadmat
import pandas as pd

def main():
    # Set format for numerical output
    np.set_printoptions(precision=15)
    
    # Define site and paths
    site = 'BMTN'
    dircli1 = '/Users/erappin/Documents/Mesonet/ClimateIndices/sitesTEST_CCindices/'
    dircli = os.path.join(dircli1, site)
    
    # Load threshold file
    filethresh = os.path.join(dircli, f"{site}_CLIthresh_daily.mat")
    thresh_data = loadmat(filethresh)
    
    # Load annual file
    fileannual = os.path.join(dircli, f"{site}_DATAinput_annual.mat")
    annual_data = loadmat(fileannual)
    
    # Extract dimensions
    DATAannual = annual_data['DATAannual']
    ny = len(DATAannual['year'][0][0])
    nv = len(DATAannual['var'][0][0])
    nd = 365
    
    # Extract data
    DATAyear = DATAannual['year'][0][0]
    DATAall = DATAannual['data'][0][0]

if __name__ == '__main__':
    main()