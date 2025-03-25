"""
RunLength.py

Python implementation of the RunLength function used in the MATLAB climate code.
This function calculates run lengths of True values in boolean arrays.

Functions:
- run_length: Identifies consecutive True values and returns run information

Author: Converted from MATLAB
Date: 2025-03-24
"""

import numpy as np

def run_length(arr):
    """
    Calculate run lengths of True values in boolean array
    
    Parameters:
    -----------
    arr : array-like
        Boolean array
        
    Returns:
    --------
    B : array
        Boolean values (True for runs)
    N : array
        Length of each run
    BI : array
        Starting indices of runs
    """
    # Ensure input is a numpy array
    arr = np.asarray(arr)
    
    # Handle edge cases
    if arr.size == 0:
        return np.array([], dtype=bool), np.array([], dtype=int), np.array([], dtype=int)
    
    # For all-False array, return empty arrays
    if not np.any(arr):
        return np.array([], dtype=bool), np.array([], dtype=int), np.array([], dtype=int)
    
    # Find run starts and ends
    # Add boundary elements to handle runs at the start and end
    padded = np.concatenate(([False], arr, [False]))
    
    # Find transitions
    diff = np.diff(padded.astype(int))
    
    # Start positions are where diff is 1 (False->True)
    run_starts = np.where(diff > 0)[0]
    
    # End positions are where diff is -1 (True->False)
    run_ends = np.where(diff < 0)[0]
    
    # Calculate run lengths
    N = run_ends - run_starts
    
    # B indicates which elements are part of runs
    B = np.ones_like(N, dtype=bool)
    
    # BI contains starting indices of runs
    BI = run_starts
    
    return B, N, BI

def test_run_length():
    """Test the run_length function with various inputs"""
    # Test case 1: Simple alternating pattern
    arr1 = np.array([True, True, False, True, True, True, False, False, True])
    B1, N1, BI1 = run_length(arr1)
    print("Test case 1:")
    print(f"B: {B1}")
    print(f"N: {N1}")
    print(f"BI: {BI1}")
    
    # Test case 2: All True
    arr2 = np.array([True, True, True, True])
    B2, N2, BI2 = run_length(arr2)
    print("\nTest case 2:")
    print(f"B: {B2}")
    print(f"N: {N2}")
    print(f"BI: {BI2}")
    
    # Test case 3: All False
    arr3 = np.array([False, False, False])
    B3, N3, BI3 = run_length(arr3)
    print("\nTest case 3:")
    print(f"B: {B3}")
    print(f"N: {N3}")
    print(f"BI: {BI3}")
    
    # Test case 4: Empty array
    arr4 = np.array([])
    B4, N4, BI4 = run_length(arr4)
    print("\nTest case 4:")
    print(f"B: {B4}")
    print(f"N: {N4}")
    print(f"BI: {BI4}")
    
    # Test case 5: Single element
    arr5 = np.array([True])
    B5, N5, BI5 = run_length(arr5)
    print("\nTest case 5:")
    print(f"B: {B5}")
    print(f"N: {N5}")
    print(f"BI: {BI5}")

if __name__ == "__main__":
    # Run tests if executed as a script
    test_run_length()
