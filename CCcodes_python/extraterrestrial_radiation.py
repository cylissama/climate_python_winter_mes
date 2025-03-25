"""
extraterrestrial_radiation.py

Calculates extraterrestrial radiation (Ra) using the FAO Penman-Monteith methodology.
This calculation is essential for various climate indices, particularly for
potential evapotranspiration (PET) estimations.

References:
- Allen, R.G., Pereira, L.S., Raes, D., Smith, M., 1998. Crop evapotranspiration:
  Guidelines for computing crop water requirements. FAO Irrigation and Drainage
  Paper 56, Food and Agriculture Organization of the United Nations, Rome.

Author: Climate Data Team
Date: 2025-03-24
"""

import numpy as np


def extraterrestrial_radiation(doy, lat):
    """
    Calculate extraterrestrial radiation (Ra) using the FAO formula

    Parameters:
    -----------
    doy : int or array-like
        Day of the year (1-365/366)
    lat : float or array-like
        Latitude in degrees (-90 to 90)

    Returns:
    --------
    Ra : float or array-like
        Extraterrestrial radiation (MJ/m²/day)

    Examples:
    ---------
    >>> extraterrestrial_radiation(180, 45)  # June 29th at 45°N
    41.8...

    >>> import numpy as np
    >>> days = np.array([15, 46, 74])  # Jan 15, Feb 15, Mar 15
    >>> extraterrestrial_radiation(days, 35)  # Multiple days at 35°N
    array([19.1..., 25.2..., 33.4...])
    """
    # Input validation
    lat = np.asarray(lat)
    doy = np.asarray(doy)

    if np.any((lat < -90) | (lat > 90)):
        raise ValueError("Latitude must be between -90 and 90 degrees")

    if np.any((doy < 1) | (doy > 366)):
        raise ValueError("Day of year must be between 1 and 366")

    # Convert latitude to radians
    phi = np.deg2rad(lat)

    # Solar declination (radians)
    # The angle between the rays of the sun and the plane of the earth's equator
    delta = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)

    # Inverse relative Earth-Sun distance squared
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)

    # Sunset hour angle (radians)
    # The hour angle at which the sun sets
    with np.errstate(invalid='ignore'):  # Handle warnings at extreme latitudes
        ws = np.arccos(-np.tan(phi) * np.tan(delta))
        # Fix edge cases at poles where arccos can give NaN
        if np.any(np.isnan(ws)):
            ws = np.where(np.isnan(ws), np.pi if np.abs(lat) < 66.5 else 0, ws)

    # Solar constant (MJ/m²/min)
    Gsc = 0.0820

    # Extraterrestrial radiation
    # Ra = 24*60/π * Gsc * dr * [ws*sin(φ)*sin(δ) + cos(φ)*cos(δ)*sin(ws)]
    Ra = 24 * 60 / np.pi * Gsc * dr * (
            ws * np.sin(phi) * np.sin(delta) +
            np.cos(phi) * np.cos(delta) * np.sin(ws)
    )

    return Ra


def test_extraterrestrial_radiation():
    """Test the extraterrestrial_radiation function with known values"""
    # Test values from FAO-56 Table 2.6 (Allen et al., 1998)
    # Lat = 50°N
    test_days = [15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]  # 15th of each month
    test_lat = 50

    # Expected values (MJ/m²/day) from FAO paper
    expected = [8.8, 15.0, 22.9, 32.4, 38.8, 41.6, 40.2, 34.4, 25.2, 16.2, 9.5, 7.2]

    # Calculate values
    calculated = extraterrestrial_radiation(test_days, test_lat)

    # Print results
    print("Testing extraterrestrial radiation calculation:")
    print(f"{'Month':>5} {'Day':>5} {'Expected':>10} {'Calculated':>10} {'Diff (%)':>10}")
    print("-" * 45)

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    for i, month in enumerate(months):
        diff_pct = (calculated[i] - expected[i]) / expected[i] * 100
        print(f"{month:>5} {test_days[i]:>5} {expected[i]:>10.2f} {calculated[i]:>10.2f} {diff_pct:>10.2f}")


if __name__ == "__main__":
    test_extraterrestrial_radiation()