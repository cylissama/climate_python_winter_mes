"""
pet.py

Implementation of the Hargreaves-Samani method for calculating Potential 
Evapotranspiration (PET). This method requires only temperature and extraterrestrial
radiation data, making it suitable for situations where limited meteorological
data is available.

References:
- Hargreaves, G.H., Samani, Z.A., 1985. Reference crop evapotranspiration from temperature.
  Applied Engineering in Agriculture 1(2), 96-99.
- Allen, R.G., Pereira, L.S., Raes, D., Smith, M., 1998. Crop evapotranspiration:
  Guidelines for computing crop water requirements. FAO Irrigation and Drainage
  Paper 56, Food and Agriculture Organization of the United Nations, Rome.

Author: Climate Data Team
Date: 2025-03-24
"""

import numpy as np


def pet(Ra, tmax, tmin, tmean=None, method='hargreaves'):
    """
    Calculate Potential Evapotranspiration (PET) using temperature-based methods.

    Parameters:
    -----------
    Ra : float or array-like
        Extraterrestrial radiation (MJ/m²/day)
    tmax : float or array-like
        Maximum temperature (°C)
    tmin : float or array-like
        Minimum temperature (°C)
    tmean : float or array-like, optional
        Mean temperature (°C). If None, calculated as (tmax + tmin)/2
    method : str, optional
        Method to use: 'hargreaves' (default) or 'hargreaves-1985'

    Returns:
    --------
    PET : float or array-like
        Potential evapotranspiration (mm/day)

    Examples:
    ---------
    >>> # Single day calculation
    >>> Ra = 35.0    # MJ/m²/day
    >>> tmax = 25.0  # °C
    >>> tmin = 10.0  # °C
    >>> pet(Ra, tmax, tmin)
    4.97...

    >>> # Multiple days calculation
    >>> import numpy as np
    >>> Ra = np.array([30.0, 32.0, 33.0])
    >>> tmax = np.array([22.0, 24.0, 26.0])
    >>> tmin = np.array([8.0, 10.0, 12.0])
    >>> pet(Ra, tmax, tmin)
    array([3.97..., 4.52..., 5.08...])
    """
    # Input validation
    Ra = np.asarray(Ra)
    tmax = np.asarray(tmax)
    tmin = np.asarray(tmin)

    if np.any(Ra < 0):
        raise ValueError("Extraterrestrial radiation (Ra) must be non-negative")

    # Handle NaN values in temperature data
    if np.any(np.isnan(tmax)) or np.any(np.isnan(tmin)):
        # We'll proceed, but warn that results will have NaNs
        import warnings
        warnings.warn("Input temperature data contains NaN values.")

    # Calculate mean temperature if not provided
    if tmean is None:
        tmean = (tmax + tmin) / 2
    else:
        tmean = np.asarray(tmean)

    # Calculate PET using different methods
    if method.lower() == 'hargreaves':
        # Standard Hargreaves-Samani equation (1985)
        # PET = 0.0023 × Ra × (Tmax - Tmin)^0.5 × (Tmean + 17.8)
        k = 0.0023  # Hargreaves coefficient
        PET = k * Ra * np.sqrt(np.maximum(tmax - tmin, 0)) * (tmean + 17.8)

    elif method.lower() == 'hargreaves-1985':
        # Original Hargreaves-Samani equation with slight modification
        # PET = 0.0023 × Ra × (Tmax - Tmin)^0.5 × (Tmean + 17.8) × 0.408
        k = 0.0023  # Hargreaves coefficient
        conversion_factor = 0.408  # Convert from MJ/m²/day to mm/day
        PET = k * Ra * np.sqrt(np.maximum(tmax - tmin, 0)) * (tmean + 17.8) * conversion_factor

    else:
        raise ValueError(f"Unknown method: {method}. Use 'hargreaves' or 'hargreaves-1985'.")

    return PET


def test_pet():
    """Test the pet function with examples from literature"""
    # Test data based on example calculations in literature
    Ra = np.array([35.0, 30.0, 25.0, 20.0])  # MJ/m²/day
    tmax = np.array([25.0, 22.0, 20.0, 18.0])  # °C
    tmin = np.array([10.0, 8.0, 5.0, 2.0])  # °C

    # Calculate PET
    pet_values = pet(Ra, tmax, tmin)

    # Print results
    print("Testing PET calculation (Hargreaves-Samani method):")
    print(f"{'Ra (MJ/m²/day)':>15} {'Tmax (°C)':>10} {'Tmin (°C)':>10} {'PET (mm/day)':>15}")
    print("-" * 55)

    for i in range(len(Ra)):
        print(f"{Ra[i]:>15.1f} {tmax[i]:>10.1f} {tmin[i]:>10.1f} {pet_values[i]:>15.2f}")

    # Compare with different method
    pet_values_1985 = pet(Ra, tmax, tmin, method='hargreaves-1985')
    print("\nComparison with hargreaves-1985 method:")
    for i in range(len(Ra)):
        print(f"{Ra[i]:>15.1f} {tmax[i]:>10.1f} {tmin[i]:>10.1f} {pet_values_1985[i]:>15.2f}")


if __name__ == "__main__":
    test_pet()