import numpy as np

def pet(Ra, tmax, tmin, tmean):
    """
    Hargreaves-Samani Potential Evapotranspiration (PET) formula
    
    Parameters:
    -----------
    Ra : float or array-like
        Extraterrestrial radiation (MJ/m²/day)
    tmax : float or array-like
        Maximum temperature (°C)
    tmin : float or array-like
        Minimum temperature (°C)
    tmean : float or array-like
        Mean temperature (°C)
    
    Returns:
    --------
    PET : float or array-like
        Potential evapotranspiration (mm/day)
    """
    # Hargreaves-Samani coefficient (typically 0.0023)
    k = 0.0023
    
    # Calculate PET
    PET = k * Ra * np.sqrt(tmax - tmin) * (tmean + 17.8)
    
    return PET
