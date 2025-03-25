import numpy as np

def extraterrestrial_radiation(doy, lat):
    """
    Calculate extraterrestrial radiation (Ra) using the FAO formula
    
    Parameters:
    -----------
    doy : int or array-like
        Day of the year (1-365)
    lat : float or array-like
        Latitude in degrees
    
    Returns:
    --------
    Ra : float or array-like
        Extraterrestrial radiation (MJ/m²/day)
    """
    # Convert latitude to radians
    phi = np.deg2rad(lat)
    
    # Inverse relative Earth-Sun distance
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    
    # Solar declination
    delta = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    
    # Sunset hour angle
    ws = np.arccos(-np.tan(phi) * np.tan(delta))
    
    # Extraterrestrial radiation
    Ra = 118.1 / np.pi * dr * (ws * np.sin(phi) * np.sin(delta) + 
                              np.cos(phi) * np.cos(delta) * np.sin(ws))
    
    return Ra
