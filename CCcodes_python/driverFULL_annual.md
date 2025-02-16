# Climate Data Analysis Pipeline
# Driver Full Annual

## Overview
This Python script processes daily climate data to generate aggregated statistics, thermal indices, and growing degree days (GDD) calculations. It converts raw daily meteorological observations into structured analyses for agricultural and climate research purposes.

## Key Functionalities
- **Data Ingestion**: Reads CSV files containing daily climate observations
- **Temporal Filtering**: Focuses on complete years between start+1 and end-1 years
- **Statistical Aggregation**:
  - Monthly means
  - Annual means
- **Thermal Indices Calculation**:
  - Frost Days (FD)
  - Consecutive Frost Days (CFD)
  - Heating Degree Days (HDD)
  - Ice Days (ID)
- **Agricultural Metrics**:
  - Growing Degree Days (GD4 and GD10)
- **Data Export**: Structured CSV outputs for analysis and visualization

## Input Data Requirements
**File Format**: CSV with daily records  
**Required Columns**:
- `TimestampCollected` (datetime format)
- `TAIR` (Air Temperature)
- `DWPT` (Dew Point Temperature) 
- `TAIRx` (Daily Max Temperature)
- `TAIRn` (Daily Min Temperature)
- `PRCP` (Precipitation)
- `RELH` (Relative Humidity)
- `PRES` (Pressure)
- `SM02` (Soil Moisture)
- `WDIR` (Wind Direction)
- `WSPD` (Wind Speed)
- `WSMX` (Max Wind Speed)
- `SRAD` (Solar Radiation)

## Output Files Structure

### 1. Metadata (`metadata.csv`)
- **Format**: Key-value pairs
- **Contents**:
  - Analysis start/end dates
  - List of processed variables
  - Input file information

### 2. Monthly Statistics (`monthly_statistics.csv`)
- **Structure**:
  - Index: Month-end dates (YYYY-MM-DD)
  - Columns: All climate variables
  - Values: Monthly averages

### 3. Annual Statistics (`annual_statistics.csv`)
- **Structure**:
  - Index: Year-end dates (YYYY-MM-DD)
  - Columns: All climate variables
  - Values: Annual averages

### 4. Growing Degree Days (`growing_degree_days.csv`)
- **Structure**:
  | Year | Month | GD4 | GD10 |
  |------|-------|-----|------|
  | 2015 | 1     | 120 | 85   |
  | 2015 | 2     | 95  | 62   |
  
- **Calculations**:
  - GD4: ∑(max(T<sub>air</sub> - 4°C, 0))
  - GD10: ∑(max(T<sub>air</sub> - 10°C, 0))

### 5. Temperature Indices (`temperature_indices.csv`)
- **Structure**:
  | TimestampCollected | FD | CFD | HDD | ID |
  |---------------------|----|-----|-----|----|
  | 2015-01-15          | 1  | 3   | 5.2 | 0  |
  
- **Indices Definitions**:
  - **FD**: Frost Days (T<sub>min</sub> < 0°C)
  - **CFD**: Max consecutive frost days
  - **HDD**: ∑(18.3°C - T<sub>air</sub>)
  - **ID**: Ice Days (T<sub>max</sub> < 0°C)

### 6. Daily Data (`daily_data.csv`)
- **Structure**: Mirror of input CSV with filtered dates
- **Purpose**: Provides cleaned baseline data for validation

## Processing Workflow
1. **Data Loading**  
   - Reads CSV with strict datetime parsing
   - Validates required columns exist

2. **Date Filtering**  
   - Removes first/lest partial years
   - Ensures only complete annual cycles

3. **Statistical Resampling**  
   ```python
   monthly = data.resample('ME').mean()  # Month-end averages
   annual = data.resample('YE').mean()   # Year-end averages


---

### Usage Requirements

pandas >= 1.4
numpy >= 1.21
pathlib

### Execution

python driver_full_annual.py


### Benefits from the original MATLAB

Open Source: No license requirements
Reproducibility: Clear dependency management
Scalability: Handles large datasets efficiently
Interoperability: Standard CSV format outputs
Maintainability: Modular Pythonic structure