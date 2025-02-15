function PET = pet(Ra, tmax, tmin, tmean)
% Hargreaves-Samani Potential Evapotranspiration (PET) formula
% Inputs:
%   Ra    : Extraterrestrial radiation (MJ/m²/day)
%   tmax  : Maximum temperature (°C)
%   tmin  : Minimum temperature (°C)
%   tmean : Mean temperature (°C)
% Output:
%   PET   : Potential evapotranspiration (mm/day)

% Hargreaves-Samani coefficient (typically 0.0023)
k = 0.0023; 

% Calculate PET
PET = k .* Ra .* sqrt(tmax - tmin) .* (tmean + 17.8);
end