function Ra = extraterrestrial_radiation(doy, lat)
% Calculate extraterrestrial radiation (Ra) using the FAO formula
% Inputs:
%   doy : Day of the year (1-365)
%   lat : Latitude (degrees)
% Output:
%   Ra  : Extraterrestrial radiation (MJ/m²/day)

phi = deg2rad(lat); % Convert latitude to radians
dr = 1 + 0.033 * cos(2 * pi * doy / 365); % Inverse relative Earth-Sun distance
delta = 0.409 * sin(2 * pi * doy / 365 - 1.39); % Solar declination
ws = acos(-tan(phi) * tan(delta)); % Sunset hour angle
Ra = 118.1 / pi * dr .* (ws .* sin(phi) .* sin(delta) + cos(phi) .* cos(delta) .* sin(ws));
end