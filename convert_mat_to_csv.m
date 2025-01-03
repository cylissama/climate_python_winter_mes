% Convert a .mat file to a .csv file.
%
% Parameters:
% mat_file_path (str): Path to the .mat file.
% csv_file_path (str): Path to the output .csv file.

outputfolder = 'output_data/';

[folder, name, ~] = fileparts(mat_file_path);
csv_file_path = fullfile(outputfolder, [name, '.csv']);

try
    % Load the .mat file
    data = load(mat_file_path);
    
    % Check if the 'TT_hourly' variable exists
    if ~isfield(data, 'TT_hourly')
        fprintf('Warning: ''TT_hourly'' variable missing in %s. Skipping.\n', mat_file_path);
        return;
    end
    
    % Extract the TT_hourly data
    TT_hourly = data.TT_hourly;
    
    % Convert the data to a table
    % df_hourly = struct2table(TT_hourly);
    
    % Convert TIMESTAMP from numeric to datetime if needed:
    % df_hourly.TIMESTAMP = datetime(df_hourly.TIMESTAMP, 'ConvertFrom', 'posixtime');
    
    % Write the table to a CSV file
    writetimetable(TT_hourly, csv_file_path);
    fprintf('Converted %s to %s\n', mat_file_path, csv_file_path);
    
catch ME
    if strcmp(ME.identifier, 'MATLAB:load:couldNotReadFile')
        fprintf('File not found: %s. Skipping.\n', mat_file_path);
    else
        fprintf('An error occurred while converting %s: %s\n', mat_file_path, ME.message);
    end
end
