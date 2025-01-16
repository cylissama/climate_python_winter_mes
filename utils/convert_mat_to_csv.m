% filepath: /Volumes/Mesonet/winter_break/utils/convert_mat_to_csv.m

function csv_file_out = convert_mat_to_csv(mat_file_path, data_type)
    % CONVERT_MAT_TO_CSV Convert a .mat file containing time series data to a .csv.
    %
    %   csv_file_out = convert_mat_to_csv(MAT_FILE_PATH, DATA_TYPE) 
    %   reads data from the .mat file MAT_FILE_PATH and writes it as a .csv
    %   file. DATA_TYPE should be either 'hourly' or 'daily' to specify which
    %   type of data to convert (TT_hourly or TT_daily).
    %
    %   Parameters:
    %       mat_file_path (char/string): Path to the .mat file
    %       data_type (char/string): Either 'hourly' or 'daily'
    %
    %   Returns:
    %       csv_file_out (char/string): Path to the saved .csv file or '' on error

    % Default return value
    csv_file_out = '';
    
    % Choose output folder for CSV
    outputfolder = 'output_data/';
    
    % Validate data_type parameter
    if ~strcmp(data_type, 'hourly') && ~strcmp(data_type, 'daily') && ~strcmp(data_type, 'dailyMES')
        error('data_type must be either ''hourly'' or ''daily''');
    end
    
    disp(['Processing file: ' mat_file_path])
    disp(['Data type: ' data_type])

    % Construct the output CSV path
    [~, name, ~] = fileparts(mat_file_path);
    csv_file_path = fullfile(outputfolder, [name, '.csv']);

    try
        % Load the .mat file
        data = load(mat_file_path);
        
        % Check for appropriate variable based on data_type
        if strcmp(data_type, 'hourly')
            var_name = 'TT_hourly';
        elseif strcmp(data_type, 'daily')
            var_name = 'TT_daily';
        elseif strcmp(data_type, 'dailyMES')
            var_name = 'TT_dailyMES';
        else
            error('Invalid data_type: %s', data_type);
        end
        
        if ~isfield(data, var_name)
            fprintf('Warning: ''%s'' variable missing in %s. Skipping.\n', var_name, mat_file_path);
            return;
        end

        % Extract the data
        TT_data = data.(var_name);

        % Convert the data based on its type
        if istimetable(TT_data)
            % If it's already a timetable, write directly
            writetimetable(TT_data, csv_file_path);

        elseif isstruct(TT_data)
            % If it's a struct, convert to table first
            data_table = struct2table(TT_data);
            writetable(data_table, csv_file_path);
        else
            % For other types, try direct conversion
            writematrix(TT_data, csv_file_path);
        end

        fprintf('Successfully Converted %s to %s!\n', mat_file_path, csv_file_path);
        csv_file_out = csv_file_path;

    catch ME
        if strcmp(ME.identifier, 'MATLAB:load:couldNotReadFile')
            fprintf('File not found: %s. Skipping.\n', mat_file_path);
        else
            fprintf('An error occurred while converting %s: %s\n', mat_file_path, ME.message);
        end
    end
end