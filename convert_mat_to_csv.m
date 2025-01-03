function csv_file_out = convert_mat_to_csv(mat_file_path)
    % CONVERT_MAT_TO_CSV Convert a .mat file containing TT_hourly to a .csv.
    %
    %   csv_file_out = convert_mat_to_csv(MAT_FILE_PATH) 
    %   reads TT_hourly from the .mat file MAT_FILE_PATH and writes it as a .csv
    %   file into the 'output_data' folder under the same base filename. 
    %   The function returns the file path of the created CSV. If no CSV was 
    %   created (e.g., error or missing TT_hourly), it returns ''.
    %
    %   Parameters:
    %       mat_file_path (char/string): Path to the .mat file.
    %
    %   Returns:
    %       csv_file_out (char/string): Path to the saved .csv file or '' on error.
    %
    % Example:
    %   csv_path = convert_mat_to_csv('/Volumes/Mesonet/cliSITES/DRFN/1980_DRFN_hourly.mat');
    %   if ~isempty(csv_path)
    %       fprintf('CSV saved at %s\n', csv_path);
    %   end

    % Default return value
    csv_file_out = '';

    % Choose output folder for CSV
    outputfolder = 'output_data/';

    disp(mat_file_path)

    % Construct the output CSV path with the same base filename
    [~, name, ~] = fileparts(mat_file_path);
    csv_file_path = fullfile(outputfolder, [name, '.csv']);

    try
        % Load the .mat file
        data = load(mat_file_path);

        % Check if 'TT_hourly' exists
        if ~isfield(data, 'TT_hourly')
            fprintf('Warning: ''TT_hourly'' variable missing in %s. Skipping.\n', mat_file_path);
            return;
        end

        % Extract the TT_hourly data
        TT_hourly = data.TT_hourly;

        % If TT_hourly is a timetable, writetimetable works directly.
        % Otherwise, convert it (e.g. struct/table -> timetable).
        %
        % For example, if TT_hourly is a struct:
        %   df_hourly = struct2table(TT_hourly);
        %   TT_hourly = table2timetable(df_hourly, 'RowTimes', yourDatetimeVar);

        % Write timetable to CSV file
        writetimetable(TT_hourly, csv_file_path);
        fprintf('Converted %s to %s\n', mat_file_path, csv_file_path);

        % On success, set the function's return value
        csv_file_out = csv_file_path;

    catch ME
        % If the MAT file could not be read
        if strcmp(ME.identifier, 'MATLAB:load:couldNotReadFile')
            fprintf('File not found: %s. Skipping.\n', mat_file_path);
        else
            fprintf('An error occurred while converting %s: %s\n', mat_file_path, ME.message);
        end
        % csv_file_out remains ''
    end
end