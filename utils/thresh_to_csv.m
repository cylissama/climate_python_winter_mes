% filepath: /Volumes/Mesonet/winter_break/utils/thresh_to_csv.m

function csv_file_out = thresh_to_csv(mat_file_path)
    % Choose output folder for CSV
    outputfolder = 'output_data/';
    
    disp(['Processing file: ' mat_file_path])

    % Construct the output CSV path
    [~, name, ~] = fileparts(mat_file_path);
    csv_file_path = fullfile(outputfolder, [name, '.csv']);

    try
        % Load the .mat file
        data = load(mat_file_path);
        
        % Display all variables in the .mat file
        fprintf('\nVariables in %s:\n', name);
        fprintf('------------------------\n');
        fields = fieldnames(data);
        for i = 1:length(fields)
            varName = fields{i};
            varData = data.(varName);
            varSize = size(varData);
            varClass = class(varData);
            fprintf('Variable: %s\n', varName);
            fprintf('Type: %s\n', varClass);
            fprintf('Size: [%s]\n', strjoin(string(varSize), 'x'));
            fprintf('------------------------\n');
        end
end