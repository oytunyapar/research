function failed = createDataSetSpectrum(problem_class, dimension)
    failed = 0;    

    folder_name = sprintf('dimension_%d',dimension);
    
    if exist(folder_name,'dir') ~= 7
        failed = 1;
        fprintf("Folder: %s could not be found.\n", folder_name);
        return
    end
    
    input_file_name = strcat(folder_name, sprintf('/%d.input', problem_class));
    output_file_name = strcat(folder_name, sprintf('/%d.output', problem_class));
    
    if exist(input_file_name,'file') ~= 2
        failed = 1;
        fprintf("File: %s could not be found.\n", input_file_name);
        return;
    end
    
    if exist(output_file_name,'file') ~= 2
        failed = 1;
        fprintf("File: %s could not be found.\n", output_file_name);
        return;
    end
    
    spectrum_folder_name = strcat(folder_name, "/spectrum");
    createFolder(spectrum_folder_name);
    
    spectrum_output_file_name = strcat(spectrum_folder_name, sprintf('/%d.output', problem_class));
    copyfile(output_file_name, spectrum_output_file_name);
    
    spectrum_input_file_name = strcat(spectrum_folder_name, sprintf('/%d.input', problem_class));

    number_of_variables = 2^dimension;

    spectrum = sum(...
                   diag(ix2prob(problem_class,number_of_variables)) * ...
                   monsetup(dimension)...
               );
           
    input_data = dlmread(input_file_name, ' ');
    
    input_data_row_size = size(input_data,1);
    
    for row_iterator = 1:input_data_row_size
        input_data(row_iterator,1:number_of_variables) = spectrum;
    end
    
    dlmwrite(spectrum_input_file_name, input_data, 'delimiter', ' ');
end

