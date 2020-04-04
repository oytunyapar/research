function createDataSet(problem_class, dimension)
    
    [all_combination_map,all_conflict_map] = findZeroables(problem_class,dimension);
    
    number_of_variables = 2^dimension;
    
    function_matrix = repmat(ix2prob(problem_class, number_of_variables)',...
        [2^(number_of_variables) 1]);
    
    input_data = cat(2, function_matrix, all_combination_map);
    
    folder_name = sprintf('dimension_%d',dimension);
    createFolder(folder_name);
    
    input_file_name = strcat(folder_name, sprintf('/%d.input', problem_class));
    output_file_name = strcat(folder_name, sprintf('/%d.output', problem_class));
    
    dlmwrite(input_file_name, input_data, 'delimiter', ' ');
    dlmwrite(output_file_name, all_conflict_map, 'delimiter', ' ');
end

