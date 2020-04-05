function functions = findBentFunctions(dimension)
    
    functions = [];
    
    if mod(dimension, 2) ~= 0
        fprintf("There is not any bent function in odd dimensions.\n");
        return
    end
    
    number_of_variables = 2^dimension;
    number_of_functions = 2^number_of_variables;

    for function_iterator = 0:number_of_functions
        spectrum = abs(...
                       sum(...
                           diag(ix2prob(function_iterator,number_of_variables)) * ...
                           monsetup(dimension)...
                       )...
                   );
        
        if all(spectrum == spectrum(1))
            functions = [functions;function_iterator];
        end
    end
end

