dimension = 4;
number_of_variables = 2^dimension;

iterator_final = 2^number_of_variables;

bent_functions = findBentFunctions(4);

number_of_bent_functions = size(bent_functions,1);

parfor iterator = 1:number_of_bent_functions
    createDataSetSpectrum(bent_functions(iterator), dimension);
end