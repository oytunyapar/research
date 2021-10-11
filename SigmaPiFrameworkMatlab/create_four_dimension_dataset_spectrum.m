dimension = 4;
iterator_begin = 0;
iterator_end = 200;
number_of_variables = 2^dimension;

%bent_functions = findBentFunctions(4);

%number_of_bent_functions = size(bent_functions,1);

%parfor iterator = 1:number_of_bent_functions
%    createDataSetSpectrum(bent_functions(iterator), dimension);
%end

parfor func = iterator_begin:iterator_end
    createDataSetSpectrum(func, dimension);
end