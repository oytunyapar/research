function [functions, max_scores, max_populations] = minTermKGeneticAlgortihmRunner(dimension)

number_of_variables = 2^dimension;
functions = BFEquivalentFunctions(dimension);
length_functions = length(functions);
max_scores = zeros(1,length_functions);
max_populations = zeros(length_functions,number_of_variables);

for iterator = 1:length_functions
    fprintf("Dimension:%d Function:%d/%d\n",dimension,iterator,length_functions);
    [~,~,exit_flag,~,population,scores] = ...
        findMinTermKGeneticAlgortihm(functions(iterator), dimension);
    [max_value, max_value_index] = max(scores);
    max_scores(iterator) = max_value;
    max_populations(iterator,:) = population(max_value_index,:);
end

end