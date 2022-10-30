function [max_scores_matrix, max_populations_3d] = minTermGeneticAlgortihmRunner(functions, dimension, population_type, repeats)

number_of_variables = 2^dimension;
length_functions = length(functions);

max_scores_matrix = zeros(length_functions, repeats);
max_populations_3d = zeros(repeats, number_of_variables, length_functions);

for iterator = 1:length_functions
    max_scores = zeros(1,repeats);
    max_populations = zeros(repeats,number_of_variables);
    for repeats_iterator = 1:repeats
        fprintf("Dimension:%d Function:%d/%d Repeats:%d/%d\n",dimension,iterator,length_functions, ...
            repeats_iterator,repeats);
        [~,~,~,~,population,scores] = ...
            findMinTermGeneticAlgortihm(functions(iterator), dimension,population_type);
        [max_value, max_value_index] = max(scores);
        max_scores(repeats_iterator) = max_value;
        max_populations(repeats_iterator,:) = population(max_value_index,:);
    end

    max_scores_matrix(iterator, :) = max_scores;
    max_populations_3d(:,:,iterator) = max_populations;
end

end