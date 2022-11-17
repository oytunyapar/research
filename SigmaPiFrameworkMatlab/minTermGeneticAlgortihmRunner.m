function [max_scores_matrix, max_populations_3d] = minTermGeneticAlgortihmRunner(functions, dimension, population_type, repeats)

number_of_variables = 2^dimension;

if dimension > 5
    length_functions = size(functions,1);
else
    length_functions = length(functions);
end

max_scores_matrix = zeros(length_functions, repeats);
max_populations_3d = zeros(repeats, number_of_variables, length_functions);

global intermadiate_max_scores;
global intermadiate_max_population;

for iterator = 1:length_functions
    max_scores = zeros(1,repeats);
    max_populations = zeros(repeats,number_of_variables);
    for repeats_iterator = 1:repeats
        fprintf("Dimension:%d Function:%d/%d Repeats:%d/%d\n",dimension,iterator,length_functions, ...
            repeats_iterator,repeats);
        
        if dimension > 5
            func = functions(iterator,:);
        else
            func = functions(iterator);
        end

        output_function = @(options, state, flag)ga_examine_intermediate_population...
        (options, state, flag);

        %[~,~,~,~,population,scores]
        [~,~,~,~,~,~] = ...
            findMinTermGeneticAlgortihm(func, dimension, population_type, output_function);
        
        [max_value, max_value_index] = max(intermadiate_max_scores);
        
        max_scores(repeats_iterator) = max_value;
        max_populations(repeats_iterator,:) = intermadiate_max_population(max_value_index,:);
    end

    max_scores_matrix(iterator, :) = max_scores;
    max_populations_3d(:,:,iterator) = max_populations;
end

end

function [state, options, optchanged] =... 
    ga_examine_intermediate_population(options, state, flag)
optchanged = false;

global intermadiate_max_scores;
global intermadiate_max_population;

[max_value, max_value_index] = max(state.Score);
intermadiate_max_scores = [intermadiate_max_scores, max_value];
intermadiate_max_population = [intermadiate_max_population; state.Population(max_value_index,:)];

end