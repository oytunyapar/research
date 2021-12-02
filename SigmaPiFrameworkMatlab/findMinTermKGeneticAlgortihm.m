function [x,Fval,exit_flag,output,population,scores] = ...
    findMinTermKGeneticAlgortihm(boolean_function, dimension)
number_of_variables = 2^dimension;

if(boolean_function >= 2^number_of_variables)
    fprintf('FUNCTION IS BIGGER THAN MAXIMUM\n');
    return;
end

function_vector = diag(ix2prob(boolean_function,number_of_variables));
d_matrix = monsetup(dimension);
q_matrix=d_matrix*function_vector;

lower_bounds(1,1:number_of_variables) = 1;
upper_bounds(1,1:number_of_variables) = number_of_variables/2;

intcon=(1:number_of_variables);
FitnessFunction = @(k_vector)minTermBFFitnessFunction(k_vector,q_matrix);

options = optimoptions('ga','PopulationSize',numberOfPopulation(dimension),...
    'MaxGenerations',numberOfIterations(dimension),'UseParallel',true);

[x,Fval,exit_flag,output,population,scores] = ...
    ga(FitnessFunction,number_of_variables,[],[],[],[],lower_bounds,upper_bounds,[],intcon,options);
end


function fitness = minTermBFFitnessFunction(k_vector, q_matrix)

coeffcients = q_matrix*k_vector';
number_of_zeros = sum(~coeffcients);
fitness = number_of_zeros^2;

end

function number_of_iterations = numberOfIterations(dimension)

if dimension == 3
    number_of_iterations = 200000;
elseif dimension == 4
    number_of_iterations = 1000000;
elseif dimension == 5
    number_of_iterations = 2000000;
end

end

function number_of_population= numberOfPopulation(dimension)

population_constant = 2000;
number_of_population = population_constant * 2^dimension;

end
