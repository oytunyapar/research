function [x,Fval,exit_flag,output] = findMinTermKGeneticAlgortihm(boolean_function, dimension)
number_of_variables = 2^dimension;

if(boolean_function >= 2^number_of_variables)
    fprintf('FUNCTION IS BIGGER THAN MAXIMUM\n');
    return;
end

global q_matrix;

function_vector = diag(ix2prob(boolean_function,number_of_variables));
d_matrix = monsetup(dimension);
q_matrix=d_matrix*function_vector;

lower_bounds(1,1:number_of_variables) = 0.5;
upper_bounds(1,1:number_of_variables) = Inf;

FitnessFunction = @minTermBFFitnessFunction;

[x,Fval,exit_flag,output] = ...
    ga(FitnessFunction,number_of_variables,[],[],[],[],lower_bounds,upper_bounds);
end


function fitness = minTermBFFitnessFunction(k_vector)
global q_matrix;
coeffcients = q_matrix*k_vector';
number_of_zeros = sum(~coeffcients);
fitness = number_of_zeros^2;
end