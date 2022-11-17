function [x,Fval,exit_flag,output,population,scores] = ...
    findMinTermGeneticAlgortihm(boolean_function, dimension, population_type, output_function)
number_of_variables = 2^dimension;

if(boolean_function >= 2^number_of_variables)
    fprintf('FUNCTION IS BIGGER THAN MAXIMUM\n');
    return;
end

if dimension > 5
    function_vector = diag(boolean_function);
else
    function_vector = diag(ix2prob(boolean_function,number_of_variables));
end

d_matrix = monsetup(dimension);
q_matrix=d_matrix*function_vector;

fprintf("Function spectrum:\n");
disp(sum(q_matrix,2)')

intcon=(1:number_of_variables);

if population_type == 1
    fprintf("Population type: linear programming\n");
    lower_bound = 0;
    upper_bound = 1;
    lp_lower_bounds(1,1:number_of_variables) = 0.5;
    lp_upper_bounds(1,1:number_of_variables) = Inf;
    lp_options = optimset('linprog');
    lp_options.Display = 'off';
    lp_options.Algorithm = 'interior-point';
    FitnessFunction = @(selected_monomials)minTermLPBFFitnessFunction( ...
        selected_monomials, q_matrix, lp_lower_bounds, lp_upper_bounds, lp_options);
elseif population_type == 2
    fprintf("Population type: k vector");
    lower_bound = 1;
    upper_bound = number_of_variables/2;
    FitnessFunction = @(k_vector)minTermKVectorBFFitnessFunction(k_vector,q_matrix);
else
    fprintf("Undefined population type");
    return
end

lower_bounds(1,1:number_of_variables) = lower_bound;
upper_bounds(1,1:number_of_variables) = upper_bound;

options = optimoptions('ga','PopulationSize',numberOfPopulation(dimension, population_type),...
    'MaxGenerations',numberOfIterations(dimension), 'UseParallel',true, 'OutputFcn',output_function);

[x,Fval,exit_flag,output,population,scores] = ...
    ga(FitnessFunction,number_of_variables,[],[],[],[],lower_bounds,upper_bounds,[],intcon,options);
end


function fitness = minTermKVectorBFFitnessFunction(k_vector, q_matrix)

coeffcients = q_matrix*k_vector';
number_of_zeros = sum(~coeffcients);
fitness = number_of_zeros^2;

end

function fitness = minTermLPBFFitnessFunction(selected_monomials, q_matrix, ...
    lp_lower_bounds, lp_upper_bounds, lp_options)

selected_monomials_indices = find(selected_monomials==1);
[~,~,exitflag,~] = linprog([],[],[],q_matrix(selected_monomials_indices, :), ...
    zeros(1, length(selected_monomials_indices))',lp_lower_bounds,lp_upper_bounds,[],lp_options);

if(exitflag ~= 1)
    fitness = -1;
else
    fitness = length(selected_monomials_indices)^2;
end

end

function number_of_iterations = numberOfIterations(dimension)

if dimension == 3
    number_of_iterations = 10000;
elseif dimension == 4
    number_of_iterations = 20000;
elseif dimension == 5
    number_of_iterations = 80000;
elseif dimension == 6
    number_of_iterations = 500000;
end

end

function number_of_population= numberOfPopulation(dimension, population_type)

if population_type == 1
    population_constant = 300;
elseif population_type == 2
    population_constant = 2000;
end
number_of_population = population_constant * 2^dimension;

end
