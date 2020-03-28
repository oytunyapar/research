function all_non_conflicts = findZeroables(problem_class, dimension)
% problem_class is a hexadecimal number

global number_of_variables;

number_of_variables = 2^dimension;

if(problem_class >= 2^number_of_variables)
    fprintf('YOUR CLASS INPUT IS NOT POWER OF TWO\n');
    return;
end

cleanupObject = onCleanup(@() cleanUpFunc());

%W_SPECTRUM = sum(SOLUTION_MATRIX,2);

SUBSET_MATRIX_NUMBER_OF_BITS = 32;
array_size = 2^number_of_variables / SUBSET_MATRIX_NUMBER_OF_BITS;

if (SUBSET_MATRIX_NUMBER_OF_BITS == 8)
    type = 'uint8';
elseif (SUBSET_MATRIX_NUMBER_OF_BITS == 16)
    type = 'uint16';
elseif (SUBSET_MATRIX_NUMBER_OF_BITS == 32)
    type = 'uint32';
else
    fprintf('NUMBER OF BITS IS INAPPROPRIATE.\n');
    return;
end

parallel_pool = gcp();
number_of_workers = parallel_pool.NumWorkers;
parfeval_handles(number_of_workers,1) = parallel.FevalFuture;

subset_matrix = zeros(array_size,1,type);

all_non_conflicts = [];

for ITERATOR0 = 1:number_of_variables
    combinations = nchoosek(uint8(1:number_of_variables),ITERATOR0);
    combinations_size = nchoosek(number_of_variables,ITERATOR0);
    %new_conflicts = coefficientEquationsConflicts(problem_class,combinations,subset_matrix,SUBSET_MATRIX_NUMBER_OF_BITS);
    for POOL_ITERATOR = 1:number_of_workers
        factor = uint32(combinations_size/number_of_workers);
        begin = (POOL_ITERATOR - 1)*factor + 1;
        finish = POOL_ITERATOR*factor;
        if(finish > combinations_size)
            finish = combinations_size;
        end
        parfeval_handles(POOL_ITERATOR) = ...
        parfeval(parallel_pool,@coefficientEquationsConflicts,3,...
        problem_class, dimension, combinations(begin:finish,:),...
        subset_matrix, SUBSET_MATRIX_NUMBER_OF_BITS);
    end
    
    for POOL_ITERATOR = 1:number_of_workers
        [~,new_subset_matrix,~,new_non_conflicts] = fetchNext(parfeval_handles);
        subset_matrix = bitor(subset_matrix,new_subset_matrix);
        all_non_conflicts = [all_non_conflicts;sum(power(2,double(new_non_conflicts - 1)),2)];
    end
    
    fprintf("ITERATION: %d\n",ITERATOR0);
end

end

function [subset_matrix,conflicts,non_conflicts] = coefficientEquationsConflicts(problem_class, dimension, combinations, subset_matrix,subset_matrix_resolution)    
    number_of_variables = 2^dimension;
    PROBLEM = diag(ix2prob(problem_class,number_of_variables));
    INPUT_MATRIX = monsetup(dimension);
    SOLUTION_MATRIX=INPUT_MATRIX*PROBLEM;
    [number_of_combinations, number_of_equations] = size(combinations);
    
    options = optimset('linprog');
    options.Display = 'off';
    options.Algorithm = 'interior-point';
    lower_bounds(1,1:number_of_variables) = 0.5;
    upper_bounds(1,1:number_of_variables) = Inf;
    
    conflicts = [];
    non_conflicts = [];

    for ITERATOR1 = 1:number_of_combinations
        ITERATOR = sum(power(2,double(combinations(ITERATOR1,:)) - 1),2);
        if(bitand(subset_matrix(floor(ITERATOR/subset_matrix_resolution) + 1) , power(2,mod(ITERATOR,subset_matrix_resolution)) ) == 0)
            query_matrix = SOLUTION_MATRIX(combinations(ITERATOR1,:),:);
            [~,~,exitflag,~] = linprog([],[],[],query_matrix,zeros(1,number_of_equations)',lower_bounds,upper_bounds,[],options);
            if(exitflag ~= 1)
                conflicts = [conflicts;combinations(ITERATOR1,:)];
            else
                non_conflicts = [non_conflicts;combinations(ITERATOR1,:)];
            end
        end
    end
    
    for ITERATOR1 = 1:size(conflicts,1)
        ITERATOR = sum(power(2,double(conflicts(ITERATOR1,:)) - 1),2);
        [subset_neg,size_of_subset_neg] = binaryClassification(ITERATOR,number_of_variables,0);
        ITERATOR_= ITERATOR + sum(pow2(subset_neg(nchoosek(double(1:size_of_subset_neg),1))),1);
        for ITERATOR3 = 1:size(ITERATOR_,2)
            subset_matrix(floor(ITERATOR_(ITERATOR3)/subset_matrix_resolution) + 1) = ...
            bitor(subset_matrix(floor(ITERATOR_(ITERATOR3)/subset_matrix_resolution) + 1), power(2,mod(ITERATOR_(ITERATOR3),subset_matrix_resolution)) );
        end
        for ITERATOR3 = 2:size_of_subset_neg
            neg_combinations = nchoosek(uint8(1:size_of_subset_neg),ITERATOR3);
            neg_combinations_size = nchoosek(size_of_subset_neg,ITERATOR3);
            for ITERATOR4 = 1:neg_combinations_size
                ITERATOR_ = ITERATOR + sum(power(2,double(subset_neg(neg_combinations(ITERATOR4,:)))),2);
                subset_matrix(floor(ITERATOR_/subset_matrix_resolution) + 1) = ...
                bitor(subset_matrix(floor(ITERATOR_/subset_matrix_resolution) + 1), power(2,mod(ITERATOR_,subset_matrix_resolution)) );
            end
        end
    end
end

function cleanUpFunc
    fprintf('CLEAN UP FUNCTION.\n');
end