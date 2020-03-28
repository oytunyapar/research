function findZeroables_v9_parallel(problem_class)
% problem_class is a hexadecimal number

global fileID_ARRAY;
global number_of_variables;

number_of_variables = 4 * size(problem_class,2);
dimension = log2(number_of_variables);

if(number_of_variables > 0)
    if(floor(dimension) ~= dimension)
        fprintf('YOUR CLASS INPUT IS NOT POWER OF TWO\n');
        return;
    end
else
    fprintf('YOUR CLASS INPUT IS EMPTY STRING\n');
    return;
end

folder_creation_success = 0;
for ITERATOR=1:10
    folder_name_extension = sprintf('_NON_ZEROABILITY_CHECK%d',ITERATOR);
    folder_name = strcat(problem_class,folder_name_extension);
    if (exist(folder_name,'dir') ~= 7)
        if(mkdir(folder_name) == 1)
            folder_creation_success = 1;
            break;
        end
    end
end

if(folder_creation_success == 0)
    fprintf('FOLDER CREATION PROBLEM\n');
    return;
end

cleanupObject = onCleanup(@() cleanUpFunc());

fileID_ARRAY = zeros(number_of_variables);
for ITERATOR=1:number_of_variables
    file_name = sprintf('NON_ZEROABLE%d.txt',ITERATOR);
    file_folder = strcat(folder_name,'/');
    full_file_name = strcat(file_folder,file_name);
    fileID = fopen(full_file_name,'w+');
    if(fileID < 0)
        fprintf('FILE OPEN PROBLEM.\n');
        return;
    else
        fileID_ARRAY(ITERATOR) = fileID;
    end
end
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
        parfeval(parallel_pool,@coefficientEquationsConflicts,1,...
        problem_class,combinations(begin:finish,:),...
        subset_matrix,SUBSET_MATRIX_NUMBER_OF_BITS);
    end
    
    for POOL_ITERATOR = 1:number_of_workers
        [~,new_conflicts] = fetchNext(parfeval_handles);
    
        for ITERATOR1 = 1:size(new_conflicts,1)
            ITERATOR = sum(power(2,double(new_conflicts(ITERATOR1,:)) - 1),2);

            fprintf(fileID_ARRAY(ITERATOR0),'%d ',new_conflicts(ITERATOR1,:));
            fprintf(fileID_ARRAY(ITERATOR0),'\n');
            [subset_neg,size_of_subset_neg] = binaryClassification(ITERATOR,number_of_variables,0);
            ITERATOR_= ITERATOR + sum(pow2(subset_neg(nchoosek(double(1:size_of_subset_neg),1))),1);
            for ITERATOR3 = 1:size(ITERATOR_,2)
                subset_matrix(floor(ITERATOR_(ITERATOR3)/SUBSET_MATRIX_NUMBER_OF_BITS) + 1) = ...
                bitor(subset_matrix(floor(ITERATOR_(ITERATOR3)/SUBSET_MATRIX_NUMBER_OF_BITS) + 1), power(2,mod(ITERATOR_(ITERATOR3),SUBSET_MATRIX_NUMBER_OF_BITS)) );
            end
            for ITERATOR3 = 2:size_of_subset_neg
                neg_combinations = nchoosek(uint8(1:size_of_subset_neg),ITERATOR3);
                neg_combinations_size = nchoosek(size_of_subset_neg,ITERATOR3);
                for ITERATOR4 = 1:neg_combinations_size
                    ITERATOR_ = ITERATOR + sum(power(2,double(subset_neg(neg_combinations(ITERATOR4,:)))),2);
                    subset_matrix(floor(ITERATOR_/SUBSET_MATRIX_NUMBER_OF_BITS) + 1) = ...
                    bitor(subset_matrix(floor(ITERATOR_/SUBSET_MATRIX_NUMBER_OF_BITS) + 1), power(2,mod(ITERATOR_,SUBSET_MATRIX_NUMBER_OF_BITS)) );
                end
            end
        end
    end
    fprintf("ITERATION: %d\n",ITERATOR0);
end

for ITERATOR=1:number_of_variables
    if(fileID_ARRAY(ITERATOR) > 0)
        fclose(fileID_ARRAY(ITERATOR));
        fileID_ARRAY(ITERATOR) = 0;
    end
end
end

function conflicts = coefficientEquationsConflicts(problem_class, combinations, subset_matrix,subset_matrix_resolution)    
    number_of_variables = 4 * size(problem_class,2);
    dimension = log2(number_of_variables);
    PROBLEM = diag(ix2prob_v2(problem_class,number_of_variables));
    INPUT_MATRIX = monsetup(dimension);
    SOLUTION_MATRIX=INPUT_MATRIX*PROBLEM;
    [number_of_combinations, number_of_equations] = size(combinations);
    
    options = optimset('linprog');
    options.Display = 'off';
    options.Algorithm = 'interior-point';
    lower_bounds(1,1:number_of_variables) = 0.5;
    upper_bounds(1,1:number_of_variables) = Inf;
    
    conflicts= [];

    for ITERATOR1 = 1:number_of_combinations
        ITERATOR = sum(power(2,double(combinations(ITERATOR1,:)) - 1),2);
        if(bitand(subset_matrix(floor(ITERATOR/subset_matrix_resolution) + 1) , power(2,mod(ITERATOR,subset_matrix_resolution)) ) == 0)
            query_matrix = SOLUTION_MATRIX(combinations(ITERATOR1,:),:);
            [~,~,exitflag,~] = linprog([],[],[],query_matrix,zeros(1,number_of_equations)',lower_bounds,upper_bounds,[],options);
            if(exitflag ~= 1)
                conflicts = [conflicts;combinations(ITERATOR1,:)];
            end
        end
    end
end

function cleanUpFunc
    global fileID_ARRAY;
    global number_of_variables;
    
    for ITERATOR=1:number_of_variables
        if(fileID_ARRAY(ITERATOR) > 0)
            fclose(fileID_ARRAY(ITERATOR));
        end
    end
    fprintf('CLEAN UP FUNCTION.\n');
end