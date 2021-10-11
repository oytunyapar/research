function not_containing_sets = isSubsetMatrix( matrix_1 , matrix_2 )

not_containing_sets = [];

if(size(matrix_1,2) > size(matrix_2,2))
    first_matrix = matrix_1;
    second_matrix = matrix_2;
else
    first_matrix = matrix_2;
    second_matrix = matrix_1;
end

for ITERATOR_1 = 1:size(first_matrix,1)
    subset_exist = 0;
    for ITERATOR_2 = 1:size(second_matrix,1)
        if(isSubset(first_matrix(ITERATOR_1,:),second_matrix(ITERATOR_2,:)) == 1)
            subset_exist = 1;
            break;
        end
    end
    if(subset_exist == 0)
       not_containing_sets = [not_containing_sets;first_matrix(ITERATOR_1,:)];  
    end
end
end

