function [ noncompliant_sets,zeroable_sets ] = three_equivalent_check()

main_set = [  1  1  1;
              1  1 -1;
              1 -1  1;
              1 -1 -1;
             -1  1  1;
             -1  1 -1;
             -1 -1  1;
             -1 -1 -1 ];
         
noncompliant_sets = [];
zeroable_sets = [];
         
set_size = size(main_set,1);
total_number_of_patterns = 2^set_size;

three_vector = [1 2 3];
two_vector_one = [1 2];
two_vector_two = [1 3];
two_vector_three = [2 3];

for ITERATOR1 = 1:total_number_of_patterns - 1
    
    sets = dec2bin(ITERATOR1,set_size);
    sets_vector = find(sets == '1');
    
    subset = [];
    for ITERATOR2 = 1:size(sets_vector,2)
        subset = [ subset;
                   main_set(sets_vector(ITERATOR2),:) ];
    end
    
    eliminated_matrix = FM(subset,three_vector);
    
    if(isempty(find(eliminated_matrix > 0 , 1)) == 0 || isempty(find(eliminated_matrix < 0 , 1)) == 0 )
        eliminated_matrix = FM(subset,two_vector_one);
        reduced_eliminated_matrix = [eliminated_matrix(:,two_vector_one(1)) eliminated_matrix(:,two_vector_one(2))];
        if(isempty(find(reduced_eliminated_matrix > 0 , 1)) == 1 && isempty(find(reduced_eliminated_matrix < 0 , 1)) == 1 )
            eliminated_matrix = FM(subset,two_vector_two);
            reduced_eliminated_matrix = [eliminated_matrix(:,two_vector_two(1)) eliminated_matrix(:,two_vector_two(2))];
            if(isempty(find(reduced_eliminated_matrix > 0 , 1)) == 1 && isempty(find(reduced_eliminated_matrix < 0 , 1)) == 1 )
                eliminated_matrix = FM(subset,two_vector_three);
                reduced_eliminated_matrix = [eliminated_matrix(:,two_vector_three(1)) eliminated_matrix(:,two_vector_three(2))];
                if(isempty(find(reduced_eliminated_matrix > 0 , 1)) == 1 && isempty(find(reduced_eliminated_matrix < 0 , 1)) == 1 )
                    noncompliant_sets = [ noncompliant_sets;ITERATOR1 ];
                end
            end
        end
    else
        zeroable_sets = [zeroable_sets;ITERATOR1];        
    end
end

if(isempty(noncompliant_sets) == 1)
    fprintf('THREE EQUIVALENT IS VERIFIED.\n');
end

end

