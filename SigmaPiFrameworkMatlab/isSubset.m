function result = isSubset( first_set,second_set )

if(size(first_set,1) > 1 || size(second_set,1) > 1)
    fprintf('THIS FUNCTION ONLY ACCEPTS ROW VECTORS\n');%this could be changed
    return;
end

if(size(first_set,2) == 0 || size(second_set,2) == 0)
    result = 1;
    return;
end

if(size(first_set,2) > size(second_set,2))
    bigger_set=first_set;
    smaller_set=second_set;
else
    bigger_set=second_set;
    smaller_set=first_set;
end

for ITERATOR_1 = 1:size(smaller_set,2)
    found = 0;
    for ITERATOR_2 = 1:size(bigger_set,2)
        if(smaller_set(1,ITERATOR_1) == bigger_set(1,ITERATOR_2))
            found = 1;
        end
    end
    if(found == 0)
        result = 0;
        return;
    end
end

result = 1;
end

