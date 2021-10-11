function [indexes,size] = binaryClassification(number,bit_amount,ones)
    
if (number > 2^bit_amount - 1 || number < 1)
    fprintf('Number is out of limits. Upper limit:%d Lower limit:%d\n',2^bit_amount - 1,1);
    return;
end

indexes_one = zeros(1,bit_amount);
indexes_zero = zeros(1,bit_amount);
last_index_one = 0;
last_index_zero = 0;

for ITERATOR = (bit_amount - 1):-1:0
    if(number >= 2^ITERATOR)
        last_index_one = last_index_one + 1;
        indexes_one(last_index_one) = ITERATOR;
        number = number - 2^ITERATOR;
    else
        last_index_zero = last_index_zero + 1;
        indexes_zero(last_index_zero) = ITERATOR;
    end
end

if(ones <= 0)
    indexes = indexes_zero(1:last_index_zero);
    size = last_index_zero;
else
    indexes = indexes_one(1:last_index_one);
    size = last_index_one;
end
    
end

