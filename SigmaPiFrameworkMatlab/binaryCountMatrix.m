function bin_boolean_matrix = binaryCountMatrix(dimension)

ASCII_ZERO_BASE = 48;
bin_boolean_matrix = zeros(2^dimension,dimension);

for ITERATOR = 0:2^dimension - 1
    temp_line = dec2bin(ITERATOR,dimension) - ASCII_ZERO_BASE;
    for ITERATOR2 = 1:dimension
        if(temp_line(ITERATOR2) == 0)
            temp_line(ITERATOR2) = 1;
        else
            temp_line(ITERATOR2) = -1;
        end
    end
    bin_boolean_matrix(ITERATOR + 1,:) = temp_line;
end

end