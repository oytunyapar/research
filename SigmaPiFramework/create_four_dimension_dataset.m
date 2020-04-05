iterator_begin = 0;
iterator_end = 5000;
dimension = 4;

number_of_variables = 2^dimension;
bent_function_ws = 4*ones(1,number_of_variables);

iterator_final = 2^number_of_variables;

for func = iterator_begin:iterator_final
    if bent_function_ws == abs(sum(diag(ix2prob(func,number_of_variables))*monsetup(dimension)))
        createDataSet(func, dimension);
        fprintf("Function: %d is finished\n",func);
    end
end