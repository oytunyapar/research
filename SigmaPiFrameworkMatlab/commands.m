
dimension = 3;
number_of_variables = 2^dimension;
fs = BFEquivalentFunctions(dimension);
function_vector = diag(ix2prob(fs(2),number_of_variables));
d_matrix = monsetup(dimension);
q_matrix=d_matrix*function_vector;
sum(q_matrix,2)

100 25  36  25  49  25  25  16
81	25	36	16	64	25	16	16
64	36	36	25	49	36	16	16
81	36	36	25	36	25	36	16
64	25	36	16	49	25	16	16
81	36	36	16	64	16	25	16
64	36	25	25	49	25	16	9
81	36	36	36	49	25	25	16
64	25	36	49	49	25	16	9
64	25	36	25	64	25	25	16
64	49	36	25	49	25	25	16

10 7 6 7 8 6 6 4

max_scores_array = [];
max_populations_array = [];

for iterator=1:10
    fprintf("ITERATOR:%d\n",iterator);
    [functions, max_scores, max_populations] = minTermKGeneticAlgortihmRunner(4);
    max_scores_array = [max_scores_array;max_scores];
    max_populations_array = [max_populations_array;max_populations];
end