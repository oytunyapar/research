function [rQ,info]=FM(Q,c)
% eliminate x_i form Qx<0 using Fourier-Motzkin elimination
% Elimination order, i is given in the vector c
info=[];
rQ=Q;
if (isempty(Q)),
    return;
end;
for i=1:length(c),
    [rQ,pc,nc,zc]=FMc(rQ,c(i));
    %fprintf('eliminated x%d info_row1:#pos _row2:#neg _row3:#zero\n',c(i));
    
    %info=[pc; nc; zc] 
    
end;