function [y,kk]=ix2prob(ix,bitlen)
%% given the problem id returns the output spec.
%% use as ix2prob(id, 2^problem_dimension);
%% The standard form of a given problem with index=id can be obtained like this
%% Q=diag(ix2prob(id, 2^problem_dimension))*monsetup(dim);
%% Find a such that Qa>0  (of course a=Q') is a solution but we want a with
%% a lot of zeros in it.
   y=zeros(bitlen,1);
   kk=0;
   
   if isa(ix,'int64')
       divisionFunction = @(dividend, divisior)int64Divide(dividend, divisior);
   elseif isa(ix,'double')
       divisionFunction = @(dividend, divisior)doubleDivide(dividend, divisior);
   end

 for h=1:bitlen   % usually = 2^dim
     y(h)=2*mod(ix,2)-1;
     ix = divisionFunction(ix,2);
     kk=kk+(y(h)+1)/2;
 end % h
 
 kk = bitlen - kk;
 y = -y;


function result = int64Divide(dividend, divisior)
    result = idivide(int64(dividend),int64(divisior),'floor');

function result = doubleDivide(dividend, divisior)
    result = floor(dividend/divisior);

