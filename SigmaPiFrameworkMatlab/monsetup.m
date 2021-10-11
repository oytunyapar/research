function [D, Dinv]=monsetup(dim)

% Given the dimension it constructs the high order expansion
%  The polynomial coefficents, a, of the exact interpolation can be found with a=D^-1*Y
%  But the sign freedom allows us to write a=Dinv*Y*K for K>0
for i=0:2^dim-1,
    n=i;
    
    rr=9*10^(dim+1);
    for k=1:dim,
        if mod(n,2)==0,
            x(k)=1;
            
        else
            x(k)=-1;
            rr = rr+10^(k-1);
        end;
        n=floor(n/2);
    end;
    

    
    for j=0:2^dim-1,
        n=j;
       v=1;
        for k=1:dim,
        if mod(n,2)==1,
            v = v*x(k);
        end;
        n=floor(n/2);
        end; %k
       % fprintf('  %d',v);
      
        D(i+1,j+1)=v;
    end;
    %fprintf('\n');
end;
Dinv = 2^-dim*D;
    