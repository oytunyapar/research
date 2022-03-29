function [y]=hexix2prob(hexstr,dim)
% given the problem id in hex string it returns the output spec.

y=zeros(2^dim,1);
kk=0;
digpos = 0;
for d = length(hexstr):-1:1
    ix = hex2dec(hexstr(d)); % a value 0..15
    for h=1:4
        y(digpos+h)=1-2*mod(ix,2);
        ix=floor(ix/2);
    end % h
    digpos = digpos + 4;
end