%%USE this to print the readable form of Equiv. Class representative functions.
function s=coef2str(spec)

i=1;
M{i}='1'; i=i+1;
M{i}='x1';i=i+1;
M{i}='x2';i=i+1;
M{i}='x2*x1';i=i+1;
M{i}='x3';i=i+1;
M{i}='x3*x1';i=i+1;
M{i}='x3*x2';i=i+1;
M{i}='x3*x2*x1';i=i+1;
M{i}='x4'; i=i+1;
M{i}='x4*x1';i=i+1;
M{i}='x4*x2';i=i+1;
M{i}='x4*x2*x1';i=i+1;
M{i}='x4*x3';i=i+1;
M{i}='x4*x3*x1';i=i+1;
M{i}='x4*x3*x2';i=i+1;
M{i}='x4*x3*x2*x1';i=i+1;

if (length(spec)>2^4)
    fprintf('Works for dim < 5, (need to change code for larger)\n');
    return;
end


first = 1;
for i=1:length(spec)
    if spec(i)~=0
       if first==1
           first=0;
           s=sprintf('%2.2f(%s) ',spec(i),M{i});
       else
          s=sprintf('%s + %2.2f(%s) ',s,spec(i),M{i});
       end
    end
end
