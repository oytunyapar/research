function [rQ,pc,nc,zc]=FMc(Q,c)
% eliminate x_c form Qx<0 using Fourier-Motzkin elimination
[m,n]=size(Q);
col=Q(:,c);
pos=find(col>0);
neg=find(col<0);
zer=col==0;
check_red=0;
if (length(pos)*length(neg)~=0),    
%neg
%pos
rQ=Q(zer,:);
for i=1:length(pos),
   for j=1:length(neg),
   
       ir1=-Q(neg(j),c);
       ir2= Q(pos(i),c);
       %k=gcd(ir1,ir2);
       %k=1;
       %ir1=ir1/k;
       %ir2=ir2/k;
       el=ir1*Q(pos(i),:)+ir2*Q(neg(j),:);
       %size(rQ)
       %el
   if (check_red==1),
       x=pinv(rQ')*el';
       if sum((x<0))==0 & (norm(rQ'*x-el')<1e-10), % in the cone
          fprintf('pos row %d and neg row %d combination is redundant\n',pos(i),neg(j));
          %x
          continue;
      end;
  end;
       %if (el==2*round(el/2)),
       %    el=el/2;
       %end;
       rQ=[rQ;el];
   end;
end;
else 
    rQ=Q;  % cannot reduce
end;  % if poslen*neglen=0;
pnc=[];
pc=[];
nc=[];
zc=[];
for c=1:n,
    col=rQ( :,c);
   pos=(col>0);
   neg=(col<0);
   zer=col==0;
   pc=[pc,sum(pos)];
   nc=[nc,sum(neg)];
   zc=[zc,sum(zer)];
end;
pn=pc-nc;
%size(rQ)
%pc
%nc
