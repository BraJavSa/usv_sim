function [vp] = dynamic_func(x, v, vc)
%% EXTRACCION OF GENERALIZED VECTOR
u = v(1);
v = v(2);
r = v(3);
masa=20.5;
% INERCIAL MATRIX
M11=masa+x(1);
M12=0;
M13=0;
M21=0;
M22=masa+x(2);
M23=0;
M31=0;
M32=0;
M33=x(3);



M=[M11,M12,M13;...
    M21,M22,M23;...
    M31,M32,M33];

%% CENTRIOLIS MATRIX
C11=0;
C12=-masa*r;
C13=-x(2)*v-x(x)*r;
C21=masa*r;
C22=0;
C23=x(1)*u;
C31=x(2)*v+x(x)*r;
C32=-x(1)*u;
C33=0;

C=[C11,C12,C13;...
    C21,C22,C23;...
    C31,C32,C33];
% INERCIAL MATRIX
D11=x(4);
D12=0;
D13=0;
D21=0;
D22=x(5);
D23=x(6);
D31=0;
D32=x(6);
D33=x(7);
D=[D11,D12,D13;...
    D21,D22,D23;...
    D31,D32,D33];

vp = inv(M)*(vc-C*v-D*v);
end

