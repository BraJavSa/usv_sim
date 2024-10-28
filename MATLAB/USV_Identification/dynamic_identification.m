function [tau] = dynamic_identification(x, vel_p, vel)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

%% EXTRACCION OF GENERALIZED VECTOR
u = vel(1);
v = vel(2);
r = vel(3);
masa=20.5;
% INERCIAL MATRIX
M11=x(1);
M12=0;
M13=0;
M21=0;
M22=x(2);
M23=x(3);
M31=0;
M32=x(3);
M33=x(4);



M=[M11,M12,M13;...
    M21,M22,M23;...
    M31,M32,M33];

%% CENTRIOLIS MATRIX
C11=0;
C12=-x(5)*r;
C13=-x(6)*v-x(3)*r;
C21=x(5)*r;
C22=0;
C23=x(7)*u;
C31=x(5)*v+x(3)*r;
C32=-x(7)*u;
C33=0;

C=[C11,C12,C13;...
    C21,C22,C23;...
    C31,C32,C33];
% INERCIAL MATRIX
D11=x(8);
D12=0;
D13=0;
D21=0;
D22=x(9);
D23=x(10);
D31=0;
D32=x(10);
D33=x(11);
D=[D11,D12,D13;...
    D21,D22,D23;...
    D31,D32,D33];

tau = M*vel_p + C*vel + D*vel;
end

