clc
clear

n=0.1;
L=30;
f=0.1;
fs=400*f;
Ts=1/fs;
t_exp=10/f;
N=t_exp/Ts+1;
t=0:Ts:t_exp;
x=sin((2*pi*f)*t)+sin(2*(2*pi*f)*t)+sin(3*(2*pi*f)*t)+sin(4*(2*pi*f)*t)+sin(5*(2*pi*f)*t)+sin(6*(2*pi*f)*t);
h=0.250156*exp(-2.0025*t)-0.250156*exp(-9.9975*t);
y=conv(x,h)*Ts;
y=y(1:N)';

w=(1/L)*ones(N,L+1);
xd=zeros(1,L);

X=zeros(N,L);
for i=1:N
    X(i,:)=xd;
    xd=circshift(xd, 1);
    xd(1)=x(i);
end