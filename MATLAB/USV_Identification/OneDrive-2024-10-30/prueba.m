clc
clear
n=0.01;
L=20;
f=0.1;
fs=400*f;
Ts=1/fs;
t_exp=30/f;
t_valid=t_exp;
N=t_exp/Ts+1;
t=0:Ts:t_exp;
x=sin((2*pi*f)*t)+sin(2*(2*pi*f)*t)+sin(3*(2*pi*f)*t)+sin(4*(2*pi*f)*t);
h=0.250156*exp(-2.0025*t)-0.250156*exp(-9.9975*t);
y=conv(x_t,h)*Ts;
y_d=y(1:N);
plot(t,y_d)
hold on

%ini_w=[1/L -1/L];
%r=randi([1 2],L+1,1);
%w=ones(N,L+1);
%for i=1:(L+1)
%    w(1,i)=w(1,i)*ini_w(r(i));
%end

w=(1/L)*ones(N,L+1);
xd=zeros(1,L);
e=zeros(N,1);
v=zeros(N,1);

for i=1:N
    v(i)=w(i,:)*([x_t(i) xd]');
    e(i)=y_d(i)-v(i);
    w(i+1,:)=w(i,:)+n*e(i)*[x_t(i) xd];
    xd=circshift(xd, 1);
    xd(1)=x_t(i);
end
plot(t,v);
figure(2)
plot(1:N,abs(e))
figure(3)
plot(1:N+1,w)

w(1,:)=w(N+1,:);
figure(4)
N=t_valid/Ts+1;
t=0:Ts:t_valid;
k=1;
TRect=50;
x_e(1)=0;
for i=2:length(t)
    if(k<(TRect/Ts))
        x_e(i)=1;
    else
        x_e(i)=-1;
    end
    if k==2*TRect/Ts
       k=1;
    end
    k=k+1;
end
%x_e=ones(1,length(t));
y=conv(x_e,h)*Ts;
y_d=y(1:N);
plot(t,y_d)
hold on
xd=zeros(1,L);
for i=1:N
    v(i)=w(i,:)*([x_e(i) xd]');
    e(i)=y_d(i)-v(i);
    w(i+1,:)=w(i,:)+n*e(i)*[x_e(i) xd];
    xd=circshift(xd, 1);
    xd(1)=x_e(i);
end
plot(t,v);

x_t=sin((2*pi*f)*t)+sin(2*(2*pi*f)*t)+sin(3*(2*pi*f)*t)+sin(4*(2*pi*f)*t);
h=0.250156*exp(-2.0025*t)-0.250156*exp(-9.9975*t);
y=conv(x_t,h)*Ts;
y_d=y(1:N);
w=w(N+1,:);
xd=zeros(1,L);
for i=1:N
    v(i)=w*([x_t(i) xd]');
    e(i)=y_d(i)-v(i);
    xd=circshift(xd, 1);
    xd(1)=x_t(i);
end 
figure(6)
plot(t,y_d)
hold on
plot(t,v)
