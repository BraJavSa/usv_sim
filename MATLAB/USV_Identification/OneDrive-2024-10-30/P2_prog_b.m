clear
clc
t_exp=20;
t_valid=20;
ts=0.01;
Nd=3;
n=0.2;
N=t_exp/ts+1;
k=0:ts:t_exp;
%x=sin(k+sin(k.^2));
x=sin(k)+sin(2*k);
h=0.250156*exp(-2.0025*k)-0.250156*exp(-9.9975*k);
y=conv(x,h)*ts;
y=y(1:N);
%plot(k,x);
%hold on
%plot(k,y);

figure(1)
hold on
plot(k,y)

xd=zeros(1,Nd);
ini_w=[0.1 -0.1];
r=randi([1 2],Nd+1,1);
w=ones(N,Nd+1);
for i=1:(Nd+1)
    w(1,i)=w(1,i)*ini_w(r(i));
end
e=zeros(N,1);
v=zeros(N,1);
for i=1:N
    v(i)=w(i,:)*([x(i) xd]');
    e(i)=y(i)-v(i);
    w(i+1,:)=w(i,:)+n*e(i)*[x(i) xd];
    xd=circshift(xd, 1);
    xd(1)=x(i);
end
plot(k,v);
%%
figure(2)
plot(k,abs(e))

figure(3)
plot(1:(N-Nd),w(1:(N-Nd),:))
%%
w=w(N+1,:);
N=t_valid/ts+1;
k=0:ts:t_valid;
x=sin(k+sin(k.^2));
xd=ones(1,Nd)*0.2;
y=zeros(N,1);
e=zeros(N,1);
for i=1:N
    y(i)=w*(xd');
    e(i)=y(i)-x(j);
    xd=circshift(xd, [0 1]);
    xd(1)=x(j);
end
figure(4)
hold on
plot(k,x)
plot(k,y)

figure(5)
plot(k,abs(e))