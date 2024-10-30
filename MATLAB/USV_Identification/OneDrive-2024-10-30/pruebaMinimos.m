clear
clc
t_exp=10;
ts=0.001;
Nd=30;
D=100;
n=0.1;
N=t_exp/ts+1;
k=0:ts:t_exp;
x=sin(k+sin(k.^2));
%x=sin(k.^2);
y=circshift(x, -D);
y=y(1:(N-D))';
%plot(k(1:N-D),x(1:N-D))

xd=zeros(1,Nd);

X=zeros(N-D,Nd);
for i=1:(N-D)
    X(i,:)=xd;
    xd=circshift(xd, 1);
    xd(1)=x(i);
end

w=pinv(X)*y;
v=w'*X';
figure(1);hold on
plot(k(1:(N-D)),v,'--')
plot(k(1:(N-D)),y)
%plot(k(1:(N-D)),x(1:(N-D)))