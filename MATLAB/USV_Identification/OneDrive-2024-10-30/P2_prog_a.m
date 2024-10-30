clear
clc
t_exp=2;
ts=0.001;
Nd=5;
D=50;
n=0.1;
N=t_exp/ts+1;
k=0:ts:t_exp;
x=sin(k+sin(k.^2));
%x=sin(k.^2);

figure(1)
hold on
plot(k(1:N-D),x(1:N-D))

xd=zeros(1,Nd);

w=(1/Nd)*ones(N,Nd);

e=zeros(N,1);
v=zeros(N,1);
for i=1:(N-D)
    v(i)=w(i,:)*(xd');
    e(i)=x(i+D)-v(i);
    w(i+1,:)=w(i,:)+n*e(i)*xd;
    xd=circshift(xd, 1);
    xd(1)=x(i);
end
plot(k(1:N-D),v(1:N-D));

figure(2)
plot(k,abs(e))

figure(3)
plot(1:(N-D),w(1:(N-D),:))
%%
w=w((N-D),:);
N=t_exp/(ts/5)+1;
k=0:(ts/5):t_exp;
x=sin(k+sin(k.^2));
%x=sin(k.^2);
xd=zeros(1,Nd);
y=zeros(N-D,1);
e=zeros(N-D,1);
for i=1:(N-D)
    y(i)=w*(xd');
    e(i)=x(i+D)-y(i);
    xd=circshift(xd, [0 1]);
    xd(1)=x(i);
end
figure(4)
hold on
plot(k(1:(N-D)),x(1:(N-D)))
plot(k(1:(N-D)),y)

figure(5)
plot(k(1:(N-D)),abs(e(1:(N-D))))