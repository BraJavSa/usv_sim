clear
clc
t_exp=4;
ts=0.02;
Nd=30;
D=3;
n=0.1;
N=t_exp/ts+1;
k=0:ts:t_exp;
x=sin(k+sin(k.^2));
%x=sin(k.^2);
y=circshift(x, -D);
y=y(1:(N-D));
%plot(k(1:N-D),x(1:N-D))

xd=zeros(1,Nd);

X=zeros(N-D,Nd);
for i=1:(N-D)
    X(i,:)=xd;
    xd=circshift(xd, 1);
    xd(1)=x(i);
end
Z=-X(1:(N-D),:);

it=100;
w=(pinv(X)*y')';
%w=(1/Nd)*ones(it,Nd);

%I=eye(Nd,Nd);
%E=zeros(it,1);
%for i=1:(it-1)
%    v=w(i,:)*X';
%    e=(y-v)';
%    w(i+1,:)=w(i,:)-((Z'*Z+n*I)\Z'*e)';
%    E(i)=(1/length(e))*(e')*e;
%end
%figure(1)
%plot(0:(it-1),w(1:it,:),'LineWidth',1);
%xlabel('Iteraciones');
%ylabel('w_i');
%xlim([0 it-1])
%title('Evolución de los pesos de la red')

figure(2);hold on
%w=w(i+1,:);
v=w*X';
e=(y-v)';
%E(it)=(1/length(e))*(e')*e;
plot(k(1:(N-D)),y,'Color',[0.7 0.7 0.7],'LineWidth',2)
plot(k(1:(N-D)),v,'r--','LineWidth',2)
xlabel('Tiempo [s]');
xlim([0 (N-D)*ts-1])
legend('x(k+3)','x(k+3) estimado')
title('Comparación entre la predicción de 3 pasos exacta y la estimada')

figure(3);hold on
plot(k(1:(N-D)),x(1:(N-D)),'k','LineWidth',2)
plot(k(1:(N-D)),v,'r--','LineWidth',2)
xlabel('Tiempo [s]')
title('Señal original y su predicción estimada con 3 pasos')
legend('x(k)','x(k+3) estimado')

%figure(4)
%plot(0:(it-1),E,'k','LineWidth',2)
%plot(2:it,E(2:length(E)),'k','LineWidth',2)
%xlim([2 it]);
%ylim([0 E(2)]);
%xlabel('Iteraciones')
%ylabel('MSE');
%title('Evolución del error cuadrático medio')