clc
clear

n=0.01;
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
y=y(1:N);
xd=zeros(1,L);

X=zeros(N,L);
for i=1:N
    X(i,:)=xd;
    xd=circshift(xd, 1);
    xd(1)=x(i);
end

Z=-X;

it=100;

w=10*(1/L)*ones(it,L);
I=eye(L,L);
E=zeros(it,1);

for j=1:(it-1)
    v=w(j,:)*X';
    e=(y-v)';
    delta_w = -((Z'*Z+n*I)\Z'*e)';
    w(j+1,:)=w(j,:)+delta_w;
    E(j)=(1/length(e))*(e')*e;
    %if (E(j)<0.001)
     %   break
    %end
end

w2=10*(1/L)*ones(it,L);
I=eye(L,L);
E2=zeros(it,1);

for j=1:(it-1)
    v=w2(j,:)*X';
    e=(y-v)';
    delta_w = -((Z'*Z)\Z'*e)';
    w2(j+1,:)=w2(j,:)+delta_w;
    E2(j)=(1/length(e))*(e')*e;
    %if (E(j)<0.001)
     %   break
    %end
end
figure(1);hold on
plot(0:j,w2(1:(j+1),:),'k','LineWidth',2);
plot(0:j,w(1:(j+1),:),'r','LineWidth',1);
xlabel('Iteraciones');
ylabel('w_i');
xlim([0 j])
title('Evolución de los pesos de la red')


w=w(j+1,:);
v=w*X';
e=(y-v)';
E(j+1)=(1/length(e))*(e')*e;
E2(j+1)=E2(j);

figure(2);hold on
plot(0:j,E2(1:(j+1)),'k','LineWidth',1)
plot(0:j,E(1:(j+1)),'r','LineWidth',1.5)
%plot(2:it,E(2:length(E)),'k','LineWidth',2)
xlim([1 6]);
xlabel('Iteraciones')
ylabel('MSE');
legend('Método de Gauss-Newton','Método de Levenberg-Marquardt')
title('Evolución del error cuadrático medio')

figure(3);hold on
plot(t,y,'Color',[0.7 0.7 0.7],'LineWidth',2)
plot(t,v,'r--','LineWidth',2)
xlabel('Tiempo [s]');
ylabel('Velocidad [rad/s]');
xlim([0 N*Ts-1])
legend('y(t)','y(t) estimado')
title('Comparación entre la salida del sistema exacta y la estimada (entrenamiento)')

figure(4)
plot(t,x,'k','LineWidth',2)
xlabel('Tiempo [s]');
ylabel('Tensión de Armadura [V]');
title('Entrada del sistema para entrenamiento')

x=sin((2*pi*f)*t)+cos(2*(2*pi*f)*t)+sin(3*(2*pi*f)*t)+cos(4*(2*pi*f)*t);

y=conv(x,h)*Ts;
y=y(1:N);

xd=zeros(1,L);
X=zeros(N,L);
for i=1:N
    X(i,:)=xd;
    xd=circshift(xd, 1);
    xd(1)=x(i);
end

v=w*X';
e=(y-v)';
E=(1/length(e))*(e')*e;

figure(5);hold on
plot(t,y,'Color',[0.7 0.7 0.7],'LineWidth',2)
plot(t,v,'r--','LineWidth',2)
xlabel('Tiempo [s]');
ylabel('Velocidad [rad/s]');
xlim([0 N*Ts-1])
legend('y(t)','y(t) estimado')
title('Comparación entre la salida del sistema exacta y la estimada (validación)')

figure(6)
plot(t,x,'k','LineWidth',2)
xlabel('Tiempo [s]');
ylabel('Tensión de Armadura [V]');
title('Entrada del sistema para validación')

k=1;
TRect=20;
for i=1:length(t)
    if(k<(TRect/Ts))
        x(i)=1;
    else
        x(i)=-1;
    end
    if k==2*TRect/Ts
       k=1;
    end
    k=k+1;
end
y=conv(x,h)*Ts;
y=y(1:N);

xd=zeros(1,L);
X=zeros(N,L);
for i=1:N
    X(i,:)=xd;
    xd=circshift(xd, 1);
    xd(1)=x(i);
end

v=w*X';
e=(y-v)';
E=(1/length(e))*(e')*e;

figure(7);hold on
plot(t,y,'Color',[0.7 0.7 0.7],'LineWidth',2)
plot(t,v,'r--','LineWidth',2)
xlabel('Tiempo [s]');
ylabel('Velocidad [rad/s]');
xlim([0 N*Ts-1])
legend('y(t)','y(t) estimado')
title('Comparación entre la salida del sistema exacta y la estimada (validación)')

figure(8)
plot(t,x,'k','LineWidth',2)
xlabel('Tiempo [s]');
ylabel('Tensión de Armadura [V]');
title('Entrada del sistema para validación')

