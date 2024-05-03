clc
clear
close all

% 1) Tiempo
ts=0.1;
t=0:ts:45;

% 2) Condiciones iniciales
    xi(1)=-3;     
    yi(1)=2;    
    psir(1)=pi;   

     xrd = 10*sin(0.4*t);                           %Posición x
     yrd = 10*sin(0.2*t);                               %Posición y 
     xrdp= 10*cos(0.4*t)*0.4;  
     yrdp= 10*cos(0.2*t)*0.2;

a=0.2;

xr(1)=xi(1)+a*cos(psir(1));
yr(1)=yi(1)+a*sin(psir(1));
kx=3;
ky=2;
ft(1)=0;
ft(1)=0;
w_obs(1)=0;
w_obs(2)=0;
for k=1:length(t)

% 4) Control

        %a) Errores de control
        xre(k) = xrd(k) - xr(k);
        yre(k) = yrd(k) - yr(k);

        
    
        %d) Ley de control
        angle=calculate_angle(xr(k), yr(k), psir(k));
        fuerza=cal_fuerza(xr(k),yr(k));
        ft(k)=fuerza*cos(angle);
        fr(k)=fuerza*sin(angle);
        if k>2
            w_obs(k)=impedancia(ft(k),fr(k),w_obs(k-1),w_obs(k-2),ts);
        end
        distancia = sqrt((0 - xr(k))^2 + (9.8 - yr(k))^2);
        if distancia>5
            w_obs(k)=0;
        end
        vx=xrdp(k)+kx*(xre(k));
        vy=yrdp(k)+ky*(yre(k));
        vpsi=-(vx/a)*sin(psir(k))+(vy/a)*cos(psir(k));
        u(k)=vx*cos(psir(k))+vy*sin(psir(k));
        w(k)=vpsi+w_obs(k);


 % 5) Aplicar acciones de control al robot
 
    xrp(k)=u(k)*cos(psir(k))-a*w(k)*sin(psir(k));
    yrp(k)=u(k)*sin(psir(k))+a*w(k)*cos(psir(k));
 
 % Hallar posiciones
    xr(k+1)=xr(k)+ts*xrp(k);
    yr(k+1)=yr(k)+ts*yrp(k);
    psir(k+1)=psir(k)+ts*w(k);
 
    xc(k+1)=xr(k+1)-a*cos(psir(k+1));
    yc(k+1)=yr(k+1)-a*sin(psir(k+1));
    
end


%% Simulacion

pasos=10;  fig=figure('Name','Simulacion');


set(fig,'position',[60 60 980 600]);
axis square; cameratoolbar
axis([-13 13 -13 13 0 1]);
grid on
MobileRobot;
M1=MobilePlot(xr(1),yr(1),psir(1));hold on
M2=plot(xr(1),yr(1),'b','LineWidth',2);
plot(xrd,yrd,'r','LineWidth',2);

for i=1:pasos:length(t)
    
    delete (M1)
    delete (M2)
    M1=MobilePlot(xc(i),yc(i),psir(i)); hold on
    M2=plot(xr(1:i),yr(1:i),'b','LineWidth',2);
    
   pause(ts)
    
end


%% Graficas
% figure('Name','Errores')
% subplot(211)
% plot(t,xre,'linewidth',2), grid on
% legend('Error en x')
% xlabel('Tiempo'), ylabel('Error  [m]')
% title("ERRORES")
% subplot(212)
% plot(t,yre,'g','linewidth',2), grid on
% legend('Error en y')
% xlabel('Tiempo'), ylabel('Error  [m]')
% 
% figure('Name','Acciones de control')
% subplot(211)
% plot(t,u,'linewidth',2), grid on
% legend('Velocidad lineal u')
% xlabel('Tiempo'), ylabel('Velocidad [m/s]')
% title("ACCIONES DE CONTROL")
% 
% subplot(212)
% plot(t,w,'g','linewidth',2), grid on
% legend('Velocidad angular w')
% xlabel('Tiempo'), ylabel('Velocidad  [rad/s]')


function angle = calculate_angle(Xr, Yr, robot_angle)

% Calculate the difference in x and y coordinates between the robot and the object
    dx = 0 - Xr;
    dy = 9.8 - Yr;
    
    % Calculate the angle between the robot and the object
    object_angle = atan2(dy, dx);
    
    % Convert the angle to a range from 0 to 2*pi
    if object_angle < 0
        object_angle = object_angle + 2*pi;
    end
    
    % Calculate the difference between the robot angle and the angle to the object
    angle_difference = object_angle - robot_angle;
    
    % Convert the angle difference to a range from -pi to pi
    if angle_difference > pi
        angle_difference = angle_difference - 2*pi;
    elseif angle_difference < -pi
        angle_difference = angle_difference + 2*pi;
    end
    
    % Return the angle difference
    angle = angle_difference;
end

function fuerza = cal_fuerza(xr,yr)
    % Definición de los parámetros
    d_min = 0.5;
    d_max = 5;
    f_max = 18;
    n=3;
    % Calcular el valor de 'b'
    b = f_max / (d_max - d_min)^n;
    distancia = sqrt((0 - xr)^2 + (9.8 - yr)^2);
    % Calcular el valor de la función para el tiempo 't'
    fuerza = f_max - b * (distancia - d_min);
end

function w_obs=impedancia(ft,fr,w_obs_a,w_obs_aa,ts)
    B = 0.6;
    K = 4;
    I = 3;
    w_obs = (2*I*w_obs_a - I*w_obs_aa + B*w_obs_a - ft*sign(fr)) / (I*(ts^2) + B*ts + K);

end

