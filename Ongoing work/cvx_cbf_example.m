clear all;
close all;

warning('off');

%% Initialization

% Display
figure(1)
hold on

% Parameters
r=0.25;  % goal display radius

% robot dynamics
input_per_robot = 2;

% Robots 
robot = Unicycle2D(1,-40,-30,pi/4,1,2,'nominal');   %ID,x,y,yaw,r_safe,D,status

% Goals
G1=[20.0 20.0]';
robot(1).G = G1;
circle(G1(1),G1(2),r);

% Obstacles
c1=[0 0]';
r1=10;
Obstacle = EnvObject2D( c1(1),c1(2) ,'circle',r1,0);

% Simulation parameters
dt = 0.1;
tf = 150;
T = 0.6;
dt = 0.1;

%CBF and CLF parameters
alpha = 0.5;

for t=0:dt:tf
    
    % Nominal Controller
    
    u_min = [0;-0.25];
    u_max = [5;0.25];
    
    dx = robot.X - robot.G;
    kw = 0.5*u_max(2)/pi;
    phi_des = atan2( -dx(2),-dx(1) );
    delta_phi = wrap_pi( phi_des - robot.yaw );
    
    w0 = kw*delta_phi;
    kv = 0.1;%0.1;
    v0 = kv*norm(dx)*max(0.1,cos(delta_phi)^2);
    u0 = [v0;w0];
       
    cvx_begin quiet
  
       variable u(2,1)
       
       % Objective
       minimize ( norm(u-u0,2) )

       % Constraints
       subject to 
       
%             h = Obstacle.length - sqrt( norm(robot.X-Obstacle.X )^2 - wrap_pi( robot.yaw - atan2(robot.X(2),robot.X(1)) )^2   );  
%             dh_dx = [-2*(robot.X-Obstacle.X)' 0];  
            x1 = robot.X(1); x2 = robot.X(2); yaw = robot.yaw;
            rho = Obstacle.length;
            h = rho - sqrt( norm([x1;x2]-Obstacle.X )^2 - wrap_pi( yaw - atan2(x2,x1) )^2   );  
            dh_dx = [ (-x1 + wrap_pi(yaw-atan2(x2,x1))/(x1^2+x2^2)  )/(rho-h)  ( -x2 - wrap_pi( yaw - atan2( x2,x1 ) )*x1/(x1^2+x2^2)  )/(rho-h) wrap_pi(yaw-atan2(x2,x1))/(rho-h)];
%             dh_dx = [ (-robot.X(1) + wrap_pi(robot.yaw-atan2(robot.X(2),robot.X(1)))/(robot.X(1)^2+robot.X(2)^2)  )/(Obstacle.length-h)  ( -robot.X(2) - wrap_pi( robot.yaw - atan2( x2,x1 ) )*x1/(x1^2+x2^2)  )/(rho-h) wrap_pi(robot.yaw-atan2(x2,x1))/(rho-h)];    
            
            v3g = 0.1319;
            gamma = 1.0;%0.1;
            phi_k = -gamma/T*h - v3g;

            dh_dx*( robot.f + robot.g*u ) <= phi_k;%-alpha*h;
            
    cvx_end
    barrier = dh_dx*( robot.f + robot.g*u ) - phi_k
    
%     kx = 0.6;
%     komega=0.1;
%     ex = robot.G-robot.X
%     etheta = wrap_pi(atan2( robot.G(2)-robot.X(2),robot.G(1)-robot.X(1))-robot.yaw    )
%     u = [kx*ex*cos(etheta);
%          komega*etheta];
%     keyboard
    robot = control_state(robot,u,dt);
    
    pause(0.01);
    
    
    
end


function h = circle(x,y,r)
    th = 0:pi/50:2*pi;
    xunit = r * cos(th) + x;
    yunit = r * sin(th) + y;
    h = plot(xunit, yunit,'r','LineWidth',1.5);
end


