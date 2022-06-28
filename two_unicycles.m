clear all;
close all;

warning('off');

%% Initialization

% Display
figure(1)
hold on
axis equal
set(gca,'DataAspectRatio',[1 1 1])

% Parameters
global r d_min
r=0.25;  % goal display radius
d_min = 1.0;%0.5;

% Robots 
n_robots = 1;
robot = Unicycle2D(1,0,0,pi/4,1,2,'nominal');   %ID,x,y,yaw,r_safe,D,status

% Goals
G1=[7.0 7.0]';
robot(1).G = G1;
circle(G1(1),G1(2),r);

% Obstacles
c1=[5 4]';
r1=3.0;%0.7;%1.6;
% Obstacle(1) = EnvObject2D( c1(1),c1(2) ,'circle','filled',r1,0);
agent = Unicycle2D(1,c1(1),c1(2),-3*pi/4,1,2,'other');   %ID,x,y,yaw,r_safe,D,status

% Simulation parameters
dt = 0.1;
tf = 10.0;%6.0;%2200;


%CBF and CLF parameters
alpha_clf = 0.3;
alpha_cbf = 0.5; %0.5;%0.0;%1.0;

u_plot = [];

%% Simulation
data_counter = 1;
for t=0:dt:tf    
   
   u_agent = [1.8;0.0];
    
   cvx_begin quiet
  
       variable u(2,1)
       variable slack

       % Constraints
       subject to 
            % Obstacle 
%             [h, dh_dxi, dh_dxj] = agent_barrier(robot,agent);
            [h, dh_dxi, dh_dxj] = agent_barrier_angle(robot,agent);
            dh_dxi*( robot.f + robot.g*u ) + dh_dxj*( agent.f + agent.g*u_agent )<= -alpha_cbf*h;
    
            % goal reaching constraints
            ubound = 2.0;
            u_min = [-ubound;-ubound];
            u_max = [ubound;ubound];  
                  
            u(1) <= u_max(1);
            u(1) >= u_min(1);

            u(2) <= u_max(2);
            u(2) >= u_min(2);

            %Lyaounov
            [V, dV_dx] = goal_lyapunov(robot(1));

            % Lyapunov constraint
            dV_dx*( robot.f + robot.g*u ) <= -alpha_clf*V + slack;                         
                
            %Nominal Controller
            u_ref(1:2,1) = nominal_controller(robot,u_min,u_max);     
            
            minimize(  norm( u - u_ref , 2) ) 
    % QP solve            
    cvx_end    
    
    if sum( isnan(u) )>0
        keyboard
        break;
    end
    
    u_plot = [u_plot u];
    
    robot = control_state(robot,u,dt);
    agent = control_state(agent,u_agent,dt);
    
    pause(0.01)   % to show live plots
            
end

t_plot = 0:dt:tf;

figure(2)
hold on
plot(t_plot,u_plot(1,:),'r')
plot(t_plot,u_plot(2,:),'g')
legend('u','w')
