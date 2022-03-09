%%
% Case: smooth following. stuck in a circle
% no input bounds, no OBSTACLES. min of u and slack norm

%%

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
d_min=0.1;   % inter agent collision minimum distance

% robot dynamics
inputs_per_robot = 2;

% Robots 
n_robots = 2;
robot(1) = SingleIntegrator2D(1,0,2);   %ID,x,y,yaw,r_safe,D,status
robot(2) = SingleIntegrator2D(2,1,6.5); 

% Dynamics Estimator

% GP parameters
% Concatenate all f(x) and g(x) in a single vector: vector size = n + n*m
n = 3;
m = 2;
input_vec_size = n + m; %x + u % OR n + n*m;
output_vec_size = n;
Omega = eye(output_vec_size);  % between components of output vector
sigma = 0.1;    % sigma for Gaussian kernel
l = 2.0;        % length scale for Gaussian kernel

% Goals
G1=[-4.5 7.2]';
G2=[5 -1]';

robot(1).G = G1;
robot(2).G = G2;

circle(G1(1),G1(2),r);
circle(G2(1),G2(2),r);

% Obstacles
c1=[3 4]';
r1=0.7;%1.6;
Obstacle(1) = EnvObject2D( c1(1),c1(2) ,'circle',r1,0);

c2=[-3 8]';
r2=0.6;%1;
Obstacle(2) = EnvObject2D(c2(1),c2(2),'circle',r2,0 );

n_obstacles = 2;

% Simulation parameters
dt = 0.1;
tf = 20.0;%2200;
tf2 = 8.0;
prediction_horizon = 40;

update_freq = 20;


%CBF and CLF parameters
alpha_clf = 0.3;%0.01;
alpha_cbf = 0.3;
use_horizon = 0;
horizon = 10;

%% Simulation
data_counter = 1;
for t=0:dt:tf    
    
   no_of_variables = inputs_per_robot*n_robots;
   
   cvx_begin quiet
  
       variable u(no_of_variables,1)
       variable slack(n_robots,1)
       dual variable y_agents{2}
       dual variable y_obstacles{2}

       % Constraints
       subject to 

            % Interagent collisions
 
                    [h, dh_dxi, dh_dxj] = agent_barrier(robot(1),robot(2));
                    robot(1).dh_dx(data_counter,:) = -dh_dxi;
                    robot(1).dh_dx_complete(data_counter,:) = [-dh_dxi -dh_dxj];
                    robot(2).dh_dx(data_counter,:) = -dh_dxj;
                    if use_horizon==1
                        
                        fprintf("hello")
                        
                    else
                        if (data_counter>1)
                            robot(1).sum_h(data_counter) = robot(1).sum_h(data_counter-1) + h*dt;
                            robot(1).upper_bound(data_counter) = ( robot(1).h(1) - h )/alpha_cbf;
                        else
                            robot(1).sum_h(1) = h*dt;
                            robot(1).upper_bound(1) = 0;
                        end                        
                    end
                    
                    robot(1).h(data_counter) = h;
                    
                    u1 = u(1:2,1);
                    u2 = u(3:4,1);
                    
                    %barrier constraint
                    dh_dxi*( robot(1).f + robot(1).g*u1 ) + dh_dxj*( robot(2).f + robot(2).g*u2 ) <= -alpha_cbf*h;
                    
                    dh_dxj*( robot(2).f + robot(2).g*u2 ) > 6;%3; % -alpha_cbf*h;
                    
            %Objective
%             minimize( norm( u , 2) + norm(slack) )           
%             minimize( dh_dxi*( robot(1).g*u1 ) - dh_dxj*( robot(2).g*u2 ) + 100*norm(slack))
            
            %Objective
%             minimize( norm( u(1:2) , 2) + norm(slack,2) + dh_dxj*robot(2).g*u2 )    
            minimize(  norm(u,2) + norm(slack(1),2) )%- dh_dxj*robot(2).g*u2 )  
%             minimize(  - dh_dxj*robot(2).g*u2 ) 
%             minimize( 10*norm(slack,2) + dh_dxj*robot(2).g*u2 )  
%             minimize( 10*norm(slack,2))  

            % Obstacles
            
%             for i=1:n_robots
%                 for j=1:n_obstacles
%                     
%                     ui = u(2*i-1:2*i,1);
% 
%                     [h, dh_dxi] = obstacle_barrier(robot(i),Obstacle(j));
%                     
%                     dh_dxi*( robot(i).f + robot(i).g*ui ) <= -alpha_cbf*h;
%                     
%                 end
%             end



            % goal reaching constraints
            ubound = 20.0;
            u_min = [-ubound;-ubound];
            u_max = [ubound;ubound];
            
            for i=1:n_robots
                
                ui = u(2*i-1:2*i,1);
                slacki = slack(i);
                
%                 ui(1) <= u_max(1);
%                 ui(1) >= u_min(1);
%                 
%                 ui(2) <= u_max(2);
%                 ui(2) >= u_min(2);
%                
                %Lyaounov
                [V, dV_dx] = goal_lyapunov(robot(i));
                robot(i).V(data_counter) = V;
                robot(i).dV_dx(data_counter,:) = -dV_dx;
                
                % Lyapunov constraint
                dV_dx*( robot(i).f + robot(i).g*ui ) <= -alpha_clf*V + slacki;
               
%                 %Nominal Controller
%                 u_ref(2*i-1:2*i,1) = nominal_controller(robot(i),u_min,u_max);         
                         
                
            end      
            
            robot(1).dV_dx_complete(data_counter,:) = [robot(1).dV_dx(data_counter,:) robot(2).dV_dx(data_counter,:)];
            
            robot(1).inner_prod(data_counter) = robot(1).dh_dx(data_counter,:) * robot(1).dV_dx(data_counter,:)';
            robot(2).inner_prod(data_counter) = robot(2).dh_dx(data_counter,:) * robot(2).dV_dx(data_counter,:)';
            
            robot(1).inner_prod_complete(data_counter) = robot(1).dh_dx_complete(data_counter,:) * robot(1).dV_dx_complete(data_counter,:)';
            
    % QP solve            
    cvx_end    
    
    if sum( isnan(u) )>0
        keyboard
        break;
    end
    
    all_states = [];
    % Robot state propagation
    for i=1:n_robots
        all_states = [all_states;robot(i).X];
    end
    for i=1:n_robots

        prev_state = robot(i).X;
        ui = u(2*i-1:2*i,1);
        robot(i) = control_state(robot(i),ui,dt);
        title("t="+num2str(t))
        x_dot = (robot(i).X - prev_state)/dt;  % = f(x) + g(x)u
        robot(i).input_data(:,data_counter) = all_states;
        robot(i).observed_data(:,data_counter) = (robot(i).X-prev_state)/dt + 0.1*randn(2,1);        
    end
    data_counter = data_counter + 1;
    
%     keyboard
    
    pause(0.01)   % to show live plots
            
end
tplot=0:dt:t;  
figure(2)
plot(tplot,robot(1).sum_h)
hold on
plot(tplot,robot(1).upper_bound,'k')
legend('\int h  dt','upper_bound')
% yline(robot(1).sum_h(1)/dt/alpha_cbf,'g')
% legend('\int h  dt','upper_bound','lower bound')

figure(3)
plot(tplot,robot(1).h)

figure(4)
hold on
plot(tplot,robot(1).inner_prod,'r')
% plot(robot(2).inner_prod,'g')
plot(tplot,-robot(1).h*10,'k')
plot(tplot,robot(1).V,'m')
yline(0)
% legend('Robot 1','Robot 2','barrier function','1 Lyapunov')
legend('Robot 1','barrier function','1 Lyapunov')

% plot(tplot,robot(1).inner_prod_complete)


function h = circle(x,y,r)
    th = 0:pi/50:2*pi;
    xunit = r * cos(th) + x;
    yunit = r * sin(th) + y;
    h = plot(xunit, yunit,'r','LineWidth',1.5);
end