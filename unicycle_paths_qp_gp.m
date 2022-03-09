clear all;
close all;

warning('off');

%% Initialization

% Display
figure(1)
hold on

% Parameters
global r d_min
r=0.25;  % goal display radius
d_min=0.1;   % inter agent collision minimum distance

% robot dynamics
inputs_per_robot = 2;

% Robots 
n_robots = 3;
robot(1) = Unicycle2D(1,0,0,pi,1,2,'nominal');   %ID,x,y,yaw,r_safe,D,status
robot(2) = Unicycle2D(1,1,6.5,0,1,2,'nominal'); 
robot(3) = Unicycle2D(3,-4,5,1.73,1,2,'nominal');

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
G3=[-3 11]';

robot(1).G = G1;
robot(2).G = G2;
robot(3).G = G3;

circle(G1(1),G1(2),r);
circle(G2(1),G2(2),r);
circle(G3(1),G3(2),r);

% Obstacles
c1=[4 4]';
r1=1.6;
Obstacle(1) = EnvObject2D( c1(1),c1(2) ,'circle',r1,0);

c2=[-3 8]';
r2=1;
Obstacle(2) = EnvObject2D(c2(1),c2(2),'circle',r2,0 );

n_obstacles = 2;

% Simulation parameters
dt = 0.1;
tf = 2.5;%2200;
tf2 = 3.5;


%CBF and CLF parameters
alpha_clf = 0.3;%0.01;
alpha_cbf = 1.0;

%% Simulation
data_counter = 1;
for t=0:dt:tf    
    
   no_of_variables = inputs_per_robot*n_robots;
   
   cvx_begin quiet
  
       variable u(no_of_variables,1)
       variable slack(n_robots,1)

       % Constraints
       subject to 

            % Interagent collisions
            
            for i=1:n_robots               
                for j=i+1:n_robots
                    
                    [h, dh_dxi, dh_dxj] = agent_barrier(robot(i),robot(j));
                    
                    ui = u(2*i-1:2*i,1);
                    uj = u(2*j-1:2*j,1);
                    
                    %barrier constraint
                    dh_dxi*(robot(i).f + robot(j).g *ui ) + dh_dxj*( robot(i).f + robot(j).g*uj )<= -alpha_cbf*h;
                    
                end                
            end
            
        

            % Obstacles
            
            for i=1:n_robots
                for j=1:n_obstacles
                    
                    ui = u(2*i-1:2*i,1);

                    [h, dh_dxi] = obstacle_barrier(robot(i),Obstacle(j));
                    
                    dh_dxi*( robot(i).f + robot(i).g*ui ) <= -alpha_cbf*h;
                    
                end
            end



            % goal reaching constraints
            u_min = [-5;-5.0];
            u_max = [5;5.0];
            
            for i=1:n_robots
                
                ui = u(2*i-1:2*i,1);
                slacki = slack(i);
                
                ui(1) <= u_max(1);
                ui(1) >= u_min(1);
                
                ui(2) <= u_max(2);
                ui(2) >= u_min(2);
               
                %Lyaounov
                [V, dV_dx] = goal_lyapunov(robot(i));
                
                % Lyapunov constraint
                dV_dx*( robot(i).f + robot(i).g*ui ) <= -alpha_clf*V + slacki;
               
                %Nominal Controller
                u_ref(2*i-1:2*i,1) = nominal_controller(robot(i),u_min,u_max);         
                         
                
            end
           
            %Objective
            minimize( norm( u - u_ref , 2) + norm(slack,2) )              
            
    % QP solve            
    cvx_end    

    % Robot state propagation            
    for i=1:n_robots

        prev_state = robot(i).X;
        ui = u(2*i-1:2*i,1);
        robot(i) = control_state(robot(i),ui,dt);
        x_dot = (robot(i).X - prev_state)/dt;  % = f(x) + g(x)u
        robot(i).input_data(:,data_counter) = t;
        robot(i).observed_data(:,data_counter) = [prev_state;u];        
    end
    data_counter = data_counter + 1;
    
%     keyboard
    
    pause(0.01)   % to show live plots
            
end

%% Fit GP on motion
sigma0 = sigma;
kernel = 'ardmatern52';
gp_x1 = fitrgp( robot(1).input_data', robot(1).observed_data(1,:)','FitMethod','exact','PredictMethod','exact','KernelFunction',kernel,'Sigma',sigma0);
gp_x2 = fitrgp( robot(2).input_data', robot(2).observed_data(1,:)','FitMethod','exact','PredictMethod','exact','KernelFunction',kernel,'Sigma',sigma0);
gp_x3 = fitrgp( robot(3).input_data', robot(3).observed_data(1,:)','FitMethod','exact','PredictMethod','exact','KernelFunction',kernel,'Sigma',sigma0);

gp_y1 = fitrgp( robot(1).input_data', robot(1).observed_data(2,:)','FitMethod','exact','PredictMethod','exact','KernelFunction',kernel,'Sigma',sigma0);
gp_y2 = fitrgp( robot(2).input_data', robot(2).observed_data(2,:)','FitMethod','exact','PredictMethod','exact','KernelFunction',kernel,'Sigma',sigma0);
gp_y3 = fitrgp( robot(3).input_data', robot(3).observed_data(2,:)','FitMethod','exact','PredictMethod','exact','KernelFunction',kernel,'Sigma',sigma0);


%% Visualize GP
% indexes = 1:1:size(robot.input_data,2);
t_new = t:dt:tf2;
% indexes = 1:1:(size(robot(1).input_data,2)+size(t_new,2));
% 
% c = linspace(1,100,length(indexes));
% 
% figure(2)
% hold on
% [mu, ~, cov] = predict( gp_x1,[robot(1).input_data'; t_new'] );  
% std2 = ( cov(:,2) - cov(:,1) )/2.0;
% errorbar( indexes, mu,std2,'--ko' ); %errorbar( i, mean(1),2*sqrt(cov(1,1)),'--ko' )
% % scatter( indexes, robot.observed_data(1,:),[],c )
% 
% figure(3)
% hold on
% [mu, ~, cov] = predict( gp_x2,[robot(1).input_data'; t_new'] );  
% std2 = ( cov(:,2) - cov(:,1) )/2.0;
% errorbar( indexes, mu,std2,'--ko' ); %errorbar( i, mean(1),2*sqrt(cov(1,1)),'--ko' )
% % scatter( indexes, robot.observed_data(2,:),[],c )

%% Visualize Paths
figure(1)


% robot 1
[mu_x, ~, cov_x] = predict( gp_x1,t_new' );  
std_x = ( cov_x(:,2) - cov_x(:,1) )/2.0;

[mu_y, ~, cov_y] = predict( gp_y1,t_new' );  
std_y = ( cov_y(:,2) - cov_y(:,1) )/2.0;

% errorbar( mu_x, mu_y,std_y,'--ko' );
% errorbar( mu_x, mu_y,std_x,'--ko','horizontal' );

for i=1:1:size(mu_x,1)
    h = plot_gaussian_ellipsoid([mu_x(i) mu_y(i)],[std_x(i)/2 0 ; 0 std_y(i)/2],2);
end

% robot 2
[mu_x, ~, cov_x] = predict( gp_x2,t_new' );  
std_x = ( cov_x(:,2) - cov_x(:,1) )/2.0;

[mu_y, ~, cov_y] = predict( gp_y2,t_new' );  
std_y = ( cov_y(:,2) - cov_y(:,1) )/2.0;

% errorbar( mu_x, mu_y,std_y,'--ko' );
% errorbar( mu_x, mu_y,std_x,'--ko','horizontal' );

for i=1:1:size(mu_x,1)
    h = plot_gaussian_ellipsoid([mu_x(i) mu_y(i)],[std_x(i)/2 0 ; 0 std_y(i)/2],2);
end

% robot 3
[mu_x, ~, cov_x] = predict( gp_x3,t_new' );  
std_x = ( cov_x(:,2) - cov_x(:,1) )/2.0;

[mu_y, ~, cov_y] = predict( gp_y3,t_new' );  
std_y = ( cov_y(:,2) - cov_y(:,1) )/2.0;

% errorbar( mu_x, mu_y,std_y,'--ko' );
% errorbar( mu_x, mu_y,std_x,'--ko','horizontal' );

for i=1:1:size(mu_x,1)
    h = plot_gaussian_ellipsoid([mu_x(i) mu_y(i)],[std_x(i)/2 0 ; 0 std_y(i)/2],2);
end

%% Simulate further

for t=t:dt:tf2    
    
   no_of_variables = inputs_per_robot*n_robots;
   
   cvx_begin quiet
  
       variable u(no_of_variables,1)
       variable slack(n_robots,1)

       % Constraints
       subject to 

            % Interagent collisions
            
            for i=1:n_robots               
                for j=i+1:n_robots
                    
                    [h, dh_dxi, dh_dxj] = agent_barrier(robot(i),robot(j));
                    
                    ui = u(2*i-1:2*i,1);
                    uj = u(2*j-1:2*j,1);
                    
                    %barrier constraint
                    dh_dxi*(robot(i).f + robot(j).g *ui ) + dh_dxj*( robot(i).f + robot(j).g*uj )<= -alpha_cbf*h;
                    
                end                
            end
            
        

            % Obstacles
            
            for i=1:n_robots
                for j=1:n_obstacles
                    
                    ui = u(2*i-1:2*i,1);

                    [h, dh_dxi] = obstacle_barrier(robot(i),Obstacle(j));
                    
                    dh_dxi*( robot(i).f + robot(i).g*ui ) <= -alpha_cbf*h;
                    
                end
            end



            % goal reaching constraints
            u_min = [-5;-5.0];
            u_max = [5;5.0];
            
            for i=1:n_robots
                
                ui = u(2*i-1:2*i,1);
                slacki = slack(i);
                
                ui(1) <= u_max(1);
                ui(1) >= u_min(1);
                
                ui(2) <= u_max(2);
                ui(2) >= u_min(2);
               
                %Lyaounov
                [V, dV_dx] = goal_lyapunov(robot(i));
                
                % Lyapunov constraint
                dV_dx*( robot(i).f + robot(i).g*ui ) <= -alpha_clf*V + slacki;
               
                %Nominal Controller
                u_ref(2*i-1:2*i,1) = nominal_controller(robot(i),u_min,u_max);         
                         
                
            end
           
            %Objective
            minimize( norm( u - u_ref , 2) + norm(slack,2) )              
            
    % QP solve            
    cvx_end    

    % Robot state propagation            
    for i=1:n_robots

        prev_state = robot(i).X;
        ui = u(2*i-1:2*i,1);
        robot(i) = control_state(robot(i),ui,dt);
    end
    
%     keyboard
    
    pause(0.01)   % to show live plots
            
end


%% Extra functions

function h = circle(x,y,r)
    th = 0:pi/50:2*pi;
    xunit = r * cos(th) + x;
    yunit = r * sin(th) + y;
    h = plot(xunit, yunit,'r','LineWidth',1.5);
end



