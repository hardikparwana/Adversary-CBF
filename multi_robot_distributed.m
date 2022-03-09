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
n_robots = 2;
robot(1) = SingleIntegrator2D(1,0,0);   %ID,x,y,yaw,r_safe,D,status
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
tf = 3.5;%2200;
tf2 = 8.0;
prediction_horizon = 40;

update_freq = 20;


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
 
                    [h, dh_dxi, dh_dxj] = agent_barrier(robot(1),robot(2));
                    
                    u1 = u(1:2,1);
                    u2 = u(3:4,1);
                    
                    %barrier constraint
                    dh_dxi*( robot(1).f + robot(1).g*u1 ) <= -alpha_cbf*h;
                    
                    dh_dxj*( robot(2).f + robot(2).g*u2 ) >= 0;
                    
            %Objective
%             minimize( norm( u , 2) + norm(slack) )           
%             minimize( dh_dxi*( robot(1).g*u1 ) - dh_dxj*( robot(2).g*u2 ) + 100*norm(slack))
            
            %Objective
            minimize( norm( u(1:2) , 2) + norm(slack,2) + dh_dxj*robot(2).g*u2 )    

            % Obstacles
            
            for i=1:n_robots
                for j=1:n_obstacles
                    
                    ui = u(2*i-1:2*i,1);

                    [h, dh_dxi] = obstacle_barrier(robot(i),Obstacle(j));
                    
                    dh_dxi*( robot(i).f + robot(i).g*ui ) <= -alpha_cbf*h;
                    
                end
            end



            % goal reaching constraints
            u_min = [-10;-10.0];
            u_max = [10;10.0];
            
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
               
%                 %Nominal Controller
%                 u_ref(2*i-1:2*i,1) = nominal_controller(robot(i),u_min,u_max);         
                         
                
            end         
            
    % QP solve            
    cvx_end    
    
    if sum( isnan(u) )>0
        keyboard
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
        x_dot = (robot(i).X - prev_state)/dt;  % = f(x) + g(x)u
        robot(i).input_data(:,data_counter) = all_states;
        robot(i).observed_data(:,data_counter) = (robot(i).X-prev_state)/dt + 0.1*randn(2,1);        
    end
    data_counter = data_counter + 1;
    
%     keyboard
    
    pause(0.01)   % to show live plots
            
end

robot(1) = plot_update(robot(1));
robot(2) = plot_update(robot(2));

%% Fit GP on motion
sigma0 = sigma;
kernel = 'ardmatern52';
gp_x1 = fitrgp( robot(1).input_data', robot(1).observed_data(1,:)','FitMethod','exact','PredictMethod','exact','KernelFunction',kernel,'Sigma',sigma0);
gp_x2 = fitrgp( robot(2).input_data', robot(2).observed_data(1,:)','FitMethod','exact','PredictMethod','exact','KernelFunction',kernel,'Sigma',sigma0);

gp_y1 = fitrgp( robot(1).input_data', robot(1).observed_data(2,:)','FitMethod','exact','PredictMethod','exact','KernelFunction',kernel,'Sigma',sigma0);
gp_y2 = fitrgp( robot(2).input_data', robot(2).observed_data(2,:)','FitMethod','exact','PredictMethod','exact','KernelFunction',kernel,'Sigma',sigma0);

robot(1).gp_x = gp_x1;
robot(2).gp_x = gp_x2;

robot(1).gp_y = gp_y1;
robot(2).gp_y = gp_y2;

%% Visualize GP
indexes = 1:1:size(robot(1).input_data,2);

% X dot
% figure(2)
% hold on
% [mu, ~, cov] = predict( gp_x1,[robot(1).input_data'] );  
% std2 = ( cov(:,2) - cov(:,1) )/2.0;
% errorbar( indexes, mu,std2,'--ko' ); %errorbar( i, mean(1),2*sqrt(cov(1,1)),'--ko' )
% scatter( indexes, robot(1).observed_data(1,:),'r' )
% 
% [mu, ~, cov] = predict( gp_x2,[robot(2).input_data'] );  
% std2 = ( cov(:,2) - cov(:,1) )/2.0;
% errorbar( indexes, mu,std2,'--ko' ); %errorbar( i, mean(1),2*sqrt(cov(1,1)),'--ko' )
% scatter( indexes, robot(2).observed_data(1,:),'g')
% 
% figure(3)
% hold on
% [mu, ~, cov] = predict( gp_y1,[robot(1).input_data'] );  
% std2 = ( cov(:,2) - cov(:,1) )/2.0;
% errorbar( indexes, mu,std2,'--ko' ); %errorbar( i, mean(1),2*sqrt(cov(1,1)),'--ko' )
% scatter( indexes, robot(1).observed_data(2,:),'r' )
% 
% [mu, ~, cov] = predict( gp_y2,[robot(2).input_data'] );  
% std2 = ( cov(:,2) - cov(:,1) )/2.0;
% errorbar( indexes, mu,std2,'--ko' ); %errorbar( i, mean(1),2*sqrt(cov(1,1)),'--ko' )
% scatter( indexes, robot(2).observed_data(2,:),'g')


% keyboard
%% Simulate further

for t2=t:dt:tf2    
    
   no_of_variables = inputs_per_robot*n_robots;
   
   cvx_begin quiet
  
       variable u(no_of_variables,1)
       variable slack(n_robots,1)

       % Constraints
       subject to 

            % Interagent collisions            
            [h, dh_dxi, dh_dxj] = agent_barrier(robot(1),robot(2));
                    
            u1 = u(1:2,1);
            u2 = u(3:4,1);

            %barrier constraint
            dh_dxi*( robot(1).f + robot(1).g*u1 ) <= -alpha_cbf*h;

            dh_dxj*( robot(2).f + robot(2).g*u2 ) <= -alpha_cbf*h;
            
            %Objective
            minimize( norm( u , 2) + norm(slack) )           
%             minimize( dh_dxi*( robot(1).g*u1 ) - dh_dxj*( robot(2).g*u2 ) )

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
 
            end
            
    % QP solve            
    cvx_end    
    
    all_states = [];
    % Robot state propagation
    for i=1:n_robots
        all_states = [all_states;robot(i).X];
    end
    
    for i=1:n_robots

        prev_state = robot(i).X;
        ui = u(2*i-1:2*i,1);
        robot(i) = control_state(robot(i),ui,dt);
        x_dot = (robot(i).X - prev_state)/dt;  % = f(x) + g(x)u
        robot(i).input_data(:,data_counter) = all_states;
        robot(i).observed_data(:,data_counter) = x_dot + 0.1*randn(2,1);        
    end
    data_counter = data_counter + 1;
    
    % Train again with addition of new data
    if (mod(data_counter,update_freq)==0)
        gp_x1 = fitrgp( robot(1).input_data', robot(1).observed_data(1,:)','FitMethod','exact','PredictMethod','exact','KernelFunction',kernel,'Sigma',sigma0);
        gp_x2 = fitrgp( robot(2).input_data', robot(2).observed_data(1,:)','FitMethod','exact','PredictMethod','exact','KernelFunction',kernel,'Sigma',sigma0);

        gp_y1 = fitrgp( robot(1).input_data', robot(1).observed_data(2,:)','FitMethod','exact','PredictMethod','exact','KernelFunction',kernel,'Sigma',sigma0);
        gp_y2 = fitrgp( robot(2).input_data', robot(2).observed_data(2,:)','FitMethod','exact','PredictMethod','exact','KernelFunction',kernel,'Sigma',sigma0);
        
        robot(1).gp_x = gp_x1;
        robot(2).gp_x = gp_x2;

        robot(1).gp_y = gp_y1;
        robot(2).gp_y = gp_y2;
    end

    
    % Calculate probability of collision
    % sim forward in time with learned data
    if (mod(data_counter,update_freq)==0)
        
        bot = robot;
       
%         for ni=1:1:n_robots % best case
          ni = 1;  
           for nj=1:1:n_robots % treat as adversary
               
               if ( (ni==nj) || (norm( bot(ni).X(1:2) - bot(nj).X(1:2) )>20) )
                   continue
               end
              fprintf("With robot : %d",nj);
               % robot i best, robot j same behavior
               % best case control input
                for ti=t2:dt:(t2+dt*prediction_horizon)

                   cvx_begin quiet
  
                   variable u(2,1)
                   variable slack(n_robots,1)
                   
                   all_states = [bot(1).X;bot(2).X];

                   % Constraints
                   subject to                                                  
                                for j=1:1:n_robots                                        
                                    % collision constraint
                                    [h, dh_dxi, dh_dxj] = agent_barrier(bot(ni),bot(j));

                                    %barrier constraint
                                    [mu_x, ~, cov] = predict( bot(j).gp_x,all_states' );  
                                    [mu_y, ~, cov] = predict( bot(j).gp_y,all_states' );  
                                    x_dot_j = [mu_x;mu_y];
                                    dh_dxi*(bot(ni).f + bot(j).g *u ) + dh_dxj*x_dot_j<= -alpha_cbf*h;       
                                    
                                    if j==nj
                                        %Objective
                                        minimize( dh_dxi * bot(ni).g * u )    
                                    end
                                    bot(j).X = bot(j).X + x_dot_j * dt;
                                    bot(j).color_force = 1;
                                    bot(j) = plot_update(bot(j));
                                    
                                end  
                                
                                for j=1:n_obstacles
                                        [h, dh_dxi] = obstacle_barrier(bot(ni),Obstacle(j));
                                        dh_dxi*( bot(ni).f + bot(ni).g*u ) <= -alpha_cbf*h;
                                end
         

                    % QP solve            
                    cvx_end    
                    
                    bot(ni).color_force = 1;
                    bot(ni) = control_state(bot(ni),u,dt);
                    
                    pause(0.1)
                end
                
                
           end
%         end
    disp("done?")
  end
    
    
    % if going to collide, then change controller to best case controller
    
%     keyboard
    
    pause(0.01)   % to show live plots

end

disp("done!!")
%% Extra functions

function h = circle(x,y,r)
    th = 0:pi/50:2*pi;
    xunit = r * cos(th) + x;
    yunit = r * sin(th) + y;
    h = plot(xunit, yunit,'r','LineWidth',1.5);
end



