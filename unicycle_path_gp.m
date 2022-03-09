clear all;
close all;

% Monte Carlo Uncertainty Propogation

%% Set Parameters
N = 1000; % no. ogf samples
dim_state = 3;
dim_input = 2;
dt = 0.05;

% Input bounds
u_min = [-4;-2.0];
u_max = [4;2.0];
v_max = 0.6;

%CBF and CLF parameters
alpha_clf = 0.3;%0.01;
alpha_cbf = 1.0;

%% Initialize robot

% Display
figure(1)
set(gcf,'position',[1,1,2000,1000])
xlim([0 20])
ylim([-10 10])
hold on
daspect([1 1 1])

robot = Unicycle2Dnew(1,0,0,0,1,2,'nominal',1.0,4.0,60*pi/180);   %ID,x,y,yaw,r_safe,D,status
robot_gp = Unicycle2Dnew(1,0,0,0,1,2,'nominal',1.0,4.0,60*pi/180);
robot_kalman = Unicycle2Dnew(1,0,0,0,1,2,'nominal',1.0,4.0,60*pi/180);
robot.plot_update();
leader = SingleIntegrator2D(1,1,0); 

%% Initialize GP
Omega = eye(dim_state);  % between components of output vector
sigma = 0.2;    % sigma for Gaussian kernel
l = 2.0;        % length scale for Gaussian kernel
params = [sigma;l];
gp = MatrixVariateGaussianProcessGeneralized(Omega,params,3,3);


%% Start Simulation

% Collect data for first 10 seconds

tf = 20;
data_counter = 1;

for t=0:dt:tf
    
       
   uL = 0.6;
   vL = 2*cos(0.25*pi*t);%3*cos(0.25*pi*t);
   vel_L = [uL;vL];
   
   u_ref = robot.nominal_controller(robot.X,leader.X,u_min,u_max);     
    u = u_ref;
    if u(1)>v_max
        u(1) = v_max;
    end
    
    % propagate state
    
    leader.control_state([uL; vL],dt);
    
    prev_state = robot.X;
    robot.control_state(u,dt);
    x_dot = (robot.X - prev_state)/dt;  % = f(x) + g(x)u
    robot.input_data(:,data_counter) = t;
    robot.observed_data(:,data_counter) = [prev_state;u];% + 0.1*randn(5,1);        
    
    data_counter = data_counter + 1;
    
    pause(0.01);
    
    
end
t_prev = t;
robot_x_prev = robot.X;
leader_x_prev = leader.X;

%% Fit GP on motion
sigma0 = sigma;
% gp1 = fitrgp( robot.input_data', robot.observed_data(1,:)');
% gp2 = fitrgp( robot.input_data', robot.observed_data(2,:)');

gp1 = fitrgp( robot.input_data', robot.observed_data(1,:)','FitMethod','exact','PredictMethod','exact','KernelFunction','ardmatern32','Sigma',sigma0);
gp2 = fitrgp( robot.input_data', robot.observed_data(2,:)','FitMethod','exact','PredictMethod','exact','KernelFunction','ardmatern32','Sigma',sigma0);

%% Visualize GP
% indexes = 1:1:size(robot.input_data,2);
tf2 = 30;
t_new = t:dt:tf2;
indexes = 1:1:(size(robot.input_data,2)+size(t_new,2));

c = linspace(1,100,length(indexes));

figure(2)
hold on
[mu, ~, cov] = predict( gp1,[robot.input_data'; t_new'] );  
std2 = ( cov(:,2) - cov(:,1) )/2.0;
errorbar( indexes, mu,std2,'--ko' ); %errorbar( i, mean(1),2*sqrt(cov(1,1)),'--ko' )
% scatter( indexes, robot.observed_data(1,:),[],c )

figure(3)
hold on
[mu, ~, cov] = predict( gp2,[robot.input_data'; t_new'] );  
std2 = ( cov(:,2) - cov(:,1) )/2.0;
errorbar( indexes, mu,std2,'--ko' ); %errorbar( i, mean(1),2*sqrt(cov(1,1)),'--ko' )
% scatter( indexes, robot.observed_data(2,:),[],c )

%% Continue

figure(1)

[mu_x, ~, cov_x] = predict( gp1,t_new' );  
std_x = ( cov_x(:,2) - cov_x(:,1) )/2.0;

[mu_y, ~, cov_y] = predict( gp2,t_new' );  
std_y = ( cov_y(:,2) - cov_y(:,1) )/2.0;

errorbar( mu_x, mu_y,std_y,'--ko' );
errorbar( mu_x, mu_y,std_x,'--ko','horizontal' );


