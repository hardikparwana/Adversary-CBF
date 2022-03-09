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

tf = 10;
data_counter = 1;

for t=0:dt:tf
    
%    cvx_begin quiet
%     
%        variable u(dim_input,1)
%        variable slack(1,1)
       
       uL = 0.6;
       vL = 2*cos(0.25*pi*t);%3*cos(0.25*pi*t);
       vel_L = [uL;vL];

%        subject to
% 
%             %Lyaounov
%             [V, dV_dx_agent, dV_dx_target] = robot.goal_lyapunov(robot.X,leader.X);
%             [h1, dh1_dx_agent, dh1_dx_target] = robot.cbf1_loss(robot.X,leader.X);
%             [h2, dh2_dx_agent, dh2_dx_target] = robot.cbf2_loss(robot.X,leader.X);
%             [h3, dh3_dx_agent, dh3_dx_target] = robot.cbf3_loss(robot.X,leader.X);
%             
%             if (h3<0)
%                 h3
%             end
% 
%             if(h1<0)
%                 h1;
%             end
%             
%             if(h2<0)
%                 h2
%             end
% 
%             % Lyapunov constraint
%             dV_dx_agent*( robot.f + robot.g*u) + dV_dx_target * vel_L  <= -alpha_clf*V + slack;
%             
%             % Barrier Constraints
%             dh1_dx_agent * ( robot.f + robot.g*u ) + dh1_dx_target * vel_L >= -alpha_cbf * h1;
%             dh2_dx_agent * ( robot.f + robot.g*u ) + dh2_dx_target * vel_L >= -alpha_cbf * h2;
%             dh3_dx_agent * ( robot.f + robot.g*u ) + dh3_dx_target * vel_L >= -alpha_cbf * h3;
% 
            %Nominal Controller
            u_ref = robot.nominal_controller(robot.X,leader.X,u_min,u_max); 
%             
%             u(1) >= u_min(1);
%             u(2) >= u_min(2);
%             
%             u(1) <= u_max(1);
%             u(2) <= u_max(2);
% 
%         minimize ( norm(u - u_ref,2) + norm(slack) )
%         
%     % QP solve
%     cvx_end 
    
    u = u_ref;
    if u(1)>v_max
        u(1) = v_max;
    end
    
    % propagate state
    
    leader.control_state([uL; vL],dt);
    
    prev_state = robot.X;
    robot.control_state(u,dt);
    x_dot = (robot.X - prev_state)/dt;  % = f(x) + g(x)u
    robot.input_data(:,data_counter) = [prev_state;1;u];
    robot.observed_data(:,data_counter) = x_dot + sigma*randn(3,1);        
    
    data_counter = data_counter + 1;
    
    pause(0.01);
    
    
end
t_prev = t;
robot_x_prev = robot.X;
leader_x_prev = leader.X;
% keyboard 

%% Train GP

% % Set GP training data
% gp.set_XY(robot.input_data', robot.observed_data');
% 
% % resampling important to avoid singular matrix
% N = 200;
% gp.resample(N);
% 
% % Train GP
% max_iter = 30;
% gp.fit(max_iter, 1);
sigma0 = sigma;
gp1 = fitrgp( robot.input_data', robot.observed_data(1,:)');
gp2 = fitrgp( robot.input_data', robot.observed_data(2,:)');
gp3 = fitrgp( robot.input_data', robot.observed_data(3,:)');
% gp1 = fitrgp( robot.input_data', robot.observed_data(1,:)','FitMethod','exact','PredictMethod','exact','KernelFunction','ardsquaredexponential','Sigma',sigma0);
% gp2 = fitrgp( robot.input_data', robot.observed_data(2,:)','FitMethod','exact','PredictMethod','exact','KernelFunction','ardsquaredexponential','Sigma',sigma0);
% gp3 = fitrgp( robot.input_data', robot.observed_data(3,:)','FitMethod','exact','PredictMethod','exact','KernelFunction','ardsquaredexponential','Sigma',sigma0);

%% Visualize GP
indexes = 1:1:size(robot.input_data,2);
c = linspace(1,100,length(indexes));

figure(2)
hold on
[mu, ~, cov] = predict( gp1,robot.input_data' );  
std2 = ( cov(:,2) - cov(:,1) )/2.0;
errorbar( indexes, mu,std2,'--ko' ); %errorbar( i, mean(1),2*sqrt(cov(1,1)),'--ko' )
scatter( indexes, robot.observed_data(1,:),[],c )

figure(3)
hold on
[mu, ~, cov] = predict( gp2,robot.input_data' );  
std2 = ( cov(:,2) - cov(:,1) )/2.0;
errorbar( indexes, mu,std2,'--ko' ); %errorbar( i, mean(1),2*sqrt(cov(1,1)),'--ko' )
scatter( indexes, robot.observed_data(2,:),[],c )

figure(4)
hold on
[mu, ~, cov] = predict( gp3,robot.input_data' );  
std2 = ( cov(:,2) - cov(:,1) )/2.0;
errorbar( indexes, mu,std2,'--ko' ); %errorbar( i, mean(1),2*sqrt(cov(1,1)),'--ko' )
scatter( indexes, robot.observed_data(3,:),[],c )

% 
% for i=1:1:size(robot.input_data,2)
%    
%     [means, cov, omega] = gp.predict( robot.input_data(:,i) );
%     u = robot.input_data(dim_state+1:end,i);
%     mean = mean * u;
%     cov = u' * cov * u * omega;
%     
%     obs = robot.observed_data(:,i);
%     
%     figure(2)
%     errorbar( i, means(1),2*sqrt(cov(1,1)),'--ko' )
%     scatter( i, obs(1) )
%     
%     figure(3)
%     errorbar( i, means(2),2*sqrt(cov(2,2)),'--ko' )
%     scatter( i, obs(2) )
%     
%     figure(4)
%     errorbar( i, means(3),2*sqrt(cov(3,3)),'--ko' )
%     scatter( i, obs(3) )
%     
% end


%% Do N step propagation forward

num_particles = 500;
T = 100;

simulate_particles(robot,robot_gp,leader,t_prev,robot_x_prev,leader_x_prev,num_particles,dim_state,T,u_min,u_max,v_max,gp1,gp2,gp3);


% f2 = figure();
% ax2 = copyobj(ax1,f2);

function out = simulate_particles(robot,robot_gp,leader,t_prev,robot_x_prev,leader_x_prev,num_particles,dim_state,T,u_min,u_max,v_max,gp1,gp2,gp3)

    robot.X = robot_x_prev;
    robot_gp.X = robot_x_prev;
%     robot.kalman.X = robot_x_prev;
    leader.X = leader_x_prev;
 
    % Monte Carlo
    robot.particles = zeros( dim_state, num_particles, T );
    robot.particles(:,:,1) = repmat(robot.X,1,num_particles); % states, particle number, time index
    robot.weights = ones(num_particles,1);


    % Unscented KF type: sigma points

    colors = ['r','k','c','m','y','b'];

    dt = 0.1;
    % for t=1:1:T
    t_index = 1;
    for t=t_prev:dt:(t_prev+(T-1)*dt)

       uL = 0.6;
       vL = cos(0.25*pi*t);%3*cos(0.25*pi*t);
       vel_L = [uL;vL];

%        if mod(t_index,10)==0
%         
%            figure(6)
%            subplot(2,2,1)
%            hold off
%            subplot(2,2,2)
%            hold off
%            subplot(2,2,3)
%            hold off
%            
%        end

       % Particles      
       for j=1:1:num_particles

          x = robot.particles(:,j,t_index); 
          Sigma = 0.00001*eye(1);

          % design u based on x. same CLF here too
    %       u = [1;0.5;0.1];

          u = robot.nominal_controller(x,leader.X,u_min,u_max); 
          if u(1)>0.6
            u(1) = 0.6;
          end
          u = [1;u];

          [mu1, ~, cov1] = predict( gp1, [x;u]' );
          var1 = (cov1(2)-cov1(1))/2.0;%( (cov1(2)-cov1(1))/2.0/1.96 )^2;
          [mu2, ~, cov2] = predict( gp2, [x;u]' );
          var2 = (cov2(2)-cov2(1))/2.0; %( (cov2(2)-cov2(1))/2.0/1.96 )^2;
          [mu3, ~, cov3] = predict( gp3, [x;u]' );
          var3 = (cov3(2)-cov3(1))/2.0; %( (cov3(2)-cov3(1))/2.0/1.96 )^2;
    %       mu = mu * u;
    %       cov = (u' * cov * u) * omega ;

          mu = [mu1;mu2;mu3];
          cov = diag([var1 var2 var3]);

          fgx_gp = sqrtm(cov) * randn(3,1) + mu;

          %sample dynamics
          fgx_org = robot.fgx(x) * u;    

%           if mod(t_index,10)==0
%               
%               subplot(2,2,1)
%               errorbar( j, mu1,var1,'--ko' );
%               hold on
%               scatter(j, fgx_org(1));
%     
%               subplot(2,2,2)
%               errorbar( j, mu2,var2,'--ko' );
%               hold on
%               scatter(j, fgx_org(2));
%     
%               subplot(2,2,3)
%               errorbar( j, mu3,var3,'--ko' );
%               hold on
%               scatter(j, fgx_org(3));
%               
%           end

          robot.particles(:,j,t_index+1) = x + fgx_gp * dt;  
    %       robot.particles(:,j,t_index+1) = x + fgx_org * dt;  

       end

    %    figure(6)
    %    hold off

       % Approximate with a Gaussian based on particles;
       mu = mean( robot.particles( :,:,t_index+1 ) , 2 );
       cov = zeros(dim_state,dim_state);
       for j=1:1:num_particles
           cov = cov + ( robot.particles( :,j,t_index+1 ) - mu) * ( robot.particles( :,j,t_index+1 ) - mu)' ;
       end
       cov = cov/num_particles;


       % Approximate Gaussian from Analytical Formulas
       % % %

       % plot results
       figure(1)
       leader.control_state([uL; vL],dt);

       % Perfect Robot
       u = robot.nominal_controller(robot.X,leader.X,u_min,u_max); 
       if u(1)>v_max
         u(1) = v_max;
       end
       robot.control_state(u,dt);
       
       % GP mean Robot
       [mu1, ~, ~] = predict( gp1, [robot.X;1;u]' );
       [mu2, ~, ~] = predict( gp2, [robot.X;1;u]' );
       [mu3, ~, ~] = predict( gp3, [robot.X;1;u]' );
       robot_gp.control_state_fgx([mu1;mu2;mu3],dt);
       
       % Kalman Robot
%        A = [0 0 -u(1)*sin(robot_kalman.X(3));
%            0 0 u(1)*cos(robot_kalman.X(3));
%            0 0 0]*dt + eye(3);
%        B = [cos(robot_kalman.X(3)) 0;
%            sin(robot_kalman.X(3)) 0;
%            0 1]*dt; 
       
%        robot.Kalman

       if mod(t_index,10)==0
          figure(1)
    %       xlim([4,10])
    %       ylim([4,8])
    %       hold on
    %       fprintf( "Mean position  x: %f , y: %f  \n",mean( robot.particles( 1,:,t_index+1 ) ), mean( robot.particles( 2,:,t_index+1 ) ) )
          % Show particles
          scatter( robot.particles( 1,:,t_index+1 ), robot.particles( 2,:,t_index+1 ), colors( mod( (t_index/10),6 )+1 ), 'MarkerEdgeAlpha',.2 );       
          title(sprintf('T = %d',t_index));

          % Particle Approximated Gaussian
          h = plot_gaussian_ellipsoid(mu(1:2),cov(1:2,1:2),2);
          set(h,'color',colors( mod( (t_index/10)+5,6 )+1 )); 
          set(h,'LineWidth',2)


          scatter(robot.X(1),robot.X(2),'filled',colors( mod( (t_index/10)+5,6 )+1 ),'MarkerEdgeAlpha',.2 )
          scatter(robot_gp.X(1),robot_gp.X(2),'filled',colors( mod( (t_index/10)+5,6 )+1 ) )

          % Propagated Gaussian
          
%           pause(0.5)
       end




       t_index = t_index + 1;

    end
    
end

% Visualize particles at every time step
% N plots


