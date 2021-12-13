% Monte Carlo Uncertainty Propogation

%% Set Parameters
N = 1000; % no. ogf samples
dim_state = 3;
dim_input = 2;
dt = 0.05;

% Input bounds
u_min = [-5;-5.0];
u_max = [5;5.0];

%CBF and CLF parameters
alpha_clf = 0.3;%0.01;
alpha_cbf = 1.0;

%% Initialize robot
robot = Unicycle2D(1,0,0,pi,1,2,'nominal');   %ID,x,y,yaw,r_safe,D,status
leader = Unicycle2D(1,0,0,pi,1,2,'nominal'); 

%% Initialize GP
Omega = eye(dim_state);  % between components of output vector
sigma = 0.2;    % sigma for Gaussian kernel
l = 2.0;        % length scale for Gaussian kernel
gp = MatrixVariateGaussianProcessGeneralized(Omega,params,2,3);


%% Start Simulation

% Collect data for first 10 seconds

tf = 10;
data_counter = 1;

for t=0:dt:tf
    
   cvx_begin quiet
    
       variable u(dim_input,1)
       variable slack(1,1)

       subject to

            %Lyaounov
            [V, dV_dx] = goal_lyapunov(robot);

            % Lyapunov constraint
            dV_dx*( robot.f + robot.g*ui ) <= -alpha_clf*V + slack;

            %Nominal Controller
            u_ref = nominal_controller(robot,u_min,u_max); 

        minimize ( norm(u - u_ref,2)^2 + slack^2 )
        
    % QP solve
    cvx_end 
    
    % propagate state
    
    uL = 1.0;
    vL = 1.0;
    leader.control_state([uL; vL],dt);
    
    prev_state = robot.X;
    robot = control_state(robot,u,dt);
    x_dot = (robot.X - prev_state)/dt;  % = f(x) + g(x)u
    robot.input_data(:,data_counter) = [prev_state;ui];
    robot.observed_data(:,data_counter) = x_dot;        
    
    data_counter = data_counter + 1;
    
    
end

% Set GP training data
gp.set_XY(robot.input_data', robot.observed_data');

% resampling important to avoid singular matrix
N = 100;
gp.resample(N);

% Train GP
max_iter = 10;
gp.fit(max_iter, 1);


%% Do N step propagation forward
robot.particles(:,:,1) = robot.X; % states, particle number, time index
robot.weights(:,1) = ones(num_particles,1);


for t=1:1:N
    
   for j=1:1:num_particles
       
      x = robot.particle(:,j,t); 
       
      % design u based on x. same CLF here too
      u = [0;0];
       
      %sample dynamics
      fgx = robot.fx([ x;u ]);
      
      robot.particles(:,j,t=1) = x + fgx * u * dt;
       
   end
   
   % Approximate with a Gaussian based on particles;
   mu = mean( robot.particles( :,:,t ) , 2 );
   cov = zeros(dim_state,dim_state);
   for j=1:1:num_particles
       cov = cov = ( robot.particles( :,j,t+1 ) - mu) * ( robot.particles( :,j,t+1 ) - mu)' ;
   end
   cov = cov/num_particles;
   
   % Approximate Gaussian from Analytical Formulas
   % % %
   
   % plot results
   if mod(N,10)==0
      figure
      
      % Show particles
      scatter( robot.particles( 1,:,t+1 ), robot.particles( 2,:,t+1 ) );       
      title(sprintf('T = %d',T);
      
      % Particle Approximated Gaussian
      
      
      % Propagated Gaussian
      
      
   end
    
end

% Visualize particles at every time step
% N plots


