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
tf = 200;%2200;


%CBF and CLF parameters
alpha_clf = 0.3;%0.01;
alpha_cbf = 1.0;

%% Simulation

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

        ui = u(2*i-1:2*i,1);
        robot(i) = control_state(robot(i),ui,dt);

    end

    pause(0.01)   % to show live plots
            
end

%% Extra functions

function h = circle(x,y,r)
    th = 0:pi/50:2*pi;
    xunit = r * cos(th) + x;
    yunit = r * sin(th) + y;
    h = plot(xunit, yunit,'r','LineWidth',1.5);
end


%% miscelaneous
% 
%  te = 1601;
% if t<te
%     robot(1).goal = ;
%     robot(2).goal = ;
% else
%     robot(1).goal = ;
%     robot(2).goal = ;
% end
% 
% % Goal terms
% 
% for j=1:N_robots
%    robot(j).goal_lyapunov =  (norm(robot(j).X-robot(j).G)^p)/(norm(robot(j).X0-robot(j).G)^p);
% end    
% 
% % Obstacle terms    
% for i=1:N_obstacles
%     for j=1:N_robots
%         robot(j).obs_lyapunov(i) = (r1^2)/(norm(Obstacla(i).X-robot(j).X)^2);            
%     end        
% end



