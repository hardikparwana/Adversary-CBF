clear all;
close all;

warning('off');
num = 0;
% Display
figure(1+num)
hold on
% set(gca,'DataAspectRatio',[1 1 1])
axis([ 0 5 0 8 ])

% Parameters
global r d_min
r=0.25;  % goal display radius
d_min=0.1;   % inter agent collision minimum distance

% robot dynamics
inputs_per_robot = 2;

% Robots 
n_robots = 3;
robot(1) = SingleIntegrator2D(1,3.0,1,'reactive');   %ID,x,y,yaw,r_safe,D,status
robot(2) = SingleIntegrator2D(2,2.5,0,'reactive'); 
robot(3) = SingleIntegrator2D(3,3.5,0,'reactive');

robot_nominal(1) = SingleIntegrator2D(1,3.0,1,'nominal');   %ID,x,y,yaw,r_safe,D,status
robot_nominal(2) = SingleIntegrator2D(2,2.5,0,'nominal'); 
robot_nominal(3) = SingleIntegrator2D(3,3.5,0,'nominal');

num_robots = size(robot,2);

% Humans
human(1) = SingleIntegrator2D(1,0.0,4,'human');
human_nominal(1) = SingleIntegrator2D(1,0.0,4,'human');

dt = 0.05;
tf = 7.0;
alpha_cbf = 0.8;%1.0;

% Record Video
% myVideo = VideoWriter('trust_cbf'); %open video file
% myVideo.FrameRate = 10;  %can adjust this, 5 - 10 works well for me
% open(myVideo)

% Nominal simulation
% for t=0:dt:tf
%     
%    % Human movement: straight line
%    u_human = [1.0;0.0];
%    human(1) = control_state(human(1),u_human,dt);
%    
%    % Opne Loop Robots:
%    u_robot = [0;1.0];
%    robot(1) = control_state(robot(1),u_robot,dt);
%    robot(2) = control_state(robot(2),u_robot,dt);
%    robot(3) = control_state(robot(3),u_robot,dt);
%    
%    pause(0.01)
%     
% end
% keyboard
% Safety Critical un-cooperative behaviour
alpha_der_max = 0.5;%2.0;
min_dist = 0.05; %0.5
contri = [0];
for t=0:dt:tf
    
    % Human movement: straight line. Therefore uncooperative
    u_human = [1.0;0.0];
    human_nominal(1) = control_state(human_nominal(1),u_human,dt);
%     u_human_nominal = u_human;
%     u_human = [0.8;-0.5];

    % Human believed movement
    [V_nominal, dV_dx_nominal] = lyapunov(human,human_nominal(1).X);
    u_human_nominal = -1.0*dV_dx_nominal'/norm(dV_dx_nominal);

    % Human actual movement
    [V, dV_dx] = lyapunov(human,robot(1).X);
    u_human = -1.0*dV_dx'/norm(dV_dx);   
%     u_human = [0.8;-0.5];
    human(1) = control_state(human(1),u_human,dt);   

    % Open Loop Robots:
    u_robot = [0;1.0];
    robot_nominal(1) = control_state(robot_nominal(1),u_robot,dt);
    robot_nominal(2) = control_state(robot_nominal(2),u_robot,dt);
    robot_nominal(3) = control_state(robot_nominal(3),u_robot,dt);
   
    cvx_begin quiet        
    variable u(2,num_robots)
    
    subject to       
    
        for i=1:1:num_robots
           
           % Nominal robot
           [V, dV_dx] = lyapunov(robot(i),robot_nominal(i).X);
           V_max = 0.2; k = 2.0;
           u_nominal(:,i) = -k*V*dV_dx'/norm(dV_dx); % Exponentially stable controller  
           
           % Calculate best case input
           
%            % Human
%            cvx_begin
%                 variable u(2,1)
%                 subject to
%                     
%                 
%            % Robot 1
%            
%            
%            % Robot 2
           
           
            
        end
   
        for i=1:1:num_robots
            
            disp("start")   

           %%%%%%%%%%%%%%%%% Human CBF constraint
           [h, dh_dxi, dh_dxj] = agent_barrier(robot(i),human(1));
           A = dh_dxj; b = -robot(i).human_alpha * h - dh_dxi*robot(i).inputs(:,end); 
                             
           robot(i).trust_human = compute_trust(A,b,u_human,u_human_nominal,h,min_dist);      
           fprintf("robot: %d trust for human = %f \n", i, robot(i).trust_human);
           
           robot(i).human_alpha = robot(i).human_alpha + alpha_der_max*robot(i).trust_human;
           if (robot(i).human_alpha<0)
               robot(i).human_alpha=0.01;
           end
           robot(i).human_alphas(end+1) = robot(i).human_alpha;
           dh_dxi*( robot(i).f + robot(i).g*u(:,i)) + dh_dxj*( human(1).f + human(1).g*u_human ) <= -robot(i).human_alpha * h; 
                      
           %%%%%%%%%%%%%%%%% 1st neighbor         
           index = mod(i,num_robots)+1;
           [h, dh_dxi, dh_dxj] = agent_barrier(robot(i),robot(index));
           
           A = dh_dxj; b = -robot(i).robot_alpha(1) * h - dh_dxi*robot(i).inputs(:,end); 
           
           robot(i).trust_robot(1) = compute_trust(A,b,robot(index).inputs(:,end),u_nominal(:,index),h,min_dist);         
           fprintf("robot: %d trust for robot %d = %f \n", i, index, robot(i).trust_robot(1));
           
           robot(i).robot_alpha(1) = robot(i).robot_alpha(1) + alpha_der_max*robot(i).trust_robot(1);
           if (robot(i).robot_alpha(1)<0)
               robot(i).robot_alpha(1)=0.01;
           end
           dh_dxi*( robot(i).f + robot(i).g*u(:,i)) + dh_dxj*( robot(index).f + robot(index).g*u(:,index) ) <= -robot(i).robot_alpha(1) * h; 
           
           %%%%%%%%%%%%%%%%%% 2nd neighbor         
           index = mod(i+1,num_robots)+1;
           [h, dh_dxi, dh_dxj] = agent_barrier(robot(i),robot(index));
           A = dh_dxj; b = -robot(i).robot_alpha(1) * h - dh_dxi*robot(i).inputs(:,end); 
           
           robot(i).trust_robot(2) = compute_trust(A,b,robot(index).inputs(:,end),u_nominal(:,index),h,min_dist);  
           fprintf("robot: %d trust for robot %d = %f \n", i, index, robot(i).trust_robot(2));
  
           robot(i).robot_alpha(2) = robot(i).robot_alpha(2) + alpha_der_max*robot(i).trust_robot(2);
           if (robot(i).robot_alpha(1)<0)
               robot(i).robot_alpha(1)=0.01;
           end
           dh_dxi*( robot(i).f + robot(i).g*u(:,i)) + dh_dxj*( robot(index).f + robot(index).g*u(:,index) ) <= -robot(i).robot_alpha(2) * h;    
           
           
           %%%%%%%%%%%%%%%%%% store alphas for plot
           robot(i).robot_alphas(:,end+1) = [robot(i).robot_alpha(1); robot(i).robot_alpha(2)];
           
         end
   
   minimize( norm(u-u_nominal) );
   
   cvx_end
   
   % Know future movement so see if human is contributing or not        
    robot(i) = control_state(robot(i),u,dt);
    robot(1).inputs(:,end+1) = u(:,1);
    robot(2).inputs(:,end+1) = u(:,2);
    robot(3).inputs(:,end+1) = u(:,3);
    
    robot(1) = control_state(robot(1),u(:,1),dt);
    robot(2) = control_state(robot(2),u(:,2),dt);
    robot(3) = control_state(robot(3),u(:,3),dt);

%     frame = getframe(gcf); %get frame
%     writeVideo(myVideo, frame);
%    
   pause(0.05)
    
end

% close(myVideo)

figure(2+num)
hold on
plot(robot(1).inputs(2,:),'r')
plot(robot(2).inputs(2,:),'g')
plot(robot(3).inputs(2,:),'k')
title('Y velocities')
legend('1','2','3')

figure(3+num)
hold on
plot(robot(1).human_alphas,'r')
plot(robot(2).human_alphas,'g')
plot(robot(3).human_alphas,'k')
legend('1','2','3')

figure(4+num)
hold on
plot(robot(1).trust_factor,'r')
plot(robot(2).trust_factor,'g')
plot(robot(3).trust_factor,'k')
legend('1','2','3')

