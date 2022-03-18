classdef SingleIntegrator2D %<handle
    
   properties(Access = public)
      
       %Data
       id = 1;
       X = [0;0];
       Xt = [];
       G;
       colors = ['r','k','m'];
       
       % Dynamcs matrices for x_dot = f(x) + g(x)u 
       f = [0;0];
       g = [1 0;0 1];
       
       observed_data = [];
       input_data = [];
       predicted_data = [];
       predicted_std = [];
       predicted_normal_data = [];
       predicted_normal_std = [];
       inputs = [0;0];

       gp_x;
       gp_y;
       
       %plots
       color_force = 0;
       status = '';
       
       %input history
       
       % STL monitors
       sum_h = [0];
       h = [0];
       V = [0];
       upper_bound = [0]
       dh_dx = [0 0];
       dV_dx = [0 0];
       inner_prod = [0];
       dh_dx_complete = [0 0 0 0];
       dV_dx_complete = [0 0 0 0];
       inner_prod_complete = [0];
       
       obs_1_h = [0];
       obs_2_h = [0];
       obs_1_upper_bound = [0];
       obs_2_upper_bound = [0];
       obs_1_sum_h = [0];
       obs_2_sum_h = [0];
       obs_1_dh_dx = [0 0];
       obs_2_dh_dx = [0 0];
       obs_1_inner_prod = [0];
       obs_2_inner_prod = [0];
       
       trust_robots = [];
       trust_humans = [];
   end
   
   properties(Access = private)
        iter = 0;
        p1; % scatter plot
        p2;
   end
   
   methods(Access = public)
      
       function robot = SingleIntegrator2D(ID,x,y,status)
          
           
           robot.X = [x;y];
           robot.id = ID;
           robot.status = status;
           robot = plot_update(robot);            
       end
       
       function d = plot_update(d)
           
           center = [d.X(1) d.X(2)];
%            radius = d.safe_dist;
           d.Xt = [d.Xt;center ];
           
%            if d.iter<1
%                if d.color_force==0
%                     d.p1 = scatter(d.X(1),d.X(2),50,'g','filled');
%                else
%                     d.p1 = scatter(d.X(1),d.X(2),50,d.colors(d.id),'filled');
%                end
%                d.p2 = plot( d.Xt(:,1),d.Xt(:,2) );
%                d.iter = 1;
%            else
%                set(d.p1,'XData',d.X(1),'YData',d.X(2));
%                if (d.color_force==0)
%                    set(d.p2,'Color',d.colors(d.id))
%                    set(d.p2,'XData',d.Xt(:,1),'YData',d.Xt(:,2));
%                else
%                    set(d.p2,'Color','g')
%                    set(d.p2,'XData',d.Xt(:,1),'YData',d.Xt(:,2));
%                end
%            end
           
           if d.iter<1
               if strcmp(d.status,'human')
                    d.p1 = scatter(d.X(1),d.X(2),50,'r','filled');
               elseif strcmp(d.status,'reactive')
                   d.p1 = scatter(d.X(1),d.X(2),50,'g','filled');
               elseif strcmp(d.status,'nominal')
                   d.p1 = scatter(d.X(1),d.X(2),50,'b','filled');
               end
               d.p2 = plot( d.Xt(:,1),d.Xt(:,2) );
               d.iter = 1;
           else
               set(d.p1,'XData',d.X(1),'YData',d.X(2));
               set(d.p2,'XData',d.Xt(:,1),'YData',d.Xt(:,2));
           end
           
       end
       
       function d = control_state(d,U,dt)
                
                % Euler update with Dynamics                
                d.X = d.X + [ U(1); U(2)]*dt;
                d = plot_update(d);
%                 out = d.X;
                
                d.f = [0;0];
                d.g = [1 0;0 1];
            
       end
        
       function [h, dh_dxi, dh_dxj] = agent_barrier(d,agent)
                
                global d_min
                %barrier
                h = d_min^2 - norm(d.X(1:2)-agent.X(1:2))^2;
                dh_dxi = [-2*(d.X(1:2)-agent.X(1:2))'];    % 0 because robot state is x,y,theta
                dh_dxj = [2*(d.X(1:2)-agent.X(1:2))'];                
                
       end
       
       function [V, dV_dx] = goal_lyapunov(d)
               
                % Lyapunov
                V = norm(d.X(1:2)-d.G)^2;
                dV_dx = [2*(d.X(1:2)-d.G)'];  % 0 because robot state is x,y,theta
                
       end
       
       function [V, dV_dx] = lyapunov(d,G)
               
                % Lyapunov
                V = norm(d.X(1:2)-G)^2;
                dV_dx = [2*(d.X(1:2)-G)'];  % 0 because robot state is x,y,theta
                
       end
       
       function [h, dh_dx] = obstacle_barrier(d,Obs)
                             
                % Simple barrier function: DOES NOT work for Unicycle
                h = (Obs.length)^2 - norm(d.X-Obs.X)^2;
                dh_dx = [-2*(d.X-Obs.X)'];
       end
       
       
       
   end
    
end