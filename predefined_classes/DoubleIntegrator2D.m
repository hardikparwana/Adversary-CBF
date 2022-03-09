classdef DoubleIntegrator2D %<handle
    
   properties(Access = public)
      
       %Data
       id = 1;
       X = [0;0];
       Xt = [];
       G;
       colors = ['r','k','m'];
       
       % Dynamcs matrices for x_dot = f(x) + g(x)u 
       f = [0 0 1 0;
           0 0 0 1;
           0 0 0 0;
           0 0 0 0];
       g = [0 0;0 0;1 0;0 1];
       
       observed_data = [];
       input_data = [];
       predicted_data = [];
       predicted_std = [];
       predicted_normal_data = [];
       predicted_normal_std = [];
       inputs = [];

       gp_x;
       gp_y;
       
       %plots
       color_force = 0;
       
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
   end
   
   properties(Access = private)
        iter = 0;
        p1; % scatter plot
        p2;
   end
   
   methods(Access = public)
      
       function robot = DoubleIntegrator2D(ID,x,y,vx,vy)
          
           
           robot.X = [x;y;vx;vy];
           robot.id = ID;
           robot = plot_update(robot);            
       end
       
       function d = plot_update(d)
           
           center = [d.X(1) d.X(2)];
%            radius = d.safe_dist;
           d.Xt = [d.Xt;center ];
           
           if d.iter<1
               d.p1 = scatter(d.X(1),d.X(2),50,'r','filled');
               d.p2 = plot( d.Xt(:,1),d.Xt(:,2) );
               d.iter = 1;
           else
               set(d.p1,'XData',d.X(1),'YData',d.X(2));
               if (d.color_force==0)
                   set(d.p2,'Color',d.colors(d.id))
                   set(d.p2,'XData',d.Xt(:,1),'YData',d.Xt(:,2));
               else
                   set(d.p2,'Color','g')
                   set(d.p2,'XData',d.Xt(:,1),'YData',d.Xt(:,2));
               end
           end
           
       end
       
       function d = control_state(d,U,dt)
                
                % Euler update with Dynamics                
                d.X = d.X + d.f * d.X + d.g * [ U(1); U(2)]*dt;
                d = plot_update(d);
%                 out = d.X;
                
                d.f = [0 0 1 0;
                       0 0 0 1;
                       0 0 0 0;
                       0 0 0 0];
                d.g = [0 0;0 0;1 0;0 1];
            
       end
        
       function [h, h_dot_i, h_dot_j, h_ddot, h_ddot_i, h_ddot_j] = agent_barrier(d,agent)
                
                global d_min
                %barrier
                h = d_min^2 - norm(d.X(1:2)-agent.X(1:2))^2;                
                h_dot_i = -2*( d.X(1:2) - agent.X(1:2) )' * ( d.X(3:4) );
                h_dot_j = -2*( d.X(1:2) - agent.X(1:2) )' * (- agent.X(3:4) );
                h_ddot = -2*( d.X(3:4) - agent.X(3:4) )' * ( d.X(3:4) - agent.X(3:4) );
                h_ddot_i = -2*( d.X(1:2) - agent.X(1:2) )';
                h_ddot_j = 2*( d.X(1:2) - agent.X(1:2) )';                
       end
       
       function [V, dV_dx, dV_dx_i] = goal_lyapunov(d)
               %V = (x-x_d)^2 + v^Tv
                % Lyapunov
                V = norm(d.X(1:2)-d.G)^2 + norm(d.X(3:4))^2;
                dV_dx = 2*(d.X(1:2)-d.G)'*d.X(3:4);  % 0 because robot state is x,y,theta
                dV_dx_i = 2*d.X(3:4)';
                
       end
       
       function [h, h_dot_i, h_ddot, h_ddot_i] = obstacle_barrier(d,Obs)
                             
                % Simple barrier function: DOES NOT work for Unicycle
                h = (Obs.length)^2 - norm(d.X(1:2)-Obs.X)^2;
                
                h_dot_i = -2*( d.X(1:2) - Obs.X(1:2) )' * ( d.X(3:4) );
                h_ddot = -2*( d.X(3:4) )' * ( d.X(3:4) );
                h_ddot_i = -2*( d.X(1:2) - Obs.X(1:2) )';
       end
       
       
       
   end
    
end