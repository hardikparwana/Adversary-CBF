classdef SingleIntegrator2D <handle
    
   properties(Access = public)
      
       %Data
       id = 1;
       X = [0;0];
       
   end
   
   properties(Access = private)
        iter = 0;
        p1; % scatter plot
   end
   
   methods(Access = public)
      
       function robot = SingleIntegrator2D(ID,x,y)
          
           robot.X = [x;y];
           robot.id = ID;
           robot = plot_update(robot); 
           
       end
       
       function d = plot_update(d)
           
           if d.iter<1
               d.p1 = scatter(d.X(1),d.X(2),50,color,'filled');
               d.iter = 1;
           else
               set(d.p1,'XData',d.X(1),'YData',d.X(2));
           end
           
       end
       
       function out = control_state(d,U,dt)
                
                % Euler update with Dynamics                
                d.X = d.X + [ U(1); U(2)]*dt;
                d = plot_update(d);
                out = d.X;
            
        end
       
       
       
   end
    
end