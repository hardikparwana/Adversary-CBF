classdef EnvObject2D
   properties(Access = public)
    type = '';
    X = [0;0];
    length = 0;
    width = 0;
   end
   
   methods(Access = public)
       
       %Constructor
       function Obs = EnvObject2D(x,y,shape,lx,ly)  
               Obs.X = [x;y];
               Obs.type = shape; % circle, rectangle
               Obs.length = lx;
               Obs.width = ly;
               if strcmp(Obs.type,'circle')
                   make_circle(Obs);
               elseif strcmp(Obs.type,'rectangle')
                   make_rectangle(Obs);
               end
       end
       
       function make_circle(d)
               figure(1)
               hold on               
               center = [d.X(1) d.X(2)];
               radius = d.length;
               
               % Display the circles.
               h = viscircles(center,radius,'Color','k');
               xd = h.Children(1).XData(1:end-1); %leave out the nan
               yd = h.Children(1).YData(1:end-1);
               fill(xd, yd, 'k');
       end
           
       function make_rectangle(d)
              cx = d.X(1);
              cy = d.X(2);

              lx = d.length;
              ly = d.width;

              x = [cx+lx/2 cx-lx/2 cx-lx/2 cx+lx/2];
              y = [cy+ly/2 cy+ly/2 cy-ly/2 cy-ly/2];

              figure(1)
              hold on  
              patch( x,y,[0 0 0] );
       end
       
   end
   
   
end