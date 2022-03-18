
% axis equal
set(gca,'DataAspectRatio',[1 1 1])
hold on

% axis limit
xlim([0 9.5])

% Target Safe Regions
reg(1) = EnvObject2D(0.5,7.5,'rectangle','green',1,1);
reg(2) = EnvObject2D(0.75,3.25,'rectangle','green',0.5,1.5);

% Obstacles: unsafe regions
obs(1) = EnvObject2D(1.5,6.25,'rectangle','black',1,1.5);
obs(2) = EnvObject2D(1,4.2,'rectangle','black',2,0.4);
obs(3) = EnvObject2D(1.5,1.25,'rectangle','black',1,2.5);
obs(4) = EnvObject2D(3,4.5,'rectangle','black',0.2,6);
obs(5) = EnvObject2D(5.2,3.5,'rectangle','black',0.2,7);
obs(6) = EnvObject2D(7,4.5,'rectangle','black',1,1);
obs(7) = EnvObject2D(7.5,2.5,'rectangle','black',1,1);
obs(8) = EnvObject2D(9,4,'rectangle','black',1,1);

% Now create occupancy grid
%occupancy_grid;