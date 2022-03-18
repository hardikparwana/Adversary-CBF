% axis equal
set(gca,'DataAspectRatio',[1 1 1])
hold on

% Target Safe Regions
reg(1) = EnvObject2D(1,1,'rectangle','green',2,2);
reg(2) = EnvObject2D(9,1,'rectangle','green',2,2);
reg(3) = EnvObject2D(1,9,'rectangle','green',2,2);
reg(4) = EnvObject2D(9,9,'rectangle','green',2,2);

% Obstacles: unsafe regions
obs(1) = EnvObject2D(5,5,'rectangle','black',6,6);

% Now create occupancy grid
%occupancy_grid;