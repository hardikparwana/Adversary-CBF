import numpy as np
from robot_models.Unicycle import *
from robot_models.obstacles import *
from utils.utils import *

dt = 0.05
tf = 6.0 #5.4#8#4.1 #0.2#4.1
num_steps = int(tf/dt)
t = 0
d_min_obstacles = 1.0 #0.1

plt.ion()
fig = plt.figure()
ax = plt.axes(xlim=(0,7),ylim=(-0.5,8))
ax.set_xlabel("X")
ax.set_ylabel("Y")

robot = Unicycle(np.array([3,1.5,np.pi/2]), dt, ax, num_robots=num_robots, id = 0, color='g',palpha=1.0, alpha=alpha_cbf, num_adversaries=num_adversaries, num_obstacles=num_obstacles )

obstacle = circle( 1.8,2.5,1.0,ax,0 )