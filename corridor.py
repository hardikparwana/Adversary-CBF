from tokenize import Single
import numpy as np
import time
import cvxpy as cp
import matplotlib.pyplot as plt
from robot_models.SingleIntegrator2D import *
from robot_models.Unicycle import *
from robot_models.obstacles import *
from utils.utils import *

from matplotlib.animation import FFMpegWriter

# Plot                  
plt.rcParams.update({'font.size': 27})
plt.ion()
fig = plt.figure()
ax = plt.axes()#(xlim=(0,7),ylim=(-0.5,8))
ax.set_xlabel("X")
ax.set_ylabel("Y")

# params
dt = 0.05
tf = 5.0
num_steps = int(tf/dt)

## Define robots
# agents
robots = []
num_robots = 3
robots.append( Unicycle(np.array([3,1.5,np.pi/2]), dt, ax, num_robots=num_robots, id = 0, color='g',palpha=1.0 ) )
robots.append( Unicycle(np.array([2.5,0,np.pi/2]), dt, ax, num_robots=num_robots, id = 1, color='g',palpha=1.0 ) )
robots.append( Unicycle(np.array([3.5,0,np.pi/2]), dt, ax, num_robots=num_robots, id = 2, color='g',palpha=1.0 ) )
robots.append( SingleIntegrator2D(np.array([-1,-1]), dt, ax, id = 3, color = 'r', palpha=1.0  ) )

adversaries = []
num_adversaries = 0

obstacles = []
obstacles.append( circle(0,0,1,ax,1) )

## 

for i in range(num_steps):
    
    for j in range(num_robots):
        
        # robot neighbors
        for k in range(num_robots):
            print("hello")
            
            
        # adversarial neighbors    
        for k in range(num_adversaries):
            print("hello")
            

plt.ioff()
plt.show()



