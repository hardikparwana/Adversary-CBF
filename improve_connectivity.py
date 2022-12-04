import numpy as np
import matplotlib.pyplot as plt
from robot_models.SingleIntegrator2D import *
from robot_models.Unicycle import *
from robot_models.obstacles import *
from graph_utils import *

# Sim Parameters
tf = 6
dt = 0.05
num_steps = int(tf/dt)

# Plot                  
plt.ion()
fig = plt.figure()
ax = plt.axes(xlim=(0,7),ylim=(-0.5,3)) 
ax.set_xlabel("X")
ax.set_ylabel("Y")

robots = []
robots.append( SingleIntegrator2D(np.array([3,1.5]), dt, ax, id = 0, color='g',palpha=1.0 ) )
robots.append( SingleIntegrator2D(np.array([2.6,0]), dt, ax, id = 1, color='g',palpha=1.0 ) )
robots.append( SingleIntegrator2D(np.array([3.5,0]), dt, ax, id = 2, color='g',palpha=1.0 ) )
robots.append( SingleIntegrator2D(np.array([2.5,0.8]), dt, ax, id = 3, color='g',palpha=1.0 ) )
robots.append( SingleIntegrator2D(np.array([3.0,0.8]), dt, ax, id = 4, color='g',palpha=1.0 ) )
robots.append( SingleIntegrator2D(np.array([3.5,0.8]), dt, ax, id = 5, color='g',palpha=1.0 ) )

# Run sim
for i in range( num_steps ):
    
    # Leader motion
    uL = np.array([0.0,0.2])
    robots[0].step( uL )
    
    L = weighted_connectivity_undirected_laplacian(robots, max_dist = 2.0)
    Lambda, V = laplacian_eigen( L )
    lambda2_dx( robots, L, Lambda[1], V[:,1].reshape(-1,1) )
    
    for j in range( 1, len(robots) ):
        u = robots[j].lambda2_dx
        robots[j].step(u)
        # print(f" robot :{j}, input: {u} ")
    
    fig.canvas.draw()
    fig.canvas.flush_events()

plt.show()
















