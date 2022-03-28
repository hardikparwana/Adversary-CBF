from tkinter import Variable
import numpy as np
import time
import cvxpy as cp
import matplotlib.pyplot as plt
from robot_models.SingleIntegrator2D import *
from robot_models.Unicycle import *
from robot_models.obstacles import *
from utils.utils import *

# Sim Parameters                  
dt = 0.05
tf = 10
num_steps = int(tf/dt)
t = 0

d_min = 0.2

# Plot                  
plt.ion()
fig = plt.figure()
ax = plt.axes(xlim=(0,10),ylim=(-10,10))
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_aspect(1)

d_min = 3.0
robot =  Unicycle(np.array([0,0,np.pi/2]), dt, ax, id = 0, color='g' ) 
goal = circle(8,8,1,ax,1)

Obs = circle(5,5,d_min,ax,0)

for i in range(num_steps):
    
    # leader
    # u = np.array([1,1])
    # leader.step(u)
    # leader.render_plot()
    
    # Follower
    u = cp.Variable((2,1))
    delta = cp.Variable(1)
    
    h, dh_dxj, dh_dxk = robot.agent_barrier(Obs,d_min)
    const = [ dh_dxj @ ( robot.f() + robot.g() @ u )  <= -30000.0 * h ]
    
    v, w = robot.nominal_input(goal)
    u_ref = np.array([v,w]).reshape(-1,1)
    
    objective = cp.Minimize( cp.sum_squares(u - u_ref ) )
    problem = cp.Problem(objective, const)
    problem.solve()
    print(f"v:{v}, w:{w}, status: {problem.status}, U:{u.value}")
    robot.step(u.value)
    robot.render_plot()
    
    fig.canvas.draw()
    fig.canvas.flush_events()

