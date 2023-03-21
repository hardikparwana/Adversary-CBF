import numpy as np
import cvxpy as cp  
import matplotlib.pyplot as plt



u = cp.Variable((2,1))
delta = cp.Variable()
lfh = cp.Parameter((1,1))
lgh = cp.Parameter((1,2))
lfV = cp.Parameter((1,1))
lgV = cp.Parameter((1,2))
h = cp.Parameter()
V = cp.Parameter()
alpha = 0.5
k = 0.3
const = [ lfh + lgh @ u >= -alpha * h ]
const += [ lfV + lgV @ u<= -k * V + delta]
objective = cp.Minimize( cp.sum_squares( u ) + 1000*cp.sum_squares(delta) )
problem = cp.Problem( objective, const )



class robot:
    
    def __init__(self, x0, dt, ax):
        
        self.X = x0
        self.dt = dt
        self.body = ax.scatter([],[],c='r',s=10)
        self.render()
        
    def f(self):
        return np.array([0,0]).reshape(-1,1)
        
    def g(self):
        return np.array([ [1, 0],
                         [0, 1]   
                         ])
        
    def barrier(self, obs):
        d_min = 0.3
        h = np.linalg.norm( self.X - obs )**2 - d_min**2
        dh_dx = (self.X - obs).T
        return h, dh_dx

    def lyapunov(self, goal):
        V = np.linalg.norm( self.X - goal )**2
        dV_dx = (self.X - goal).T
        return V, dV_dx
                
    def step(self, U):
        self.X = self.X + ( self.f() + self.g() @ U )*self.dt
        self.render()
        
    def render(self):
        self.body.set_offsets( [ self.X[0,0], self.X[1,0] ] )
        
      
plt.ion() # interactive mode ON
fig = plt.figure()
ax = plt.axes(xlim=(-5,5),ylim=(-5,5)) 
ax.set_xlabel("X")
ax.set_ylabel("Y")
  
circ = plt.Circle((1,1),0.3,linewidth = 1, edgecolor='k',facecolor='k')
ax.add_patch(circ)
  
initial_location = np.array([0,0]).reshape(-1,1)
dt  = 0.05
my_robot = robot( initial_location, dt, ax )

obs = np.array([1,1]).reshape(-1,1)
goal = np.array([2.5,2.5]).reshape(-1,1)

for i in range(100):
    
    #U = np.array([ 2*np.cos( np.pi/6 ), 2*np.sin(np.pi/6) ]).reshape(-1,1)
    # my_robot.step(U)
    
    h.value, dh_dx = my_robot.barrier(obs)
    V.value, dV_dx = my_robot.lyapunov(goal)
    
    lfh.value = dh_dx @ my_robot.f()
    lgh.value = dh_dx @ my_robot.g()
    lfV.value = dV_dx @ my_robot.f()
    lgV.value = dV_dx @ my_robot.g()
    
    problem.solve()
    if problem.status != 'optimal':
        print("QP not solvable")
        
    my_robot.step(u.value)
    
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    
plt.show()
    
