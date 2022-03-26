import numpy as np
from utils import wrap_angle, euler_to_rot_mat, euler_rate_matrix

class FixedWing:
    
    def __init__(self,X0,dt,ax,id,num_robots=1,alpha=0.8):
        '''
        X0: iniytial state
        dt: simulation time step
        ax: plot axis handle
        id: robot id
        '''
        
        self.type = 'FixedWing'
        
        self.X = X0.reahpe(-1,1)
        self.dt = dt
        
        self.U = np.array([0,0,0,0]).reshape(-1,1) # linear velocity 1 component, angular velocity 3 components
        self.U_nominal = np.array([0,0,0,0]).reshape(-1,1)
        self.U = np.array([0,0]).reshape(-1,1)
        
        # Plot handles
        self.body = ax.scatter([],[],c=('k' if self.id==0 else 'm'),s=10)
        self.render_plot()
        
        # for Trust computation
        alpha = 0.8
        self.adv_alpha = [alpha]
        self.trust_adv = 1
        self.robot_alpha = alpha*np.ones(num_robots)
        self.trust_robot = 1
     
    def f(self):
        return np.array([0,0,0,0,0,0]).reshape(-1,1)
    
    def g(self):
        bx = euler_to_rot_mat( self.X[3,0], self.X[4,0], self.X[5,0] ) @ np.array([ [1],[0],[0] ]) # roll, pitch, yaw order
        rate_matrix = euler_rate_matrix( self.X[3,0], self.X[4,0], self.X[5,0] )
        return np.append( np.append(  bx, np.zeros((3,3)), axis = 1  ), np.append( np.zeros((3,1)), rate_matrix, axis=1  ), axis=0   )
         
    def step(self,U): 
        self.U = U.reshape(-1,1)
        self.X = self.X + ( self.f() + self.g() @ self.U )*self.dt
        self.X = wrap_angle(self.X[2,0])
        return self.X
    
    def render_plot(self):
        x = np.array([self.X[0,0],self.X[1,0],self.X[2,0]])
        self.body.set_offsets([x[0],x[1],x[2]])
          
    def lyapunov(self, G):
        V = np.linalg.norm( self.X - G )**2
        dV_dx = np.append( 2*( self.X - G ).T, [[0,0,0]], axis=1)
        return V, dV_dx
    
    def agent_barrier(self,agent,d_min):
        h = d_min**2 - np.linalg.norm(self.X[0:3] - agent.X[0:3])**2
        dh_dxi = np.append( -2*( self.X[0:3] - agent.X[0:3] ).T, [[0]], axis=1)
        
        if agent.type=='SingleIntegrator2D':
            dh_dxj = 2*( self.X[0:3] - agent.X[0:3] ).T
        elif agent.type=='Unicycle':
            dh_dxj = np.append( -2*( self.X[0:3] - agent.X[0:3] ).T, [[0,0,0]], axis=1 )
        elif agent.type=='FixedWing':
            print("TO DO here!!!!")
        
        return h, dh_dxi, dh_dxj
    
    
    
## 
# 
# ax = fig.add_subplot(111,projection='3d')


