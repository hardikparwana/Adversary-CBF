import numpy as np
from utils import wrap_angle

class Unicycle:
    
    def __init__(self,X0,dt,ax,id,num_robots=1,alpha=0.8):
        '''
        X0: iniytial state
        dt: simulation time step
        ax: plot axis handle
        id: robot id
        '''
        
        self.type = 'Unicycle'
        
        self.X = X0.reahpe(-1,1)
        self.dt = dt
        
        self.U = np.array([0,0]).reshape(-1,1)
        self.U_nominal = np.array([0,0]).reshape(-1,1)
        self.U = np.array([0,0]).reshape(-1,1)
        
        # Plot handles
        self.body = ax.scatter([],[],c=('r' if self.id==0 else 'g'),s=10)
        self.render_plot()
        
        # for Trust computation
        alpha = 0.8
        self.adv_alpha = [alpha]
        self.trust_adv = 1
        self.robot_alpha = alpha*np.ones(num_robots)
        self.trust_robot = 1
     
    def f(self):
        return np.array([0,0]).reshape(-1,1)
    
    def g(self):
        return np.array([ [ np.cos(self.X[2,0]), 0 ],
                          [ np.sin(self.X[0,0]), 0],
                          [0, 1] ])       
         
    def step(self,U): 
        self.U = U.reshape(-1,1)
        self.X = self.X + ( self.f() + self.g() @ self.U )*self.dt
        self.X[2,0] = wrap_angle(self.X[2,0])
        return self.X
    
    def render_plot(self):
        x = np.array([self.X[0,0],self.X[1,0]])
        self.body.set_offsets([x[0],x[1]])
        
    def lyapunov(self, G):
        V = np.linalg.norm( self.X - G )**2
        dV_dx = np.append( 2*( self.X - G ).T, [[0]], axis=1)
        return V, dV_dx
    
    def agent_barrier(self,agent,d_min):
        h = d_min**2 - np.linalg.norm(self.X[0:2] - agent.X[0:2])**2
        dh_dxi = np.append( -2*( self.X[0:2] - agent.X[0:2] ).T, [[0]], axis=1)
        
        if agent.type=='SingleIntegrator2D':
            dh_dxj = 2*( self.X[0:2] - agent.X[0:2] ).T
        elif agent.type=='Unicycle':
            dh_dxj = np.append( -2*( self.X[0:2] - agent.X[0:2] ).T, [[0]], axis=1 )
        elif agent.type=='FixedWing':
            print("TO DO here!!!!")
        
        return h, dh_dxi, dh_dxj
    
    def sigma(self,s):
        return (np.exp(1-s)-1)/(np.exp+1)
    
    def sigma_der(self,s):
        return -np.exp(1-s)/( 1+np.exp( 1-s ) ) * ( 1 + self.sigma(s) )
    
    def agent_barrier2(self,agent,d_min):
        beta = 1.1
        h = beta*d_min**2 - np.linalg.norm(self.X[0:2] - agent.X[0:2])**2
        theta = self.X[2,0]
        s = (self.X[0:2] - agent.X[0:2]).T @ np.array( [np.sin(theta),np.cos(theta)] ).reshape(-1,1)
        h = h - self.sigma(s)
        der_sigma = self.sigma_der(s)
        dh_dxi = np.append( -2*( self.X[0:2] - agent.X[0:2] ).T - der_sigma * ( np.array([ [np.sin(theta), np.cos(theta)] ]) ), [[ np.cos(theta)*( self.X[0,0]-agent.X[0,0] ) + np.cos(theta)*( self.X[1,0] - agent.X[1,0] ) ]], axis=1)
        
        if agent.type=='SingleIntegrator2D':
            dh_dxj = 2*( self.X[0:2] - agent.X[0:2] ).T
        elif agent.type=='Unicycle':
            dh_dxj = np.append( -2*( self.X[0:2] - agent.X[0:2] ).T + der_sigma * ( np.array([ [np.sin(theta), np.cos(theta)] ]) ), [[0]], axis=1 )
        
        return h, dh_dxi, dh_dxj