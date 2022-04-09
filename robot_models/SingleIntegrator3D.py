import numpy as np

class SingleIntegrator3D:
    
    def __init__(self,X0,dt,ax,id=0,num_robots=1,alpha=0.8):
        '''
        X0: iniytial state
        dt: simulation time step
        ax: plot axis handle
        id: robot id
        '''
        
        self.type = 'SingleIntegrator2D'        
        
        X0 = X0.reshape(-1,1)
        self.X = X0
        self.dt = dt
        self.id = id

        self.U = np.array([0,0,0]).reshape(-1,1)
        self.U_nominal = np.array([0,0,0]).reshape(-1,1)

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
        return np.array([0,0,0]).reshape(-1,1)
    
    def g(self):
        return np.array([ [1, 0, 0],[0, 1, 0], [0, 0, 1] ])
        
    def step(self,U): #Just holonomic X,T acceleration

        self.U = U.reshape(-1,1)
        self.X = self.X + ( self.f() + self.g() @ self.U )*self.dt
        return self.X

    def render_plot(self):
        
        x = np.array([self.X[0,0],self.X[1,0],self.X[2,0]])

        # scatter plot update
        self.body.set_offsets([x[0],x[1],x[2]])

    def lyapunov(self, G):
        V = np.linalg.norm( self.X - G )**2
        dV_dx = 2*( self.X - G ).T
        return V, dV_dx
    
    def agent_barrier(self,agent,d_min):
        h = d_min**2 - np.linalg.norm(self.X - agent.X[0:2])**2
        dh_dxi = -2*( self.X - agent.X[0:3] ).T
        
        if agent.type=='SingleIntegrator3D':
            dh_dxj = 2*( self.X - agent.X[0:3] ).T
        elif agent.type=='FixedWing':
            dh_dxj = np.append( -2*( self.X - agent.X[0:3] ).T, [[0,0,0]], axis=1 )
        return h, dh_dxi, dh_dxj
    