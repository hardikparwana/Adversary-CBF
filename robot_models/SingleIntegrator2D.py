import numpy as np

class SingleIntegrator2D:
    
    def __init__(self,X0,dt,ax,id=0,num_robots=1,num_adversaries = 1, alpha=0.8,color='r',palpha=1.0,plot=True):
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
        self.color = color
        self.palpha = palpha

        self.U = np.array([0,0]).reshape(-1,1)
        self.U_nominal = np.array([0,0]).reshape(-1,1)
        self.nextU = self.U

        # Plot handles
        self.plot = plot
        if self.plot:
            self.body = ax.scatter([],[],c=color,alpha=palpha,s=10)
            self.render_plot()
        
         # for Trust computation
        self.adv_alpha =  alpha*np.ones((1,num_adversaries))# alpha*np.ones((1,num_adversaries))
        self.trust_adv = np.ones((1,num_adversaries))
        self.robot_alpha = alpha*np.ones((1,num_robots))
        self.trust_robot = np.ones((1,num_robots))
        self.adv_objective = [0] * num_adversaries
        self.robot_objective = [0] * num_robots
        self.robot_h = np.ones((1,num_robots))
        self.adv_h = np.ones((1,num_adversaries))        
        
        # Old
        # for Trust computation
        # self.adv_alpha = alpha*np.ones(num_adversaries)
        # self.trust_adv = 1
        # self.robot_alpha = alpha*np.ones(num_robots)
        # self.trust_robot = 1
        # self.adv_objective = [0] * num_adversaries
        # self.robot_objective = [0] * num_robots
        
        num_constraints1  = num_robots - 1 + num_adversaries
        self.A1 = np.zeros((num_constraints1,2))
        self.b1 = np.zeros((num_constraints1,1))
        
        # For plotting
        self.adv_alphas = alpha*np.ones((1,num_adversaries))
        self.trust_advs = np.ones((1,num_adversaries))
        self.robot_alphas = alpha*np.ones((1,num_robots))
        self.trust_robots = 1*np.ones((1,num_robots))
        self.Xs = X0.reshape(-1,1)
        self.Us = np.array([0,0]).reshape(-1,1)
        self.adv_hs = np.ones((1,num_adversaries))
        self.robot_hs = np.ones((1,num_robots))
        
        
    def f(self):
        return np.array([0,0]).reshape(-1,1)
    
    def g(self):
        return np.array([ [1, 0],[0, 1] ])
        
    def step(self,U): #Just holonomic X,T acceleration

        self.U = U.reshape(-1,1)
        self.X = self.X + ( self.f() + self.g() @ self.U )*self.dt
        self.render_plot()
        self.Xs = np.append(self.Xs,self.X,axis=1)
        self.Us = np.append(self.Us,self.U,axis=1)
        return self.X

    def render_plot(self):
        if self.plot:
            x = np.array([self.X[0,0],self.X[1,0]])

            # scatter plot update
            self.body.set_offsets([x[0],x[1]])

    def lyapunov(self, G):
        V = np.linalg.norm( self.X - G[0:2] )**2
        dV_dx = 2*( self.X - G[0:2] ).T
        return V, dV_dx
    
    def agent_barrier(self,agent,d_min):
        h = d_min**2 - np.linalg.norm(self.X - agent.X[0:2])**2
        dh_dxi = -2*( self.X - agent.X[0:2] ).T
        
        if agent.type=='SingleIntegrator2D':
            dh_dxj = 2*( self.X - agent.X[0:2] ).T
        elif agent.type=='Unicycle':
            dh_dxj = np.append( -2*( self.X - agent.X[0:2] ).T, [[0]], axis=1 )
        return h, dh_dxi, dh_dxj
    