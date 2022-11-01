import numpy as np
from utils.utils import wrap_angle

class Unicycle:
    
    def __init__(self,X0,dt,ax,id,num_robots=1,num_adversaries = 1, num_obstacles = 0, alpha=0.8,color='r',palpha=1.0,plot=True, num_connectivity = 1):
        '''
        X0: iniytial state
        dt: simulation time step
        ax: plot axis handle
        id: robot id
        '''
        
        self.type = 'Unicycle'
        
        self.X = X0.reshape(-1,1)
        self.dt = dt
        self.id = id
        
        self.U = np.array([0,0]).reshape(-1,1)
        self.x_dot_nominal = np.array([ [0],[0],[0] ])
        self.U_ref = np.array([0,0]).reshape(-1,1)
        
        # Plot handles
        self.plot = plot
        if self.plot:
            self.body = ax.scatter([],[],alpha=palpha,s=40,facecolors='none',edgecolors=color) #,c=color
            self.radii = 0.5
            self.palpha = palpha
            if palpha==1:
                self.axis = ax.plot([self.X[0,0],self.X[0,0]+self.radii*np.cos(self.X[2,0])],[self.X[1,0],self.X[1,0]+self.radii*np.sin(self.X[2,0])])
            self.render_plot()
        
        # for Trust computation
        self.adv_alpha =  alpha*np.ones((1,num_adversaries))# alpha*np.ones((1,num_adversaries))
        self.trust_adv = np.ones((1,num_adversaries))
        self.obs_alpha =  alpha*np.ones((1,num_obstacles))#
        self.trust_obs = np.ones((1,num_obstacles))
        self.robot_alpha = alpha*np.ones((1,num_robots))
        self.trust_robot = np.ones((1,num_robots))
        self.adv_objective = [0] * num_adversaries
        self.robot_objective = [0] * num_robots
        self.obs_objective = [0] * num_obstacles
        self.robot_h = np.ones((1,num_robots))
        self.adv_h = np.ones((1,num_adversaries))
        self.obs_h = np.ones((1,num_obstacles))
        self.robot_connectivity_objective = 0
        self.robot_connectivity_alpha = alpha*np.ones((1,1))
        self.robot_connectivity_h = np.array([[1.0]])
        
        
        # Old
        # self.adv_alpha =  alpha*np.ones(num_adversaries)# alpha*np.ones((1,num_adversaries))
        # self.trust_adv = 1
        # self.robot_alpha = alpha*np.ones(num_robots)
        # self.trust_robot = 1
        # self.adv_objective = [0] * num_adversaries
        # self.robot_objective = [0] * num_robots
        
        num_constraints1  = num_robots - 1 + num_adversaries + num_obstacles + num_connectivity
        self.A1 = np.zeros((num_constraints1,2))
        self.b1 = np.zeros((num_constraints1,1))
        self.slack_constraint = np.zeros((num_constraints1,1))
        
        # For plotting
        self.adv_alphas = alpha*np.ones((1,num_adversaries))
        self.trust_advs = np.ones((1,num_adversaries))
        self.robot_alphas = alpha*np.ones((1,num_robots))
        self.trust_robots = 1*np.ones((1,num_robots))
        self.obs_alphas = alpha*np.ones((1,num_obstacles))
        self.trust_obss = 1*np.ones((1,num_obstacles))
        self.Xs = X0.reshape(-1,1)
        self.Us = np.array([0,0]).reshape(-1,1)
        self.adv_hs = np.ones((1,num_adversaries))
        self.robot_hs = np.ones((1,num_robots))
        self.obs_hs = np.ones((1,num_obstacles))
        self.robot_connectivity_alphas = np.ones((1,1))
        self.robot_connectivity_hs = np.ones((1,1))

     
    def f(self):
        return np.array([0,0,0]).reshape(-1,1)
    
    def g(self):
        return np.array([ [ np.cos(self.X[2,0]), 0 ],
                          [ np.sin(self.X[2,0]), 0],
                          [0, 1] ])       
         
    def step(self,U): 
        self.U = U.reshape(-1,1)
        self.X = self.X + ( self.f() + self.g() @ self.U )*self.dt
        self.X[2,0] = wrap_angle(self.X[2,0])
        self.Xs = np.append(self.Xs,self.X,axis=1)
        self.Us = np.append(self.Us,self.U,axis=1)
        return self.X
    
    def render_plot(self):
        if self.plot:
            x = np.array([self.X[0,0],self.X[1,0]])
            self.body.set_offsets([x[0],x[1]])
            if self.palpha==1:
                self.axis[0].set_ydata([self.X[1,0],self.X[1,0]+self.radii*np.sin(self.X[2,0])])
                self.axis[0].set_xdata( [self.X[0,0],self.X[0,0]+self.radii*np.cos(self.X[2,0])] )
        # self.axis = ax.plot([self.X[0,0],self.X[0,0]+np.cos(self.X[2,0])],[self.X[1,0],self.X[1,0]+np.sin(self.X[2,0])])
        
    def lyapunov(self, G):
        V = np.linalg.norm( self.X[0:2] - G[0:2] )**2
        dV_dx = np.append( 2*( self.X[0:2] - G[0:2] ).T, [[0]], axis=1)
        return V, dV_dx
    
    def nominal_input(self,G):
        # V, dV_dx = self.lyapunov(G)
        #Define gamma for the Lyapunov function
        k_omega = 2.0 #0.5#2.5
        k_v = 2.0 #0.5
        theta_d = np.arctan2(G.X[1,0]-self.X[1,0],G.X[0,0]-self.X[0,0])
        error_theta = wrap_angle( theta_d - self.X[2,0] )

        omega = k_omega*error_theta

        distance = max(np.linalg.norm( self.X[0:2,0]-G.X[0:2,0] ) - 0.3,0)

        v = k_v*( distance )*np.cos( error_theta )
        return np.array([v, omega]).reshape(-1,1) #np.array([v,omega])
    
    # def agent_barrier(self,agent,d_min):
    #     h = d_min**2 - np.linalg.norm(self.X[0:2] - agent.X[0:2])**2
    #     dh_dxi = np.append( -2*( self.X[0:2] - agent.X[0:2] ).T, [[0]], axis=1)
        
    #     if agent.type=='SingleIntegrator2D':
    #         dh_dxj = 2*( self.X[0:2] - agent.X[0:2] ).T
    #     elif agent.type=='Unicycle':
    #         dh_dxj = np.append( -2*( self.X[0:2] - agent.X[0:2] ).T, [[0]], axis=1 )
    #     elif agent.type=='FixedWing':
    #         print("TO DO here!!!!")
        
    #     return h, dh_dxi, dh_dxj
    
    def sigma(self,s):
        k1 = 2
        return (np.exp(k1-s)-1)/(np.exp(k1-s)+1)
    
    def sigma_der(self,s):
        k1 = 2
        return -np.exp(k1-s)/( 1+np.exp( k1-s ) ) * ( 1 + self.sigma(s) )
    
    def agent_barrier(self,agent,d_min):
        # beta = 1.01
        # h = beta*d_min**2 - np.linalg.norm(self.X[0:2] - agent.X[0:2])**2
        # h1 = h
        
        # theta = self.X[2,0]
        # # s = (self.X[0:2] - agent.X[0:2]).T @ np.array( [np.sin(theta),np.cos(theta)] ).reshape(-1,1)
        # s = (- self.X[0:2] + agent.X[0:2]).T @ np.array( [np.sin(theta),np.cos(theta)] ).reshape(-1,1)
        # h = h - self.sigma(s)
        # # print(f"h1:{h1}, h2:{h}")
        # # assert(h1<0)
        # der_sigma = self.sigma_der(s)
        # dh_dxi = np.append( -2*( self.X[0:2] - agent.X[0:2] ).T - der_sigma * ( np.array([ [np.sin(theta), np.cos(theta)] ]) ),  - der_sigma * ( np.cos(theta)*( self.X[0,0]-agent.X[0,0] ) - np.sin(theta)*( self.X[1,0] - agent.X[1,0] ) ) , axis=1)
        
        # if agent.type=='SingleIntegrator2D':
        #     # dh_dxj = 2*( self.X[0:2] - agent.X[0:2] ).T
        #     dh_dxj = 2*( self.X[0:2] - agent.X[0:2] ).T + der_sigma * ( np.array([ [np.sin(theta), np.cos(theta)] ]) )
        # elif agent.type=='Unicycle':
        #     # dh_dxj = np.append( -2*( self.X[0:2] - agent.X[0:2] ).T + der_sigma * ( np.array([ [np.sin(theta), np.cos(theta)] ]) ), np.array([[0]]), axis=1 )
        #     dh_dxj = np.append( 2*( self.X[0:2] - agent.X[0:2] ).T + der_sigma * ( np.array([ [np.sin(theta), np.cos(theta)] ]) ), np.array([[0]]), axis=1 )
        # else:
        #     dh_dxj = 2*( self.X[0:2] - agent.X[0:2] ).T
 
        beta = 1.01
        h = np.linalg.norm(self.X[0:2] - agent.X[0:2])**2 - beta*d_min**2
        h1 = h
        
        theta = self.X[2,0]
        # s = (self.X[0:2] - agent.X[0:2]).T @ np.array( [np.sin(theta),np.cos(theta)] ).reshape(-1,1)
        s = ( self.X[0:2] - agent.X[0:2]).T @ np.array( [np.sin(theta),np.cos(theta)] ).reshape(-1,1)
        h = h - self.sigma(s)
        # print(f"h1:{h1}, h2:{h}")
        # assert(h1<0)
        der_sigma = self.sigma_der(s)
        dh_dxi = np.append( 2*( self.X[0:2] - agent.X[0:2] ).T - der_sigma * ( np.array([ [np.sin(theta), np.cos(theta)] ]) ),  - der_sigma * ( np.cos(theta)*( self.X[0,0]-agent.X[0,0] ) - np.sin(theta)*( self.X[1,0] - agent.X[1,0] ) ) , axis=1)
        
        if agent.type=='SingleIntegrator2D':
            # dh_dxj = 2*( self.X[0:2] - agent.X[0:2] ).T
            dh_dxj = -2*( self.X[0:2] - agent.X[0:2] ).T + der_sigma * ( np.array([ [np.sin(theta), np.cos(theta)] ]) )
        elif agent.type=='Unicycle':
            # dh_dxj = np.append( -2*( self.X[0:2] - agent.X[0:2] ).T + der_sigma * ( np.array([ [np.sin(theta), np.cos(theta)] ]) ), np.array([[0]]), axis=1 )
            dh_dxj = np.append( -2*( self.X[0:2] - agent.X[0:2] ).T + der_sigma * ( np.array([ [np.sin(theta), np.cos(theta)] ]) ), np.array([[0]]), axis=1 )
        else:
            dh_dxj = -2*( self.X[0:2] - agent.X[0:2] ).T           
        
        
        return -h, -dh_dxi, -dh_dxj
    
    def connectivity_barrier( self, agent, d_max ):
        h = np.linalg.norm( self.X[0:2] - agent.X[0:2] )**2 - d_max**2
        
        dh_dxi = np.append( 2*( self.X[0:2] - agent.X[0:2] ).T, np.array([[0]]), axis = 1 )
        if agent.type == 'SingleIntegrator2D':
            dh_dxj = -2*( self.X[0:2] - agent.X[0:2] ).T
        elif agent.type == 'Unicycle':
            dh_dxj = np.append( -2*( self.X[0:2] - agent.X[0:2] ).T, np.array([[0]]), axis = 1 )
        else:
            dh_dxj = -2*( self.X[0:2] - agent.X[0:2] ).T
        return h.reshape(-1,1), dh_dxi, dh_dxj        
    