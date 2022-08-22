import numpy as np
import torch
from utils.utils import wrap_angle, wrap_angle_tensor
import matplotlib.patches as mpatches

class Unicycle:
    
    def __init__(self,X0,dt,ax,id,num_robots=1, min_D = 0.3, max_D = 2.0, FoV_angle = np.pi/2, num_adversaries = 1, alpha=0.8, k = 10.0, color='r',palpha=1.0,plot=True, identity='nominal', num_alpha = 1, predict_function = None):
        '''
        X0: iniytial state
        dt: simulation time step
        ax: plot axis handle
        id: robot id
        '''
        
        self.type = 'Unicycle'
        self.identity = identity # or adversary
        
        self.X = X0.reshape(-1,1)
        self.X_nominal = np.copy(self.X)
        self.dt = dt
        self.id = id
        
        self.X_torch = []
        
        self.U = np.array([0,0]).reshape(-1,1)
        self.x_dot_nominal = np.array([ [0],[0],[0] ])
        self.U_ref = np.array([0,0]).reshape(-1,1)
        self.U_ref_nominal = np.copy(self.U)
        
        self.FoV_angle = FoV_angle #np.pi/2
        self.FoV_length = max_D  #3.0
        self.max_D = max_D #3.0
        self.min_D = min_D #0.2
        self.predict_function = predict_function
        
        # Plot handles
        self.plot = plot
        if self.plot:
            self.body = ax.scatter([],[],alpha=palpha,s=40,facecolors='none',edgecolors=color) #,c=color
            self.radii = 0.5
            self.palpha = palpha
            if palpha==1:
                self.axis = ax.plot([self.X[0,0],self.X[0,0]+self.radii*np.cos(self.X[2,0])],[self.X[1,0],self.X[1,0]+self.radii*np.sin(self.X[2,0])])
            self.render_plot()
            
            # self.body_nominal = ax.scatter([],[],alpha=0.1,s=40,facecolors='none',edgecolors=color) #,c=color
            # self.render_plot(mode='nominal')

            self.lines, = ax.plot([],[],'o-')
            self.poly = mpatches.Polygon([(0,0.2)], closed=True, color='r',alpha=0.1, linewidth=0) #[] is Nx2
            self.fov_arc = ax.add_patch(self.poly)
            self.areas, = ax.fill([],[],'r',alpha=0.1)
            self.body = ax.scatter([],[],c=color,s=10)            
            self.des_point = ax.scatter([],[],s=10, facecolors='none', edgecolors='r')
            
            self.render_plot_fov()

        # self.alpha = alpha*np.ones((num_robots,1))
        self.alpha = alpha*np.ones((num_alpha,1))
        self.alpha_torch = torch.tensor(alpha, dtype=torch.float, requires_grad=True)
        
        self.k = k
        self.k_torch = torch.tensor( k, dtype=torch.float, requires_grad=True )
        
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
        # self.adv_alpha =  alpha*np.ones(num_adversaries)# alpha*np.ones((1,num_adversaries))
        # self.trust_adv = 1
        # self.robot_alpha = alpha*np.ones(num_robots)
        # self.trust_robot = 1
        # self.adv_objective = [0] * num_adversaries
        # self.robot_objective = [0] * num_robots
        
        num_constraints1  = num_robots - 1 + num_adversaries
        self.A1 = np.zeros((num_constraints1,2))
        self.b1 = np.zeros((num_constraints1,1))
        self.A2 = np.zeros((num_constraints1,2))
        self.b2 = np.zeros((num_constraints1,1))
        
        # For plotting
        self.alphas = np.copy(self.alpha)
        self.ks = np.copy(self.k)
        self.adv_alphas = alpha*np.ones((1,num_adversaries))
        self.trust_advs = np.ones((1,num_adversaries))
        self.robot_alphas = alpha*np.ones((1,num_robots))
        self.trust_robots = 1*np.ones((1,num_robots))
        self.adv_hs = np.ones((1,num_adversaries))
        self.robot_hs = np.ones((1,num_robots))
        
        ## Store state
        self.X_org = np.copy(self.X)
        self.U_org = np.copy(self.U)
        
        self.Xs = [] # np.copy(self.X)
        self.Xdots = [] #np.array([0,0,0]).reshape(-1,1)
        # self.Xs = X0.reshape(-1,1)
        self.Us = [] #np.array([0,0]).reshape(-1,1)
        gp = []
        
    def f_torch(self,x):
        return torch.tensor(np.array([0.0,0.0,0.0]).reshape(-1,1),dtype=torch.float)

    def g_torch(self,x):
        # return torch.tensor([ [torch.cos(x[2,0]),0.0],[torch.sin(x[2,0]),0.0],[0,1] ])
        g1 = torch.cat( ( torch.cos(x[2,0]).reshape(-1,1),torch.tensor([[0]]) ), axis=1 )
        g2 = torch.cat( ( torch.tensor([[0]]), torch.sin(x[2,0]).reshape(-1,1) ), axis=1 )
        g3 = torch.tensor([[0,1]],dtype=torch.float)
        gx = torch.cat((g1,g2,g3))
        return gx
    
    def f(self, mode='actual'):
        if mode=='actual':
            return np.array([0,0,0]).reshape(-1,1)
        if mode=='nominal':
            return np.array([0,0,0]).reshape(-1,1)
    
    def g(self, mode='actual'):
        if mode=='actual':
            return np.array([ [ np.cos(self.X[2,0]), 0 ],
                          [ np.sin(self.X[2,0]), 0],
                          [0, 1] ])    
        if mode=='nominal':
            return np.array([ [ np.cos(self.X_nominal[2,0]), 0 ],
                          [ np.sin(self.X_nominal[2,0]), 0],
                          [0, 1] ]) 
            
         
    def step(self,U, dt, mode='actual'): 
        if mode=='actual':
            
            xold = np.copy(self.X)
            
            self.U = U.reshape(-1,1)
            self.X = self.X + ( self.f() + self.g() @ self.U ) * dt
            Xdot = self.f() + self.g() @ self.U 
            self.X[2,0] = wrap_angle(self.X[2,0])
            
            if self.Xs == []:
                self.Xs = np.copy(xold)
                self.Us = np.copy(self.U)
                self.Xdots = np.copy(Xdot)
            else:            
                self.Xs = np.append(self.Xs,xold,axis=1)
                self.Us = np.append(self.Us,self.U,axis=1)
                self.Xdots = np.append( self.Xdots, Xdot , axis=1 )
            
            self.render_plot()
            self.render_plot_fov()
            
            return self.X
        if mode=='nominal':
            self.U_nominal = U.reshape(-1,1)
            self.X_nominal = self.X_nominal + ( self.f(mode='nominal') + self.g(mode='nominal') @ self.U_nominal ) * dt
            self.X_nominal[2,0] = wrap_angle(self.X_nominal[2,0])
            # self.Xs = np.append(self.Xs,self.X,axis=1)
            # self.Us = np.append(self.Us,self.U,axis=1)
            return self.X_nominal
    
    def step_torch(self, X, U, dt):
        return X + ( self.f_torch(X) + self.g_torch(X) @ U ) * dt
    
    def render_plot(self, mode='actual'):
        
        if self.plot:
            if mode=='actual':
                x = np.array([self.X[0,0],self.X[1,0]])
                self.body.set_offsets([x[0],x[1]])
                if self.palpha==1:
                    self.axis[0].set_ydata([self.X[1,0],self.X[1,0]+self.radii*np.sin(self.X[2,0])])
                    self.axis[0].set_xdata( [self.X[0,0],self.X[0,0]+self.radii*np.cos(self.X[2,0])] )
            elif mode=='nominal':
                x = np.array([self.X_nominal[0,0],self.X_nominal[1,0]])
                self.body_nominal.set_offsets([x[0],x[1]])
        # self.axis = ax.plot([self.X[0,0],self.X[0,0]+np.cos(self.X[2,0])],[self.X[1,0],self.X[1,0]+np.sin(self.X[2,0])])
        
    def render_plot_fov(self): #,lines,areas,body, poly, des_point):
        # length = 3
        # FoV = np.pi/3   # 60 degrees

        x = np.array([self.X[0,0],self.X[1,0]])
  
        theta = self.X[2][0]
        theta1 = theta + self.FoV_angle/2
        theta2 = theta - self.FoV_angle/2
        e1 = np.array([np.cos(theta1),np.sin(theta1)])
        e2 = np.array([np.cos(theta2),np.sin(theta2)])

        P1 = x + self.FoV_length*e1
        P2 = x + self.FoV_length*e2  

        des_dist = self.min_D + (self.max_D - self.min_D)/2
        des_x = np.array( [ self.X[0,0] + np.cos(theta)*des_dist, self.X[1,0] + np.sin(theta)*des_dist    ] )

        triangle_hx = [x[0] , P1[0], P2[0], x[0] ]
        triangle_hy = [x[1] , P1[1], P2[1], x[1] ]
        
        triangle_v = [ x,P1,P2,x ]  

        # lines.set_data(triangle_hx,triangle_hy)
        self.areas.set_xy(triangle_v)

        # scatter plot update
        self.body.set_offsets([x[0],x[1]])
        self.des_point.set_offsets([des_x[0], des_x[1]])

        #Fov arc
        self.poly.set_xy(self.arc_points(x, self.FoV_length, theta2, theta1))

        # return lines, areas, body, poly, des_point
        
    def arc_points(self, center, radius, theta1, theta2, resolution=50):
        # generate the points
        theta = np.linspace(theta1, theta2, resolution)
        points = np.vstack((radius*np.cos(theta) + center[0], 
                            radius*np.sin(theta) + center[1]))
        return points.T
    
    def lyapunov(self, X, G):
        avg_D = (self.min_D + self.max_D)/2.0
        V = ( np.linalg.norm( self.X[0:2] - G[0:2] ) - avg_D  )**2
        # dV_dx = np.append( 2*( self.X[0:2] - G[0:2] ).T, [[0]], axis=1)
        return V #, dV_dx
    
    def lyapunov_tensor(self, X, G):
        avg_D = (self.min_D + self.max_D)/2.0
        V = torch.square ( torch.norm( X[0:2] - G[0:2] ) - avg_D )
        
        factor = 2*(torch.norm( X[0:2]- G[0:2] ) - avg_D)/torch.norm( X[0:2] - G[0:2] ) * (  X[0:2] - G[0:2] ).reshape(1,-1) 
        dV_dxi = torch.cat( (factor, torch.tensor([[0]])), dim  = 1 )
        dV_dxj = -factor
        
        return V, dV_dxi, dV_dxj
    
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
    
    def nominal_input_tensor(self, X, targetX):
        #simple controller for now: considers estimated disturbance
        # print(X,target.X)
        #Define gamma for the Lyapunov function
        k_omega = 2.0 #0.5#2.5
        k_v = 2.0 #0.5
        diff = targetX[0:2,0] - X[0:2,0]

        theta_d = torch.atan2(targetX[1,0]-X[1,0],targetX[0,0]-X[0,0])
        error_theta = wrap_angle_tensor( theta_d - X[2,0] )

        omega = k_omega*error_theta 
        v = k_v*( torch.norm(diff) ) * torch.cos( error_theta )
        v = v.reshape(-1,1)
        omega = omega.reshape(-1,1)
        U = torch.cat((v,omega))
        return U
    
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
        k1 = 2.0
        return (np.exp(k1-s)-1)/(np.exp(k1-s)+1)
    
    def sigma_torch(self,s):
        k1 = 2.0
        return torch.div( (torch.exp(k1-s)-1) , (torch.exp(k1-s)+1) )
    
    def sigma_der(self,s):
        k1 = 2
        return -np.exp(k1-s)/( 1+np.exp( k1-s ) ) * ( 1 + self.sigma(s) )
    
    def sigma_der_torch(self,s):
        k1 = 2
        return -torch.div ( torch.exp(k1-s),( 1+torch.exp( k1-s ) ) ) @ ( 1 + self.sigma_torch(s) )
    
    def agent_barrier(self,agent):
        beta = 1.01
        h = beta*self.min_D**2 - np.linalg.norm(self.X[0:2] - agent.X[0:2])**2
        h1 = h
        
        theta = self.X[2,0]
        s = (self.X[0:2] - agent.X[0:2]).T @ np.array( [np.sin(theta),np.cos(theta)] ).reshape(-1,1)
        h = h - self.sigma(s)
        # print(f"h1:{h1}, h2:{h}")
        # assert(h1<0)
        der_sigma = self.sigma_der(s)
        dh_dxi = np.append( -2*( self.X[0:2] - agent.X[0:2] ).T - der_sigma * ( np.array([ [np.sin(theta), np.cos(theta)] ]) ),  - der_sigma * ( np.cos(theta)*( self.X[0,0]-agent.X[0,0] ) - np.sin(theta)*( self.X[1,0] - agent.X[1,0] ) ) , axis=1)
        
        
        if agent.type=='SingleIntegrator2D':
            dh_dxj = 2*( self.X[0:2] - agent.X[0:2] ).T # row
        elif agent.type=='Unicycle':
            dh_dxj = np.append( -2*( self.X[0:2] - agent.X[0:2] ).T + der_sigma * ( np.array([ [np.sin(theta), np.cos(theta)] ]) ), np.array([[0]]), axis=1 )
        else:
            dh_dxj = 2*( self.X[0:2] - agent.X[0:2] ).T
        
        return h, dh_dxi, dh_dxj
    
    def agent_barrier_torch(self, X, targetX, target_type='Unicycle'):
        beta = 1.01
        h = beta*self.min_D**2 -  torch.square( torch.norm(X[0:2] - targetX[0:2])  )
        h1 = h
        
        theta = X[2,0]
        s = (X[0:2] - targetX[0:2]).T @ torch.cat( ( torch.cos(X[2,0]).reshape(-1,1), torch.sin(X[2,0]).reshape(-1,1) ) )
        h_final = h - self.sigma_torch(s)
        # print(f"h1:{h1}, h2:{h}")
        # assert(h1<0)
        der_sigma = self.sigma_der_torch(s)
        dh_dxi =     torch.cat( ( -2*( X[0:2] - targetX[0:2] ).T - der_sigma @ torch.cat( (torch.sin(X[2,0]).reshape(-1,1), torch.cos(X[2,0]).reshape(-1,1)),1 ) ,  - der_sigma * ( torch.cos(X[2,0]).reshape(-1,1) @ ( X[0,0]-targetX[0,0] ).reshape(-1,1) - torch.sin(X[2,0]).reshape(-1,1) @ ( X[1,0] - targetX[1,0] ).reshape(-1,1) ) ), 1)
        
        if target_type=='SingleIntegrator2D':
            dh_dxj = 2*( X[0:2] - targetX[0:2] ).T # row
        elif target_type=='Unicycle':
            dh_dxj = torch.cat( ( -2*( X[0:2] - targetX[0:2] ).T + der_sigma @ torch.cat( (torch.sin(X[2,0]).reshape(-1,1), torch.cos(X[2,0]).reshape(-1,1)),1 ) , torch.tensor([[0]]) ) , 1)
        else:
            dh_dxj = 2*( X[0:2] - targetX[0:2] ).T
        
        return -h_final, -dh_dxi, -dh_dxj
    
    def agent_fov_barrier(self, X, targetX, target_type='SingleIntegrator2D'):
        
        # Max distance
        h1 = self.max_D**2 - torch.square( torch.norm( X[0:2] - targetX[0:2] ) )
        dh1_dxi = torch.cat( ( -2*( X[0:2] - targetX[0:2] ), torch.tensor([[0.0]]) ), 0).T
        dh1_dxj =  2*( X[0:2] - targetX[0:2] ).T
        
        # Min distance
        h2 = torch.square(torch.norm( X[0:2] - targetX[0:2] )) - self.min_D**2
        dh2_dxi = torch.cat( ( 2*( X[0:2] - targetX[0:2] ), torch.tensor([[0.0]]) ), 0).T
        dh2_dxj = - 2*( X[0:2] - targetX[0:2]).T
    
        # Max angle
        p = targetX[0:2] - X[0:2]

        # dir_vector = torch.tensor([[torch.cos(x[2,0])],[torch.sin(x[2,0])]]) # column vector
        dir_vector = torch.cat( ( torch.cos(X[2,0]).reshape(-1,1), torch.sin(X[2,0]).reshape(-1,1) ) )
        bearing_angle  = torch.matmul(dir_vector.T , p )/ torch.norm(p)
        h3 = (bearing_angle - np.cos(self.FoV_angle/2))/(1.0-np.cos(self.FoV_angle/2))

        norm_p = torch.norm(p)
        dh3_dx = dir_vector.T / norm_p - ( dir_vector.T @ p)  * p.T / torch.pow(norm_p,3)    
        dh3_dTheta = ( -torch.sin(X[2]) * p[0] + torch.cos(X[2]) * p[1] ).reshape(1,-1)  /torch.norm(p)
        dh3_dxi = torch.cat(  ( -dh3_dx , dh3_dTheta), 1  )
        dh3_dxj = dh3_dx
        
        return h1, dh1_dxi, dh1_dxj, h2, dh2_dxi, dh2_dxj, h3, dh3_dxi, dh3_dxj
    
    def compute_reward(self,X,targetX, des_d = 0.7):
        # return torch.square( torch.norm( X[0:2,0] - targetX[0:2,0]  ) - torch.tensor(des_d) )
    
        return torch.square( torch.norm( X[0:2,0] - targetX[0:2,0]  ) - torch.tensor((self.min_D+self.max_D)/2) )
    