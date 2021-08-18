classdef MatrixVariateGaussianProcess <handle
    
    % https://en.wikipedia.org/wiki/Matrix_normal_distribution
    % Refer to above website for basic MVG definitions and terms involved
    % This code updates the hyperparametrs by maximizing log likelihhod
    % probability (i.e., maximizing probability of observed training data )

    properties(Access = public)
         
        %Params
        omega;
        sigma;
        l;
        
        %Data
        X;      % input training data
        Y;      % output training data
        N_data; % size of data
        K_obs;  % store covariance matrix of training data
        
        noise = 0.1;
        
        %Randomly Sampled data for hyperpeter tuning
        Xs;
        Ys;
        Ns_data % size of sampled data. the parameter update is performed by randomly sampling a subset of training data
        
    end
    
    
    methods(Access = public)
        
        function obj = MatrixVariateGaussianProcess(omega,sigma,l)
                     
            obj.omega = omega;
            obj.sigma = sigma;
            obj.l = l;
            
        end
       

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%% Methods for basic data manipulation and prediction %%%%%%%%%%%% 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % Set data for first time
        function set_XY(obj,X,Y)
            obj.X = X;
            obj.Y = Y;
            obj.N_data = size(X,1);
            obj.get_obs_covariance();
        end
        
        % add new point to training set
        function add_data(obj,x,y)
            obj.X = [obj.X; x];
            obj.Y = [obj.Y; y];
            obj.N_data = size(obj.X,1);
            obj.update_obs_covariance();
            keyboard
        end
        
        % Gaussian kernel
        function out = evaluate_kernel(obj,x1,x2)
            diff = norm(x1-x2,2);
            out = obj.sigma^2 * exp( -diff^2 / ( 2*obj.l^2 ) );
        end
        
        % Covariance array of new test point and past training points
        function out = get_X_cov(obj,Xnew)
            N = obj.N_data;
            K_star = [0];
            for i=1:1:N
                K_star(i) = obj.evaluate_kernel(obj.X(i,:),Xnew);
            end
            out = K_star(1:N);
        end
        
        % Covariance matrix of training data input points
        function out = get_obs_covariance(obj)      
            N = obj.N_data;
            obj.K_obs = zeros(N,N);
            for i =1:1:N
                for j = 1:1:N
                    cov = obj.evaluate_kernel(obj.X(i,:),obj.X(j,:));
                    if i==j
                        obj.K_obs(i,i) = cov + obj.noise;
                    else
                        obj.K_obs(i,j) = cov;
                        obj.K_obs(j,i) = cov;
                    end
                end
            end
            out = obj.K_obs(1:N,1:N);
            
        end
        
        % Update covariance matrix given new data (run after add_data)
        function out = update_obs_covariance(obj)
            N = obj.N_data;
            x = obj.X(end,:);
            for i=1:1:N
                cov = obj.evaluate_kernel(x, obj.X(i,:));
                if (i == N)
                    obj.K_obs(N, N) = cov + obj.noise;
                else
                    obj.K_obs(i, N) = cov;
                    obj.K_obs(N, i) = cov;
                end
            end
            out =  obj.K_obs(1:N, 1:N);
        end
        
        % Do prediction
        function [mean, cov] = predict(obj,Xnew)
            N = obj.N_data;
            K_inv = inv(obj.K_obs(1:N,1:N));
            k_star = obj.get_X_cov(Xnew);
            mean = ( k_star * K_inv  ) * obj.Y(1:N,:);
            Sigma = obj.evaluate_kernel(Xnew,Xnew) + obj.noise -  k_star * K_inv * k_star';
            cov = kron(Sigma, obj.omega);   
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%% Methods for Optimizing Hyperparameters %%%%%%%%%%%% 
        %%%% They all operate on sampled subset of training set %%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % Randomly sample a subset of training data
        function [Xs, Ys] = resample(obj,n_samples)
            N = size(obj.X,1);
            idx = randsample([1:1:N],min(n_samples,N));
            obj.Xs = obj.X(idx,:);
            obj.Ys = obj.Y(idx,:);      
            obj.Ns_data = size(obj.Xs,1);
        end
        
        % Evaluate derivative of kernel (w.r.t. length scale)
        function out = dk_dl(obj,x1,x2)
            diff = norm(x1 - x2);
            out = obj.sigma^2 * exp( -diff^2 / ( 2*obj.l^2 ) ) * ( diff^2 / (obj.l^3) );
        end
        
        % Evaluate derivative of kernel (w.r.t. sigma)
        function out = dk_ds(obj,x1,x2)           
            diff = norm(x1 - x2);
            out =  2*obj.sigma * exp(-diff^2 / (2*obj.l^2));            
        end
        
        % Get derivative of covariance matrix (w.r.t. length scale and sigma) (of sampled data)
        function [Kl, Ks]  = get_dK(obj)
           
            N = size(obj.Xs,1);
            Kl = zeros(N,N);
            Ks = zeros(N,N);
            for i=1:1:N
                for j=1:1:N
                    val_l = obj.dk_dl(obj.Xs(i,:), obj.Xs(j,:));
                    val_s = obj.dk_ds(obj.Xs(i,:), obj.Xs(j,:));
                if (i == j)
                    Kl(i, i) = val_l;
                    Ks(i, i) = val_s;
                else
                    Kl(i, j) = val_l;
                    Kl(j, i) = val_l;
                    Ks(i, j) = val_s;
                    Ks(j, i) = val_s;
                end
                end
            end     
          
        end       
        
        % Covariance matrix of sampled training points
        function out = get_covariance(obj)        
            N = obj.Ns_data;
            K_obs = zeros(N,N);
            for i =1:1:N
                for j = 1:1:N
                    cov = obj.evaluate_kernel(obj.Xs(i,:),obj.Xs(j,:));
                    if i==j
                        K_obs(i,i) = cov + 0.2*obj.noise;
                    else
                        K_obs(i,j) = cov;
                        K_obs(j,i) = cov;
                    end
                end
            end
            out = K_obs(1:N,1:N);
            
        end
        
        % Compute negative log likelihood
        function L = log_likelihood(obj)
            n = size(obj.Xs,1); % data size
            d = size(obj.Ys,2); % single output size
            
            L = -0.002;

            % self.omega = np.matmul(self.L, np.transpose(self.L))
            K = obj.get_covariance();


            % Formula assumes prior is 0: V= omega, U=K in https://en.wikipedia.org/wiki/Matrix_normal_distribution
            
            A = inv(K) * obj.Ys * inv(obj.omega) * obj.Ys';               
            L = (n*d/2)*log(2*pi) + (d/2)*log(det(K)) + (n/2)*log(det(obj.omega)) + (1/2)*trace(A);     
              
            iter = 1;
            Ns = size(obj.Xs,1);
            while (L<-0.001) && iter<20
                obj.resample(floor(4*Ns/5))
                K = obj.get_covariance();
                A = inv(K) * obj.Ys * inv(obj.omega) * obj.Ys';                                         % A2 = inv(obj.omega) * obj.Ys' * inv(K) * obj.Ys;
                L = (n*d/2)*log(2*pi) + (d/2)*log(det(K)) + (n/2)*log(det(obj.omega)) + (1/2)*trace(A);  % L2 = (n*d/2)*log(2*pi) + (d/2)*log(det(K)) + (n/2)*log(det(obj.omega)) + (1/2)*trace(A2);
                
                iter = iter + 1;                    
            end    
            
            if L<-0.01
%                     disp("*************** WARN:   L < 0 **********************")
                    L = 0.001;
            end

            obj.resample(Ns);

        end
        
        % Get gradient of negative log likelihood (w.r.t. length scale, sigma, omega)
        function [dL_dl, dL_ds, dL_domega] = likelihood_gradients(obj)
            
            n = size(obj.Xs,1); % data size
            d = size(obj.Ys,2); % single output size
            
            K = obj.get_covariance();
            [Kl, Ks] = obj.get_dK();
            
            Kinv = inv(K);
            omegainv = inv(obj.omega);
            A = Kinv * obj.Ys * omegainv * obj.Ys';          
            L = (n*d/2)*log(2*pi) + (d/2)*log(det(K)) + (n/2)*log(det(obj.omega)) + (1/2)*trace(A); 
            dL_dl = (d/2)*trace(Kinv * Kl) + (1/2)*trace(-Kinv * Kl * A);
            dL_ds = (d/2)*trace(Kinv * Ks) + (1/2)*trace( -Kinv * Ks * A );
            dL_domega = (n/2) * omegainv' - (1/2)*(omegainv' * obj.Ys' * Kinv' * obj.Ys * omegainv' );
            
            iter = 1;
            Ns = size(obj.Xs,1);
            while (L<-0.001) && iter<20
                obj.resample(floor(4*Ns/5))
                K = obj.get_covariance();
                [Kl, Ks] = obj.get_dK();
                Kinv = inv(K);
                omegainv = inv(obj.omega);
                A = inv(K) * obj.Ys * inv(obj.omega) * obj.Ys';
                L = (n*d/2)*log(2*pi) + (d/2)*log(det(K)) + (n/2)*log(det(obj.omega)) + (1/2)*trace(A);
                dL_dl = (d/2)*trace(Kinv * Kl) + (1/2)*trace(-Kinv * Kl * A);
                dL_ds = (d/2)*trace(Kinv * Ks) + (1/2)*trace( -Kinv * Ks * A );
                dL_domega = (n/2) * omegainv' - (1/2)*(omegainv' * obj.Ys' * Kinv' * obj.Ys * omegainv' );
                iter = iter + 1;
            end    
            
            if L<-0.01
%                     disp("*************** WARN:   L < 0 **********************")
                    L = 0.001;
            end

            obj.resample(Ns);
            
            % Note:
            % for gradient of log |A|, see https://en.wikipedia.org/wiki/Jacobi's_formula
        end
        
      
        function out = clip(obj,x,xmax,xmin)
            if x>xmax
                x = xmax;
            elseif x<xmin
                x = xmin;
            end
            out = x;
        end
        
        % Optimize hyperparameters
        function fit(obj,iter_max)
            
            %initialize GP with random parameters
            params_omega = [];
            params_sigma = [];
            params_l = [];
            vals = [];
            
            iters = 0;
            max_iters = iter_max; %15000;
            alter_iter = 2;
            
            grad_max = 50; %50
            omega_grad_max = 40;            
            rate = 0.005; % Learning rate   %0.0005;
            
            mode = randi(3);
            
            cur_omega = obj.omega;
            cur_sigma = obj.sigma;
            cur_l = obj.l;
            
            
            while iters < max_iters
                
                prev_omega = obj.omega;
                prev_sigma = obj.sigma;
                prev_l = obj.l;
                
                if (iters == 5000)
                    rate = 0.0003;
                end
                if (iters == 10000)
                    rate = 0.0002;
                end
                
                % Get gradients
                
                Ns_org = obj.Ns_data;
                obj.resample(floor(4*Ns_org/2));
                [dL_dl, dL_ds, dL_domega] = obj.likelihood_gradients();
                dL_dl = obj.clip(dL_dl, grad_max, -grad_max);
                dL_ds = obj.clip(dL_ds, grad_max, -grad_max);
                if ( (max(max(dL_domega)) > omega_grad_max) || (min(min(dL_domega)) < -omega_grad_max) )
                    
                    max_val = max(max(max(dL_domega)), abs(min(min(dL_domega))));
                    dL_domega = dL_domega * (omega_grad_max / max_val);
                    
                end
                
                % Gradient descent
                eps = 0.0005;

                % Update only one parameter at a time
                if mode==1
                    cur_omega = cur_omega - rate * dL_domega;
                    try
                        [V, D] = eig(cur_omega);
                    catch
                        keyboard
                    end
                    for i=1:1:size(D,1)
                        if (D(i,i) <= eps)
                            D(i,i) = eps;
                        end
                    end
                    cur_omega = V * D * inv(V);
                    if det(obj.omega)<0
                        keyboard
                    end
                elseif (mode == 2)
                    cur_l = cur_l - rate * dL_dl;
                    if (cur_l < eps)
                        cur_l = eps;
                    end
                elseif (mode == 3)
                    cur_sigma = cur_sigma - rate * dL_ds;
                    if (cur_sigma < eps)
                        cur_sigma = eps;
                    end
                else
                    disp("Error in parameter update")
                    keyboard
                end
                
                % sigma cannot be negative so constrain it
                if cur_sigma<0
                    cur_sigma = 0;
                end
                
                % Update parameters
                obj.omega = cur_omega;
                obj.sigma = cur_sigma;
                obj.l = cur_l;
                obj.omega = (obj.omega + obj.omega')/2;
                
                
                iters = iters+1; %iteration count
                value = obj.log_likelihood();
                
                params_omega = [params_omega obj.omega];
                params_sigma = [params_sigma obj.sigma];
                params_l = [params_l obj.l];
                vals = [vals value];
                
                if mod(iters, alter_iter) == 0
                    if (mode < 3)
                        mode = mode + 1;
                    elseif (mode == 3)
                        mode = 1;
                    else
                        disp("Error in setting modeiable to update.")
                        keyboard
                    end
                end
                
                % Print training status
                if mod(iters, 10 == 0)
                    fprintf("Iteration: %d, mode: %d, Likelihood for this dataset: %f, der: %f, %f, %f \n", iters, mode, value, dL_dl, dL_ds, dL_domega)
                end
   
            end
            fprintf("Iteration: %d, mode: %d, Likelihood for this dataset: %f \n", iters, mode, value)
            
            % reset the covariance matrix
            obj.get_obs_covariance();
        

        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%  Extra Unsed Functions: UNDER DEVELOPMENT, DO NOT USE %%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % probability of given point           
        function prob = probability(obj,X)
            prob = exp( -0.5*trace( inv(obj.B)*(X-obj.M)' * inv(obj.A) * (X-obj.M) ) )/ ( (2*pi)^(obj.n*obj.m/2) * det(obj.A)^(obj.m/2) * det(obj.B)^(obj.n/2) );
        end
             
        
        % vec version of matrix probability
        function out = vec(obj,X)
       
            out.mean = matrix_to_vec(obj.M);
            out.covriance = kron(obj.B,obj.A);
        
        end
    
    end
     
end


% x_dot = [f(x) + g(x)]u_bar
% u_bar = [1; u]; 
% x: n
% u: m
% F(x) = [f(x) g(x)]: n x (1 + m)