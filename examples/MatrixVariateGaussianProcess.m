classdef MatrixVariateGaussianProcess <handle
    
    % x_dot = [f(x) + g(x)]u_bar
    % u_bar = [1; u]; 
    % x: n
    % u: m
    % F(x) = [f(x) g(x)]: n x (1 + m)

    properties(Access = public)
         
        %Params
        omega;
        sigma;
        l;
        
        %Data
        X;      % input
        Y;      % output
        N_data; % size of data
        
        %
        K_obs;
        noise = 0.1;
        
        %Randomly Sampled data for hyperpeter tuning
        X_obs;
        Y_obs;
        
    end
    
    
    methods(Access = public)
        
        function obj = MatrixVariateGaussianProcess(omega,sigma,l)
                     
            obj.omega = omega;
            obj.sigma = sigma;
            obj.l = l;
            
        end
        
        % probability of given point
        function prob = probability(obj,X)
            prob = exp( -0.5*trace( inv(obj.B)*(X-obj.M)' * inv(obj.A) * (X-obj.M) ) )/ ( (2*pi)^(obj.n*obj.m/2) * det(obj.A)^(obj.m/2) * det(obj.B)^(obj.n/2) );
        end
             
        
        % vec version of matrix probability
        function out = vec(obj,X)
       
            out.mean = matrix_to_vec(obj.M);
            out.covriance = kron(obj.B,obj.A);
        
        end

        
        function [Xs, Ys] = resample(obj,n_samples)
            N = size(obj.X,0);
            idx = randsample([1:1:N],min(n_sampoles,N));
            obj.Xs = obj.X(idx,:);
            obj.Ys = obj.Y(idx,:);            
        end
        
        function set_XY(obj,X,Y)
            obj.X = X;
            obj.Y = Y;
%             keyboard
            obj.N_data = size(X,1);
            obj.get_obs_covariance();
        end
        
        function add_data(obj,x,y)
            obj.X = [obj.X; x];
            obj.Y = [obj.Y; y];
            obj.N_data = size(obj.X,1);
            obj.update_obs_covariance();
        end
        
        function out = evaluate_kernel(obj,x1,x2)
            diff = norm(x1-x2,2)
            out = obj.sigma^2 * exp( -diff^2 / ( 2*obj.l^2 ) );
        end
        
        function out = get_X_cov(obj,Xnew)
            N = obj.N_data;
            K_star = [0];
            for i=1:1:N
                K_star(i) = obj.evaluate_kernel(obj.X(i,:),Xnew);
%                 if i>=58
%                     keyboard
%                 end
            end
            out = K_star(1:N);
%             keyboard
        end
        
        function out = get_obs_covariance(obj)
           
            N = obj.N_data;
            obj.K_obs = zeros(N,N);
            for i =1:1:N
                for j = 1:1:N
                    cov = obj.evaluate_kernel(obj.X(i,:),obj.X(j,:));
%                     keyboard
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
        
        % Update comodeiance matrix given new data (run after add_data)
        function out = update_obs_covariance(obj)
            N = obj.N_data;
            x = obj.X(end,:);
            for i=1:1:N
                cov = obj.evaluate_kernel(x, obj.X(i,:));
                if (i == N)
                    obj.K_obs(N, N) = cov + self.noise;
                else
                    obj.K_obs(i, N) = val;
                    obj.K_obs(N, i) = val;
                end
            end
            out =  obj.K_obs(1:N, 1:N);
        end
        
        function [mean, cov] = predict(obj,Xnew)
        
            N = obj.N_data;
            K_inv = inv(obj.K_obs(1:N,1:N));
            k_star = obj.get_X_cov(Xnew);
            disp(size(K_inv))
            disp(size(k_star))
            mean = ( k_star * K_inv  ) * obj.Y(1:N,:);
            Sigma = obj.evaluate_kernel(Xnew,Xnew) + obj.noise -  k_star * K_inv * k_star';
            cov = kron(Sigma, obj.omega);            
            keyboard
        end
        
        % Evaluate derivative of kernel (w.r.t. length scale)
        function out = dk_dl(obj,x1,x2)
            diff = norm(x1 - x2);
            out = obj.sigma^2 * exp( -diff^2 / ( 2*obj.l^2 ) ) * ( diff^2 / (obj.j^3) );
        end
        
        % Evaluate derivative of kernel (w.r.t. sigma)
        function out = dk_ds(obj,x1,x2)           
            diff = np.linalg.norm(x1 - x2);
            out =  2*obj.sigma * np.exp(-diff^2 / (2*obj.l^2));            
        end
        
        % Get derivative of comodeiance matrix (w.r.t. length scale and sigma)
        function [Kl, Ks]  = get_dK_sample(obj)
           
            N = size(obj.Xs);
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
        
        
        
        
        % Get gradient of negative log likelihood (w.r.t. length scale, sigma, omega)
        function [dL_dl, dL_ds, dL_domega] = likelihood_gradients(obj)
            n, d = size(obj.X);
            
            K = obj.get_comodeiance();
            Kl, Ks = obj.get_dK();
            
            Kinv = inv(K);
            omegainv = inv(obj.omega);
            A = Kinv * self.Y * omegainv * obj.Y';
            dL_dl = (d/2)*trace(Kinv * Kl) + (1/2)*trace(-Kinv * Kl * A);
            dL_ds = (d/2)*trace(Kinv * Ks) + (1/2)*trace( -Kinv * Ks * A );
            dL_domega = (n/2) * omegainv' - (1/2)*(omegainv' * self.Y' * Kinv' * self.Y * omegainv' );
        end

        % Get gradient of negative log likelihood (w.r.t. length scale, sigma, omega)
        function [dL_dl, dL_ds, dL_domega] = likelihood_gradients_sample(obj)
            n, d = size(obj.Xs);
            
            K = obj.get_comodeiance_sample();
            Kl, Ks = obj.get_dK_sample();
            
            Kinv = inv(K);
            omegainv = inv(obj.omega);
            A = Kinv * self.Y * omegainv * obj.Ys';
            dL_dl = (d/2)*trace(Kinv * Kl) + (1/2)*trace(-Kinv * Kl * A);
            dL_ds = (d/2)*trace(Kinv * Ks) + (1/2)*trace( -Kinv * Ks * A );
            dL_domega = (n/2) * omegainv' - (1/2)*(omegainv' * obj.Ys' * Kinv' * obj.Ys * omegainv' );
        end
        
        % Compute negative log likelihood
        function L = log_likelihood(obj)
            n, d = size(obj.X);

            % self.omega = np.matmul(self.L, np.transpose(self.L))
            self.get_covariance()

            A = inv(obj.K_obs) * obj.Y * inv(obj.omega) * self.Y';
            L = (n*d/2)*log(2*np.pi) + (d/2)*log(det(obj.K_obs)) + (n/2)*log(det(obj.omega)) + (1/2)*trace(A);
        end
        
        % Compute negative log likelihood
        function L = log_likelihood_sample(obj)
            n, d = size(obj.Xs);

            % self.omega = np.matmul(self.L, np.transpose(self.L))
            self.get_comodeiance()

            A = inv(obj.Ks) * obj.Ys * inv(obj.omega) * self.Y';
            L = (n*d/2)*log(2*np.pi) + (d/2)*log(det(obj.Ks)) + (n/2)*log(det(obj.omega)) + (1/2)*trace(A);
        end
        
        function out = clip(x,xmax,xmin)
            if x>xmax
                x = xmax;
            elseif x<xmin
                x = xmin;
            end
            out = x;
        end
        
        % Optimize hyperparameters
        function fit()
            
            %initialize GP with random parameters
            params_omega = [];
            params_sigma = [];
            parama_l = [];
            
            iters = 0;
            max_iters = 15000;
            alter_iter = 30;
            
            grad_max = 50;
            omega_grad_max = 40;
            rate = 0.0005;
            
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
                
                
                obj.resample(80);
                dL_dl, dL_ds, dL_domega = gp.likelihood_gradients();
                dL_dl = obj.clip(dL_dl, -grad_max, grad_max);
                dL_ds = obj.clip(dL_ds, -grad_max, grad_max);
                
                if (max(dL_domega) > omega_grad_max || min(dL_domega) < omega_grad_max)
                    
                    max_val = max(max(dL_domega), abs(min(dL_domega)));
                    dL_domega = dL_domega * (omega_grad_max / max_val);
                    
                end
                
                % Gradient descent
                eps = 0.0005;
                
                if mode==0
                    cur_o = cur_o - rate * dL_domega;
                    D, V = eig(cur_o);
                    for i=1:1:size(D,0)
                        if (D(i) <= eps)
                            D(i) = eps;
                        end
                    end
                    cur_o = V * diag(D) * inv(V);
                elseif (mode == 1)
                    cur_l = cur_l - rate * dL_dl;
                    if (cur_l < eps)
                        cur_l = eps;
                    end
                elseif (mode == 2)
                    cur_s = cur_s - rate * dL_ds
                    if (cur_s < eps)
                        cur_s = eps;
                    end
                else
                    disp("Error in parameter update")
                end
                
                % Update parameters
                obj.omega = cur_o;
                obj.sigma = cur_s;
                obj.l = cur_l;
                obj.omega = (obj.omega + gp.omega')/2;
                
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
                        mode = 0;
                    else
                        disp("Error in setting modeiable to update.")
                    end
                end
                
                % Save parameters and likelihoods
                if mod(iters, 10 == 0)
                    fprintf("Iteration: %d", iters)
                    fprintf("Likelihood for this dataset: %f ", value)
%                 if (iters % 50 == 0 and kSave):
%                     np.save('likelihood_vals_robot_v' + str(iteration), vals)
%                     np.save('parameters_robot_v' + str(iteration), [params_omega, params_sigma, params_l]
                end
   
            end
        

        end
    
    end
     
end
