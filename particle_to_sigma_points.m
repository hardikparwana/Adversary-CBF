state_dim = 3;
num_particles = 1000;
particles = rand(state_dim,num_particles);

% Mean
mu = mean( particles , 2 );

particles_mu = particles - mu;

% Covariance

% Self
Ps = zeros(state_dim,state_dim);
for i=1:size(particles,2)
    Ps = Ps + (particles_mu(:,i))*(particles_mu(:,i))';
end
Ps = Ps/(i-1);

% Matlab function
P = cov(particles');

% Skewness Tensor
S = zeros(state_dim, state_dim, state_dim);
for i=1:1:state_dim
   for j=1:1:state_dim
      for k=1:1:state_dim
         S(i,j,k) = mean( particles_mu(i,:) .* particles_mu(j,:) .* particles_mu(k,:) ) ;
      end
   end
end

% Kurtosis Tensor
K = zeros(state_dim, state_dim, state_dim, state_dim);
for i=1:1:state_dim
   for j=1:1:state_dim
      for k=1:1:state_dim
          for l=1:1:state_dim
            K(i,j,k,l) = mean( particles_mu(i,:) .* particles_mu(j,:) .* particles_mu(k,:) .* particles_mu(l,:) ) ;
          end
      end
   end
end

