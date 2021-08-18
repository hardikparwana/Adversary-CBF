%% 
clear all;
close all;



% N = 100;
% X = 100*linspace(0,1,N)';
% X = [X,X.^2,X.^3];        
%y = 1 + X + sin(20*X*[1;-2;6]).*X.*X + 0.2*randn(N,3);  

N = 100;
X = 10*linspace(0,1,N)';
X = [X];       
y = 1 + X + X.*X + 0.2*randn(N,1);  


% y = X.^2;
% D = size(X,2);





% Omega = [1 0 0;
%          0 100 0;
%          0 0 1000];
% sigma = 0.2;
% l = 10000;

Omega = [1];  % between components of output vector
sigma = 0.2;
l = 20;



gp = MatrixVariateGaussianProcess(Omega,sigma,l);
gp.set_XY(X,y);

x_new = [6.5];


[mean, cov] = gp.predict(x_new)
% keyboard


% log likelihood
gp.resample(N);
gp.log_likelihood()
[ll, ss, oo] = gp.likelihood_gradients()
gp.fit(100)



%Plot results



figure(1)

plot(X,y)


% hold on
% plot(y(:,1),'r')
% plot(y(:,2),'v')
% plot(y(:,3),'k')
