clear all;
close all;

N = 100;
X = 100*linspace(0,1,N)';

X = [X,X.^2,X.^3];        
%               keyboard
y = 1 + X + sin(20*X*[1;-2;6]).*X.*X + 0.2*randn(N,3);  
% y = X.^2;
D = size(X,2);

% hold on
% plot(y(:,1),'r')
% plot(y(:,2),'v')
% plot(y(:,3),'k')


Omega = [1 0 0;
         0 100 0;
         0 0 10000];
sigma = 0.2;
l = 10000;

gp = MatrixVariateGaussianProcess(Omega,sigma,l);
gp.set_XY(X,y);

x_new = [60.5, 60.5^2, 60.5^3];


[mean, cov] = gp.predict(x_new)

norm(mean - y(60,:))