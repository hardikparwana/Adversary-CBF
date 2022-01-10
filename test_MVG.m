% A single output vector y is assumed to be arranged in a row vector 
% Multiple outout vectors concatenated so that the training Y coulmns
% corresponds to different training points and rows correspond to output of
% a single training point.

% different rows in different rows

close all;
clear all;

%% Create Data
N = 100;   % number of data points  
X = 10*linspace(0,1,N)';
X = [X];       
y = 1 + X + X.*X + 0.2*randn(N,1);  
y = [sin(X) cos(X)];

%% Initial GP Hyper parameters
Omega = [1 0;
         0 1];  % between components of output vector
sigma = 0.2;    % sigma for Gaussian kernel
l = 2.0;        % length scale for Gaussian kernel


%% GP without training
gp_org = MatrixVariateGaussianProcess(Omega,sigma,l);
gp_org.set_XY(X,y);
gp_org.resample(N);

%% GP with training train
gp = MatrixVariateGaussianProcess(Omega,sigma,l);
gp.set_XY(X,y);
gp.resample(N);

max_iter = 30;
gp.fit(max_iter,1);

disp("MultiVariate GP")
disp("New Parameters: ")
disp("Omega")
disp(gp.omega)
disp("Sigma")
disp(gp.sigma)
disp("l")
disp(gp.l)

%% Independent GP
gp_normal(1) = MatrixVariateGaussianProcess(1,sigma,l);
gp_normal(2) = MatrixVariateGaussianProcess(1,sigma,l);
gp_normal(1).set_XY(X,y(:,1));
gp_normal(2).set_XY(X,y(:,2));
gp_normal(1).resample(N);
gp_normal(2).resample(N);
max_iter = 10;
gp_normal(1).fit(max_iter,1);
gp_normal(2).fit(max_iter,1);

disp("UniVariate GP 1 ")
disp("New Parameters: ")
disp("Sigma")
fprintf("GP1: Sigma: %f, L: %f \n",gp_normal(1).sigma,gp_normal(1).l)
fprintf("GP2: Sigma: %f, L: %f ",gp_normal(2).sigma,gp_normal(2).l)

%% Plot y1 = sin(X) prediction
index = 1;
figure(1)
plot_results(N,X,y,gp_org,gp,gp_normal(index),index)


%% Plot y1 = cos(X) prediction
index = 2;
figure(2)
plot_results(N,X,y,gp_org,gp,gp_normal(index),index)

%% Function definitions

function plot_results(N,X,y,gp_org,gp,gp_normal,index)

    plot(X,y(:,index),'k--','LineWidth',2)
    hold on

    y_org = zeros(N,3);    % mean, mean-cov, mean+cov for N data points
    y_train = zeros(N,3);  % mean, mean-cov, mean+cov for N data points
    y_normal = zeros(N,3);  % mean, mean-cov, mean+cov for N data points
    factor_org = 5;     % factor times covariance
    factor_train = 5;  % factor times covariance

    for i=1:1:N
        [mean, cov] = gp_org.predict(X(i,:));
        y_org(i,:) = [mean(index), mean(index)+factor_org*cov(index,index), mean(index)-factor_org*cov(index,index)];
        [mean, cov] = gp.predict(X(i,:));
        y_train(i,:) = [mean(index), mean(index)+factor_train*cov(index,index), mean(index)-factor_train*cov(index,index)];
        [mean, cov] = gp_normal.predict(X(i,:));
        y_normal(i,:) = [mean, mean+factor_train*cov, mean-factor_train*cov];
    end

    plot(X,y_org(:,1),'r')
    plot(X,y_train(:,1),'g')
    plot(X,y_normal(:,1),'c')

    patch([X;flipud(X)],[y_org(:,2);flipud(y_org(:,3))],'m','FaceAlpha',0.1); 
    patch([X;flipud(X)],[y_train(:,2);flipud(y_train(:,3))],'b','FaceAlpha',0.1); 
    patch([X;flipud(X)],[y_normal(:,2);flipud(y_normal(:,3))],'c','FaceAlpha',0.1); 
    
    xlabel("X")
    ylabel("Function value")
    legend("True values",'Untrained GP prediction','Trained GP prediction','Normal GP prediction','Untrained Uncertaimnty bounds','Trained Uncertainty bound','Normal Uncertainty bound')
 end