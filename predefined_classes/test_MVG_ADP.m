% A single output vector y is assumed to be arranged in a row vector 
% Multiple outout vectors concatenated so that the training Y coulmns
% corresponds to different training points and rows correspond to output of
% a single training point.

% different rows in different rows

close all;
clear all;

%% Create Data
N = 100;   % number of data points  
X = 10*[linspace(0,1,N)' linspace(0,1,N)'];
U = [ ones(N,1) linspace(0,1,N)'];      
% y = 1 + X + X.*X + 0.2*randn(N,1);  
y = [ U(:,2).*cos(X(:,1)) U(:,2).*sin(X(:,2)) ];

X = [X U]; 

%% Initial GP Hyper parameters
Omega = [1 0;
         0 1];  % between components of output vector
sigma = 0.2;    % sigma for Gaussian kernel
l = 2.0;        % length scale for Gaussian kernel


%% GP without training
gp_org = MatrixVariateGaussianProcessGeneralized(Omega,[sigma;l], 2, 2);
gp_org.set_XY(X,y);
gp_org.resample(N);

%% GP with training train
gp = MatrixVariateGaussianProcessGeneralized(Omega,[sigma;l], 2, 2);
gp.set_XY(X,y);
gp.resample(N);
max_iter = 30;
gp.fit(max_iter,1);
% 
% disp("MultiVariate GP")
% disp("New Parameters: ")
% disp("Omega")
% disp(gp.omega)
% disp("Sigma")
% disp(gp.sigma)
% disp("l")
% disp(gp.l)

%% Plot y1 = sin(X) prediction
index = 1;
figure(1)
plot_results(N,X,y,gp_org,gp,index)


%% Plot y1 = cos(X) prediction
index = 2;
figure(2)
plot_results(N,X,y,gp_org,gp,index)

%% Function definitions

function plot_results(N,X,y,gp_org,gp,index)

    counter = linspace(1,size(X,1),size(X,1))';
    plot(counter,y(:,index),'r--','LineWidth',2)
    hold on

    y_org = zeros(N,2);    % mean, mean-cov, mean+cov for N data points
    y_train = zeros(N,2);  % mean, mean-cov, mean+cov for N data points
    factor_org = 2;     % factor times covariance
    factor_train = 2;  % factor times covariance

    for i=1:1:N
        [mean, cov, omega] = gp_org.predict(X(i,:));
        mean = mean*[X(i,3);X(i,4)];
        cov = ([X(i,3) X(i,4)] * cov * [X(i,3);X(i,4)]) * omega;
%         cov = cov * omega * ([X(i,3) X(i,4)]*[X(i,3);X(i,4)]);
        y_org(i,:) = [mean(index), factor_org*sqrt(cov(index,index))];
        
        [mean, cov, omega] = gp.predict(X(i,:));
        mean = mean*[X(i,3);X(i,4)];
        cov = cov * omega * ([X(i,3) X(i,4)]*[X(i,3);X(i,4)]);
        y_train(i,:) = [mean(index), factor_org*sqrt(cov(index,index))];
%         [mean, cov] = gp.predict(X(i,:));
%         y_train(i,:) = [mean(index), mean(index)+factor_train*cov(index,index), mean(index)-factor_train*cov(index,index)];
    end

%     plot(counter, y_org(:,1),'r')
%     plot(X,y_train(:,1),'g')
    errorbar(counter, y_org(:,1),y_org(:,2),'--ko')
    errorbar(counter, y_train(:,1),y_train(:,2),'--bo')
%     patch([X;flipud(X)],[y_org(:,2);flipud(y_org(:,3))],'m','FaceAlpha',0.1); 
%     patch([X;flipud(X)],[y_train(:,2);flipud(y_train(:,3))],'b','FaceAlpha',0.1); 
    
%     xlabel("X")
%     ylabel("Function value")
%     legend("True values",'Untrained GP prediction','Trained GP prediction','Normal GP prediction','Untrained Uncertaimnty bounds','Trained Uncertainty bound','Normal Uncertainty bound')
 end