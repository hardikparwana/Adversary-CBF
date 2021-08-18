clear all
close all

testcustomkernel_new()
% testcustomkernel_sample()

function testcustomkernel_sample()
    %%Example data
              rng(0,'twister');
%               keyboard
              N = 100;
              X = linspace(0,1,N)';
              X = [X,X.^2,X.^3];        
%               keyboard
              y = 1 + X*[1;2;3] + sin(20*X*[1;-2;0]).*X(:,3) + 0.2*randn(N,1);    
              D = size(X,2);
%               keyboard
    %%Initial values of the _unconstrained_ kernel parameters
              % Note how "easy to understand" parameters like sigmaL and sigmaF
              % are converted into unconstrained parameters in theta. See
              % mykernel below for a description of these parameters and the
              % order in which they are concatenated.
              sigmaL10 = 0.1*ones(D,1); % numer of dimensions??
              sigmaL20 = 0.1;
              sigmaF10 = 0.2;
              sigmaF20 = 0.2;        
              theta0   = [log(sigmaL10);log(sigmaL20);log(sigmaF10);log(sigmaF20)];
    %%Fit the model using custom kernel function
              % mykernel defined later in this file is a sum of two kernel
              % functions - a squared exponential and a squared exponential ARD.
              % Initial value theta0 must be supplied when using a custom kernel
              % function.
              gpr = fitrgp(X,y,'kernelfunction',@mykernel,'kernelparameters',theta0,'verbose',1)
             
    %%Plot predictions
              plot(y,'r');
              hold on;
              
              [ypred,~,yint] = predict(gpr,X);
              ind = 1:1:size(y,1); ind=ind';
%              keyboard
              patch([ind;flipud(ind)],[yint(:,1);flipud(yint(:,2))],'k','FaceAlpha',0.1); 
              plot(ypred,'b')
    %%Display kernel parameters
              gpr.KernelInformation
              gpr.KernelInformation.KernelParameterNames
              % These are the estimated parameters.
              thetaHat = gpr.KernelInformation.KernelParameters;
    %%Convert kernel parameters into an easy to understand form
              % Convert kernel parameters into length scales and signal
              % standard deviations.
              params   = exp(thetaHat);
              sigmaL1  = params(1:D,1);
              sigmaL2  = params(D+1,1);
              sigmaF1  = params(D+2,1);
              sigmaF2  = params(D+3,1);
end

function testcustomkernel_new()
    %%Example data
              rng(0,'twister');
%               keyboard
              N = 100;
              X = linspace(0,1,N)';
              X = [X,X.^2,X.^3];        
%               keyboard
              y = 1 + X*[1;2;3] + cos(20*X*[1;-2;0]).*X(:,3) + 0.05*randn(N,1);    
              D = size(X,2);
%               keyboard
    %%Initial values of the _unconstrained_ kernel parameters
              % Note how "easy to understand" parameters like sigmaL and sigmaF
              % are converted into unconstrained parameters in theta. See
              % mykernel below for a description of these parameters and the
              % order in which they are concatenated.

              sigmaL10 = 0.1;%0.1*ones(D,1); % numer of dimensions??
              sigmaL20 = 0.1;
              sigmaF10 = 2.3;
              sigmaF20 = 2.3;       
              theta0 = [log(sigmaL10);log(sigmaL20);log(sigmaF10);log(sigmaF20)];
    %%Fit the model using custom kernel function
              % mykernel defined later in this file is a sum of two kernel
              % functions - a squared exponential and a squared exponential ARD.
              % Initial value theta0 must be supplied when using a custom kernel
              % function.
              gpr = fitrgp(X,y,'kernelfunction',@ADPCK,'kernelparameters',theta0,'verbose',1)
              
    %%Plot predictions
              plot(y,'r');
              hold on;
              
              [ypred,~,yint] = predict(gpr,X);
              ind = 1:1:size(y,1); ind=ind';
%              keyboard
              patch([ind;flipud(ind)],[yint(:,1);flipud(yint(:,2))],'k','FaceAlpha',0.1); 
              plot(ypred,'b')
    %%Display kernel parameters
              gpr.KernelInformation
              gpr.KernelInformation.KernelParameterNames
              % These are the estimated parameters.
              thetaHat = gpr.KernelInformation.KernelParameters;
    %%Convert kernel parameters into an easy to understand form
              % Convert kernel parameters into length scales and signal
              % standard deviations.
end



function KMN = ADPCK(SM,SN,theta)
    % Affine Dot Product Compount Kernel 
    N = 2; %Number of states
    M = 1; %Number of inputs
    M = M + 1;  % 1 was appended to input vector
    params_per_kernel = 2;
    
    D = size(SM,2); %assuming XM,XN have same number of columns
    
    O_N = size(SM,1);
    O_M = size(SN,1);
    
%     keyboard
    
    XM = SM(:,1:N);
    UM = [ones(O_M,1) SM(:,N+1,end)];
    
    XN = SM(:,1:N);
    UN = [ones(O_N,1) SM(:,N+1,end)];
    
    KMN = zeros(size(SM,1),size(SN,1));
    
    for i=1:size(SM,1)
       for j=1:size(SN,1)           
            for k=1:M               
                params_start = params_per_kernel*(k-1) + 1;
                params_end = params_start + params_per_kernel - 1;
                KMN(i,j) = KMN(i,j) + UM(i,:)*squared_exponential_kernel(XM(i,:),XN(i,:),theta(params_start:params_end,1))*UN(j,:)';                
            end           
       end
    end
    
   
end

function kij = squared_exponential_kernel(xi,xj,theta)
% squared exponential kernel

    params = exp(theta);
    sigmal = theta(1,1);
    sigmaf = theta(2,1);
    
    kij = (sigmaf^2)*exp( -( xi - xj )'*( xi - xj )/2/(sigmal^2) );    
    
end

function KMN = mykernel(XM,XN,theta)
    %mykernel - Compute sum of squared exponential and squared exponential ARD.
    %   KMN = mykernel(XM,XN,theta) takes a M-by-D matrix XM, a N-by-D matrix
    %   XN and computes a M-by-N matrix KMN of kernel products such that
    %   KMN(i,j) is the kernel product between XM(i,:) and XN(j,:). theta is
    %   the R-by-1 unconstrained parameter vector for the kernel.
    %
    %   Let theta = [log(sigmaL1);log(sigmaL2);log(sigmaF1);log(sigmaF2)]
    %
    %   where
    %
    %   sigmaL1 = D-by-1 vector of length scales for squared exponential ARD.
    %   sigmaL2 = scalar length scale for squared exponential.
    %   sigmaF1 = scalar signal standard deviation for squared exponential ARD.
    %   sigmaF2 = scalar signal standard deviation for squared exponential.
          % 1. Get D. Assume XN, XM have the same number of columns.
          D = size(XM,2);
          % 2. Convert theta into sigmaL1, sigmaL2, sigmaF1 and sigmaF2.
          params  = exp(theta);
          sigmaL1 = params(1:D,1);
          sigmaL2 = params(D+1,1);
          sigmaF1 = params(D+2,1);
          sigmaF2 = params(D+3,1);
          % 3. Create the contribution due to squared exponential ARD.    
          KMN = pdist2(XM(:,1)/sigmaL1(1), XN(:,1)/sigmaL1(1)).^2;
          for r = 2:D
              KMN = KMN + pdist2(XM(:,r)/sigmaL1(r), XN(:,r)/sigmaL1(r)).^2;        
          end
          KMN = (sigmaF1^2)*exp(-0.5*KMN);
          % 4. Add the contribution due to squared exponential.
          KMN = KMN + (sigmaF2^2)*exp(-0.5*(pdist2(XM/sigmaL2, XN/sigmaL2).^2));       
  end