clear all;
close all;
clc;

warning('off')

%% Initialize and initial values

global c1 r1 c2 r2 G3 d alpha ;

  
alpha=0.05; 

G1=[-4.5 7.2 0]';
G2=[5 -1 0]';
G3=[-3 11 0]';
r=0.25;
r3=0.35;  %not used
G0=[3.25 -0.4]';  %not used
r0=0.75;  %not used
c1=[1 4]';
r1=1.6;
d=0.1;
c2=[-3 8]';
r2=1;

%% ODE formulation
tspan = [0:0.5:2200];%[0:0.5:2200];                
t=tspan';

% initial states
x_init=[0 0 pi 1 6.5 -1.73 -4 5 1.73]';

%[t,x]= ode45(@main_1,tspan, x_init);


sol= ode45(@main_1,tspan, x_init);
[y,yp] = deval(sol,tspan);
yp=yp';
x=y';



figure(1)
hold on
plot(x(:,1),x(:,2),'b-.','LineWidth',3);
plot(x(:,4),x(:,5),'r-.','LineWidth',3);
plot(x(:,7),x(:,8),'k-.','LineWidth',3);
ob1 = circle(c1(1),c1(2),r1);
ob2 = circle(c2(1),c2(2),r2);
GA = circle(G1(1),G1(2),r);
GB = circle(G2(1),G2(2),r);
GC = circle(G3(1),G3(2),r);
GC = circle(G0(1),G0(2),r0);
set(gca,'FontSize',42, 'FontWeight', 'bold','XGrid','on','LineWidth',2,'color','white')
set(gca,'FontSize',42,'FontWeight', 'bold','YGrid','on','LineWidth',2,'color','white')
legend({'A_{1}','A_{2}','A_{3}'},'FontSize',30,'FontWeight', 'bold')
set(gca,'GridLineStyle','-.'); set(gca,'box','on');
set(gca,'xlabel',text(0,0,'x_{2}'),'FontSize',32, 'FontWeight', 'bold','XGrid','on','LineWidth',2,'color','white')
set(gca,'ylabel',text(0,0,'x_{1}'),'FontSize',32, 'FontWeight', 'bold','XGrid','on','LineWidth',2,'color','white')



%% Calculation for plotting 

%% Nominal Behavior of each agent

%%  Goal Reaching Behavior
k=length(t);
for m=1:k
pos1=[x(m,1) x(m,2)]';
pos2=[x(m,4) x(m,5)]';
pos3=[x(m,7) x(m,8)]';

p=2;
te=1601;
if m<te
G1=[3 -1]';
G2=[3.5 -0.8]';
Bg(m,1)=(norm(x(m,1:2)'-G1)^p)/(norm(x(1,1:2)'-G1)^p);
Bg(m,2)=(norm(x(m,4:5)'-G2)^p)/(norm(x(1,4:5)'-G2)^p);
else
G1=[-5+0.5 7.5]';
G2=[5 -1]';
Bg(m,1)=(norm(x(m,1:2)'-G1)^p)/(norm(x(te,1:2)'-G1)^p);
Bg(m,2)=(norm(x(m,4:5)'-G2)^p)/(norm(x(te,4:5)'-G2)^p);
end 
Bg(m,3)=(norm(x(m,7:8)'-G3)^p)/(norm(x(1,7:8)'-G3)^p);

%% Collision avoidance Behavior
% Obstacle 1
Bo1(m,1)= (r1^2)/(norm(c1-x(m,1:2)')^2);
Bo1(m,2)= (r1^2)/(norm(c1-x(m,4:5)')^2);
Bo1(m,3)= (r1^2)/(norm(c1-x(m,7:8)')^2);

%Obstacle 2
Bo2(m,1)= (r2^2)/(norm(c2-x(m,1:2)')^2);
Bo2(m,2)= (r2^2)/(norm(c2-x(m,4:5)')^2);
Bo2(m,3)= (r2^2)/(norm(c2-x(m,7:8)')^2);
end 

for m=1:k
%% %% Inter-agent Behavior ((d+yp(m,7))^2)
Bma(m,1)= ((d)/(norm(x(m,1:2)'-x(m,4:5)'))); %Agent 12
Bma(m,2)= (d)/(norm(x(m,1:2)'-x(m,7:8)'));  %Agent 13
Bma(m,3)= (d)/(norm(x(m,4:5)'-x(m,7:8)'));   %Agent 23
ref(m,1)= 1;
end

figure(2)
plot(t,Bg(:,1),'b-.','LineWidth',3)
hold on
plot(t,Bg(:,2),'r-.','LineWidth',3)
plot(t,Bg(:,3),'k-.','LineWidth',3)
set(gca,'FontSize',42, 'FontWeight', 'bold','XGrid','on','LineWidth',2,'color','white')
set(gca,'FontSize',42,'FontWeight', 'bold','YGrid','on','LineWidth',2,'color','white')
legend({'\lambda_{1}','\lambda_{2}','\lambda_{3}'},'FontSize',30,'FontWeight', 'bold')
set(gca,'GridLineStyle','-.'); set(gca,'box','on');
set(gca,'xlabel',text(0,0,'Time (s)'),'FontSize',26, 'FontWeight', 'bold','XGrid','on','LineWidth',2,'color','white')
set(gca,'ylabel',text(0,0,'Goal Reaching'),'FontSize',26, 'FontWeight', 'bold','XGrid','on','LineWidth',2,'color','white')
xlim([0 2200])


figure(3)
plot(t,Bma(:,1),'b-.','LineWidth',3)
hold on
plot(t,Bma(:,2),'m-.','LineWidth',3)
plot(t,Bma(:,3),'g-.','LineWidth',3)
plot(t,ref(:,1),'k','LineWidth',2)
set(gca,'FontSize',42, 'FontWeight', 'bold','XGrid','on','LineWidth',2,'color','white')
set(gca,'FontSize',42,'FontWeight', 'bold','YGrid','on','LineWidth',2,'color','white')
legend({'g_{12}','g_{13}','g_{23}','g_{max}'},'FontSize',30,'FontWeight', 'bold')
set(gca,'GridLineStyle','-.'); set(gca,'box','on');
set(gca,'xlabel',text(0,0,'Time (s)'),'FontSize',26, 'FontWeight', 'bold','XGrid','on','LineWidth',2,'color','white')
set(gca,'ylabel',text(0,0,'Inter-agent Safety'),'FontSize',26, 'FontWeight', 'bold','XGrid','on','LineWidth',2,'color','white')
xlim([0 2200])





function xd= main_1(t,x)
global d c1 r1 c2 r2 G3 alpha;
x1=x(1); y1=x(2); t1=x(3); x2=x(4); y2=x(5); t2=x(6); x3=x(7); y3=x(8); t3=x(9);


p1=[x1 y1]';
p2=[x2 y2]';
p3=[x3 y3]';


pb1=[x1 y1 t1]';
pb2=[x2 y2 t2]';
pb3=[x3 y3 t3]';


%% we don't consider any f(x) part in simulation, tho we can use it in case required

k1=0.0; k2=0.0; k3=0.0; k4=1;

% System Dynamics 
f11= k1*sin(k4*x1);
f12= k1*cos(k4*y1);

f21= k2*sin(k4*x2);
f22= k2*cos(k4*y2);

f31= k3*sin(k4*x3);
f32= k3*cos(k4*y3);

f1=[f11;f12];
f2=[f21;f22];
f3=[f31;f32];
fb1=[f11;f12;0];    %k4*sin(V1
fb2=[f21;f22;0];
fb3=[f31;f32;0];


b=0.001;


g1=[1 0;
    0 1];
 
g2=[1 0;
    0 1];
 
g3=[1 0;
    0 1];


gb1=[1         0;
     0         1;
  -sin(t1)/b  cos(t1)/b];
 
gb2=[1         0;
     0         1;
  -sin(t2)/b  cos(t2)/b];


gb3=[1         0;
     0         1;
  -sin(t3)/b  cos(t3)/b];
 
 
 
UM=4;
Um=4;
Ut=2;

A5=[cos(t1) sin(t1)   zeros(1,2) zeros(1,2) zeros(1,12);
    -cos(t1) -sin(t1)    zeros(1,2) zeros(1,2) zeros(1,12);
   -sin(t1)/b cos(t1)/b  zeros(1,2) zeros(1,2) zeros(1,12);
    sin(t1)/b -cos(t1)/b  zeros(1,2) zeros(1,2) zeros(1,12)];
b5=[UM Um Ut Ut];


A6=[zeros(1,2)  cos(t2) sin(t2)   zeros(1,2) zeros(1,12);
    zeros(1,2)  -cos(t2) -sin(t2)    zeros(1,2) zeros(1,12);
    zeros(1,2)  -sin(t2)/b cos(t2)/b  zeros(1,2) zeros(1,12);
    zeros(1,2)   sin(t2)/b -cos(t2)/b  zeros(1,2) zeros(1,12)];

b6=b5;


A7=[zeros(1,2) zeros(1,2)  cos(t3) sin(t3)  zeros(1,12);
    zeros(1,2) zeros(1,2)  -cos(t3) -sin(t3)  zeros(1,12);
    zeros(1,2) zeros(1,2)  -sin(t3)/b cos(t3)/b  zeros(1,12);
    zeros(1,2) zeros(1,2)   sin(t3)/b -cos(t3)/b  zeros(1,12)];

b7=b5;



%% Inter-agent constraints
gamma=-2;

h12=d^2-norm(p1-p2)^2;
h13=d^2-norm(p1-p3)^2;
h23=d^2-norm(p2-p3)^2;


a12=2*(p1-p2)';
b12=a12*f1-a12*f2-alpha*h12;

a13=2*(p1-p3)';
b13=a13*f1-a13*f3-alpha*h13;
 
a23=2*(p2-p3)';
b23=a23*f2-a23*f3-alpha*h23;

%%

%% Obtacle 1 

del=0.4;

o1_1=(r1+del)^2-norm(c1-p1)^2;
o1_2=(r1+del)^2-norm(c1-p2)^2;
o1_3=(r1+del)^2-norm(c1-p3)^2;

ao11=2*(c1-p1)';
bo11=-ao11*f1-alpha*o1_1;

ao12=2*(c1-p2)';
bo12=-ao12*f2-alpha*o1_2;

ao13=2*(c1-p3)';
bo13=-ao13*f3-alpha*o1_3; 


%% obstacle_2

o2_1=(r2+del)^2-norm(c2-p1)^2;
o2_2=(r2+del)^2-norm(c2-p2)^2;
o2_3=(r2+del)^2-norm(c2-p3)^2;

ao21=2*(c2-p1)';
bo21=-ao21*f1-alpha*o2_1;

ao22=2*(c2-p2)';
bo22=-ao22*f2-alpha*o2_2;

ao23=2*(c2-p3)';
bo23=-ao23*f3-alpha*o2_3; 



%% Goal Reaching constraints
t0=800;

if t<t0
G1=[3.5 -1]';
V1=norm(p1-G1)^2;

G2=[3.5 -0.8]';
V2=norm(p2-G2)^2;

else
 G1=[-5 7.5]';  % Hardik: moving goal location? but t1 based on current x
V1=norm(p1-G1)^2;

G2=[5 -1]';
V2=norm(p2-G2)^2;   
    
end

G3=[-3 11]';
V3=norm(p3-G3)^2;

%% %Goal Reaching
az1=2*(p1-G1)';
az2=2*(p2-G2)';
az3=2*(p3-G3)';
bz1=-az1*f1-0.1*alpha*V1;
bz2=-az2*f2-0.1*alpha*V2;
bz3=-az3*f3-0.1*alpha*V3;

%%
%% Adversarial Chasing
V4=norm(p1-p2)^2;
az4=2*(p1-p2)';
bz4=-az4*f1-1*alpha*V4;


Ac3=[az4*g1];
bc3=[bz4];

% Ac=[Ac3;ao11*g1; ao21*g1];
% bc=[bc3 -ao11*f1-alpha*o1_1 -ao21*f1-alpha*o2_1]';

Hc=1*eye(2);
Fc=0*ones(2,1);
options =  optimoptions('quadprog','Display','off');
Uc = quadprog(Hc,Fc,Ac3,bc3,[],[],[],[],[],options);



%% QP Matrices
A1=[-a12*g1 a12*g2 zeros(1,2) -b12  0 0 zeros(1,9);
   -a13*g1 zeros(1,2) a13*g3    0  -b13 0 zeros(1,9);
   zeros(1,2) -a23*g2  a23*g3   0    0  -b23 zeros(1,9)];
b1=[b12 b13 b23];

A2=[ao11*g1  zeros(1,2) zeros(1,2) zeros(1,3) -1*o1_1 0 0 zeros(1,6);
    zeros(1,2)  ao12*g2  zeros(1,2) zeros(1,3)  0 -1*o1_2 0 zeros(1,6);
    zeros(1,2)  zeros(1,2)  ao13*g3 zeros(1,3) 0 0 -1*o1_3  zeros(1,6)];
b2=[bo11 bo12 bo13]; 


A3=[ao21*g1  zeros(1,2) zeros(1,2) zeros(1,6) -o2_1 0 0 zeros(1,3);
    zeros(1,2)  ao22*g2  zeros(1,2) zeros(1,6)  0 -o2_2 0 zeros(1,3);
    zeros(1,2)  zeros(1,2)  ao23*g3 zeros(1,6) 0 0 -o2_3  zeros(1,3)];
b3=[bo21 bo22 bo23]; 


A4=[az1*g1 zeros(1,2) zeros(1,2) zeros(1,9) -0*bz1 0 0;
  zeros(1,2) az2*g2 zeros(1,2)  zeros(1,9)   0 -0*bz2  0;
  zeros(1,2) zeros(1,2) az3*g3  zeros(1,9)   0  0  -0*bz3];

b4=[bz1 bz2 bz3];

A=[A1;A2;A3;A4;A5;A6;A7];
b=[b1 b2 b3 b4 b5 b6 b7]';

Hacc=1*eye(18);
Facc=0*ones(18,1);
options =  optimoptions('quadprog','Display','off');
U = quadprog(Hacc,Facc,A,b,[],[],[],[],[],options);


if ((t>800) && (t<=2200))
u1=U(1:2);
%u1=Uc(1:2);   %% Just use this for Adversarial Chasing instead of u1=U(1:2);

u2=U(3:4);
u3=U(5:6);
else
u1=U(1:2);
u2=U(3:4);
u3=U(5:6);
end


%Agents dynamics
xd1=fb1+gb1*u1;
xd2=fb2+gb2*u2;
xd3=fb3+gb3*u3;

xd = [xd1;xd2;xd3];
if (mod(t,50)<0.01)
   t
end
end





function h = circle(x,y,r)
hold on
th = 0:pi/50:2*pi;
xunit = r * cos(th) + x;
yunit = r * sin(th) + y;
h = plot(xunit, yunit,'r','LineWidth',1.5);
hold off
end