%script to explore analytical model (fractional Brownian motion, rectangle
%source, infinite 2D domain)
% close all;

H = 0.75; %Hurst exponents; 0 < H < 1
sigma2 = [70, 70]; %scale parameters
n = 0.1; %porosity (0 <= n < 1)
Q = 2; %source term (density per time)
v = [1, 0]; %velocity
lambda = 0.005; %reaction rate
x1 = -0.5; %x-coordinate of lower-left corner of rectangle source
y1 = -0.5; %y-coordinate of lower-left corner of rectangle source
x2 = 0.5; %x-coordinate of upper-right corner of rectangle source
y2 = 0.5; %y-coordinate of upper-right corner of rectangle source

%% see how integral converges as final time of integration increases

%point at which state is evaluated
x = 1;
y = 1;

f = @(tau,H,sigma2,n,Q,v,lambda,x1,x2,y1,y2,x,y) ...
    state_integrand(tau,H,sigma2,n,Q,v,lambda,x1,x2,y1,y2,x,y);

ints = zeros(1,7);
upperlim = 10.^(1:7);
for eep = 1:7
    ints(eep) = ...
        integral(@(tau)f(tau,H,sigma2,n,Q,v,lambda,x1,x2,y1,y2,x,y),0,upperlim(eep));
end
ints = [ints integral(@(tau)f(tau,H,sigma2,n,Q,v,lambda,x1,x2,y1,y2,x,y),0,Inf)]

%% picture of state throughout domain

nx = 600;
ny = 100;
[x,y] = meshgrid(linspace(-60,300,nx),linspace(-60,60,ny));
tf = 200;
u = integral(@(tau)f(tau,H,sigma2,n,Q,v,lambda,x1,x2,y1,y2,x,y),0,tf,'ArrayValued',true);
u = reshape(u,ny,nx);
figure;
surf(x,y,u);
axis equal tight
view(2)
shading interp