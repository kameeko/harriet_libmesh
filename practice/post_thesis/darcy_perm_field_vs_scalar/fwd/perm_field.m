%script to generate "true" permeability field

close all;

Nx = 50;
Ny = 50; 
Lx = 100; %m
Ly = 100; %m
clx = 10.0; %correlation length
cly = 10.0; %correlation length
k_avg = 2.72e-10; %m^2
V_dp = 0.9; %Dykstra-Parsons coefficient

xnodes = linspace(-Lx/2,Lx/2,Nx+1);
ynodes = linspace(-Ly/2,Ly/2,Ny+1);
xcent = 0.5*(xnodes(1:end-1)+xnodes(2:end));
ycent = 0.5*(ynodes(1:end-1)+ynodes(2:end));
[X,Y] = meshgrid(xcent,ycent);
s = -log(1-V_dp);
mu = log(k_avg)-s^2/2;
Z = s*randn(Nx,Ny);
F = exp(-(X.^2/(clx^2/2.0)+Y.^2/(cly^2/2.0)));
f = 2.0/sqrt(pi)*Lx/sqrt(Nx*Ny)/sqrt(clx)/sqrt(cly)*ifft(fft(Z).*fft(F'));
perm = exp(mu+real(f))';

imagesc(perm);

dlmwrite('true_perm.dat',[X(:)+Lx/2,Y(:)+Ly/2,perm(:)],' ')