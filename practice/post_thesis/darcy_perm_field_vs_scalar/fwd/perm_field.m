%script to generate "true" permeability field

close all;

Nx = 100; %for creating field
Ny = 20; %for creating field
Nxo = 100; %for outputting
Nyo = 100; %for outputting
Lx = 100; %m
Ly = 100; %m
clx = 10.0; %correlation length
cly = 30.0; %correlation length
%k_avg = 2.72e-10; %m^2
k_avg = 275.8; %darvies
V_dp = 0.9; %Dykstra-Parsons coefficient

xnodes = linspace(-Lx/2,Lx/2,Nx);
ynodes = linspace(-Ly/2,Ly/2,Ny);
[X,Y] = meshgrid(xnodes,ynodes);
s = -log(1-V_dp);
mu = log(k_avg)-s^2/2;
Z = s*randn(Nx,Ny);
F = exp(-(X.^2/(clx^2/2.0)+Y.^2/(cly^2/2.0)));
f = 2.0/sqrt(pi)*Lx/sqrt(Nx*Ny)/sqrt(clx)/sqrt(cly)*ifft(fft(Z).*fft(F'));
perm = exp(mu+real(f))';

imagesc(perm);
colorbar

max(perm(:))/min(perm(:))

xnodes = linspace(-Lx/2,Lx/2,Nxo+1);
ynodes = linspace(-Ly/2,Ly/2,Nyo+1);
xcent = 0.5*(xnodes(1:end-1)+xnodes(2:end));
ycent = 0.5*(ynodes(1:end-1)+ynodes(2:end));
[Xo,Yo] = meshgrid(xcent,ycent);
permo = interp2(X,Y,perm,Xo,Yo);

dlmwrite('true_perm.dat',[Xo(:)+Lx/2,Yo(:)+Ly/2,permo(:)],' ')