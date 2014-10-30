function [Alin,snatch] = getAlin(xFE,ULF,UHF,k,upgradeMe)

nElem = length(xFE)-1;
ni = [-1/sqrt(3) 1/sqrt(3)]; %quadrature nodes
wi = [1 1]; %quadrature weights

Acd = zeros(nElem+1, nElem+1);
Arl = zeros(nElem+1, nElem+1);
%for stiffness matrix
for elem = 1:nElem,
    
    n1 = elem;
    n2 = elem+1;
    
    x1 = xFE(n1);
    x2 = xFE(n2);
    
    dx = x2 - x1;
    
    %contributions from diffusion part
    Acd(n1, n1) = Acd(n1, n1) - (1/dx);
    Acd(n1, n2) = Acd(n1, n2) + (1/dx);
    Acd(n2, n1) = Acd(n2, n1) + (1/dx);
    Acd(n2, n2) = Acd(n2, n2) - (1/dx);
	

    %contributions from convection part
    bmao2 = 0.5*(x2-x1);
    bpao2 = 0.5*(x2+x1);
    nitil = bmao2*ni + bpao2;
	if upgradeMe(elem)
    Acd(n1,n1) = Acd(n1,n1) + (1/dx)*bmao2*dot(wi,...
        interp1([x1 x2],[UHF(elem) UHF(elem+1)],nitil).*interp1([x1 x2],[1 0],nitil));
    Acd(n1,n2) = Acd(n1,n2) - (1/dx)*bmao2*dot(wi,...
        interp1([x1 x2],[UHF(elem) UHF(elem+1)],nitil).*interp1([x1 x2],[1 0],nitil));
    Acd(n2,n1) = Acd(n2,n1) + (1/dx)*bmao2*dot(wi,...
        interp1([x1 x2],[UHF(elem) UHF(elem+1)],nitil).*interp1([x1 x2],[0 1],nitil));
    Acd(n2,n2) = Acd(n2,n2) - (1/dx)*bmao2*dot(wi,...
        interp1([x1 x2],[UHF(elem) UHF(elem+1)],nitil).*interp1([x1 x2],[0 1],nitil));
	else
	Acd(n1,n1) = Acd(n1,n1) + (1/dx)*bmao2*dot(wi,...
        interp1([x1 x2],[ULF(elem) ULF(elem+1)],nitil).*interp1([x1 x2],[1 0],nitil));
    Acd(n1,n2) = Acd(n1,n2) - (1/dx)*bmao2*dot(wi,...
        interp1([x1 x2],[ULF(elem) ULF(elem+1)],nitil).*interp1([x1 x2],[1 0],nitil));
    Acd(n2,n1) = Acd(n2,n1) + (1/dx)*bmao2*dot(wi,...
        interp1([x1 x2],[ULF(elem) ULF(elem+1)],nitil).*interp1([x1 x2],[0 1],nitil));
    Acd(n2,n2) = Acd(n2,n2) - (1/dx)*bmao2*dot(wi,...
        interp1([x1 x2],[ULF(elem) ULF(elem+1)],nitil).*interp1([x1 x2],[0 1],nitil));
	end
    
    if upgradeMe(elem)
    %contributions from linear part of reaction term
    bmao2 = 0.5*(x2-x1);
    bpao2 = 0.5*(x2+x1);
    nitil = bmao2*ni + bpao2;
    Arl(n1,n1) = Arl(n1,n1) - k*bmao2*dot(wi,...
        interp1([x1 x2],[1 0],nitil).*interp1([x1 x2],[1 0],nitil));
    Arl(n1,n2) = Arl(n1,n2) - k*bmao2*dot(wi,...
        interp1([x1 x2],[0 1],nitil).*interp1([x1 x2],[1 0],nitil));
    Arl(n2,n1) = Arl(n2,n1) - k*bmao2*dot(wi,...
        interp1([x1 x2],[1 0],nitil).*interp1([x1 x2],[0 1],nitil));
    Arl(n2,n2) = Arl(n2,n2) - k*bmao2*dot(wi,...
        interp1([x1 x2],[0 1],nitil).*interp1([x1 x2],[0 1],nitil));
    end
end
snatch = Acd(end-1,end) + Arl(end-1,end);
Alin = Acd + Arl;
Alin = Alin(2:end-1,2:end-1); % Set Dirichlet conditions x(0) = 0, x(1) = 0