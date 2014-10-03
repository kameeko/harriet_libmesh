%model: u_xx - U*u_x - k*u*(1-u) = f(mu); u(0) = 0; u(1) = 1;
%f(mu) = sin(2*pi*x') - cos(4*pi*x') - 2;
%u = state, q = parameter, z = adjoint

clear;

nElemFE = 200; %number of elements for finite element
nNodesEx = 201; %for data-producing "exact" solution (finite difference)
xEx = linspace(0,1,nNodesEx);
dxEx = 1/(nNodesEx-1);

k = 10; %coefficient for reaction term
utol = 10^-10; %tolerance on how negative u can get
umatchtol = 10^-10; %acceptable mismatch between consecutive iterations of u before stopping
qmatchtol = 10^-10; %acceptable mismatch between consecutive iterations of q before stopping

%velocities
Ufd = 2*ones(1,nNodesEx); 
UFE = 2*ones(1,nElemFE+1); 

%for regularization
alpha = 1e-8; 
aRtR = alpha*[1 0 0 0 0; 0 1/2 0 0 0; 0 0 1/2 0 0; 0 0 0 1/2 0; 0 0 0 0 1/2];

%sensor locations (no touching ends!)
spyLoc = [0.1 0.2];

%for Gaussian quadrature
ni = [-1/sqrt(3) 1/sqrt(3)]; %quadrature nodes
wi = [1 1]; %quadrature weights

%% generate data - finite difference; compare with FE (and to add a little noise)

diffOper = diag(ones(nNodesEx-1,1),-1) - 2*diag(ones(nNodesEx,1)) + diag(ones(nNodesEx-1,1),1);
diffOper = diffOper/(dxEx^2);
diffOper([1,end],:) = zeros(2,nNodesEx);
diffOper(1,1) = 1; diffOper(end,end) = 1;

convOper = -repmat(Ufd',1,nNodesEx).*(diag(ones(nNodesEx-1,1),1) - diag(ones(nNodesEx,1)))/dxEx;
convOper([1,end],:) = zeros(2,nNodesEx);
convOper(1,1) = 1; convOper(end,end) = 1;

AExlin = diffOper + convOper;
AExlin(1,1) = 1; AExlin(end,end) = 1;

fEx = [ones(length(xEx),1) sin(2*pi*xEx') cos(2*pi*xEx') ...
    sin(4*pi*xEx') cos(4*pi*xEx')]*[-2;1;0;0;-1];
fEx(1) = 0; fEx(end) = 1; %BCs

uExcd = AExlin\fEx; %as a starting guess
zerome = @(u)(AExlin*u-[0; k*ones(length(u)-2,1); 0].*u.*(1-u)-fEx);

%attempt iterative solution, since lsqnonlin seems unhappy
uPrev = uExcd;
Aiter = AExlin - diag([0; k*ones(nNodesEx-2,1); 0].*uPrev.*(1-uPrev));
uNext = Aiter\fEx;
resNorm = norm(AExlin*uNext-[0; k*ones(length(uNext)-2,1); 0].*uNext.*(1-uNext)-fEx);
cnt = 0;
while cnt < 100 && resNorm > 10^-10
    Aiter = AExlin - diag([0; k*ones(nNodesEx-2,1); 0].*(1-uPrev));
    uPrev = uNext;
    uNext = Aiter\fEx;
    resNorm = norm(AExlin*uNext-[0; k*ones(length(uNext)-2,1); 0].*uNext.*(1-uNext)-fEx);
    
    cnt = cnt + 1;
end
if resNorm > 10^-10; disp(['residual norm of solve: ',num2str(resNorm)]); end
if min(uNext) < -utol || max(uNext) > 1
    disp(['weird concentration: ',num2str(min(uNext)),', ',num2str(max(uNext))]);
end
uEx = uNext;

yd = interp1(xEx,uEx,spyLoc');

%% solve inverse problem (iteratively solve for parameter)

%state equation is A(u)*u = B*q + b, 
%where b is a correction for the Dirichlet boundary conditions;
%A(u) is made of a linear (Alin) and nonlinear part (calculated at every iteration)

dx = 1/nElemFE;
xFE = linspace(0,1,nElemFE+1);

[Alin, snatch] = getAlin(xFE,UFE,UFE,k,ones(1,nElemFE));

%boundary correction from Alin
BCcorrlin = zeros(nElemFE-1,1); 
BCcorrlin(end) = -snatch;

%get basis functions that will make up forcing function
BFE = getB(xFE);

xFE = xFE(2:end-1);

%map state to data
CFE = zeros(length(spyLoc),nElemFE-1);
for ind = 1:length(spyLoc)
	x1 = max(xFE(xFE<=spyLoc(ind)));
    x2 = min(xFE(xFE>=spyLoc(ind)));
    if isempty(x1)
        CFE(ind,1) = spyLoc(ind)/x2;
    elseif isempty(x2)
        CFE(ind,end) = (1-spyLoc(ind))/(1-x1);
    elseif x1 == x2
        CFE(ind,xFE==x1) = 1;
    else
        CFE(ind,xFE==x1) = abs((x2-spyLoc(ind))/(x2-x1));
        CFE(ind,xFE==x2) = abs((spyLoc(ind)-x1)/(x2-x1));
    end
    
end

xFE = [0 xFE 1];

%solve for q iteratively
Oe0 = CFE*(Alin\BFE); %initial guess of Oe
qPrev = (Oe0'*Oe0 + aRtR)\(Oe0'*(yd-CFE*(Alin\BCcorrlin)));
uPrev = Alin\(BFE*qPrev+BCcorrlin);
changeu = 1; changeq = 1;
cnt = 0; 
while cnt < 100 && (changeu > umatchtol || changeq > qmatchtol)
    Aiter = zeros(nElemFE+1, nElemFE+1);
    uPrev = [0; uPrev; fEx(end)]; %add known boundary values to state vector
    
    %linearize nonlinear part of A
    for elem = 1:nElemFE
        n1 = elem; n2 = elem+1;
        x1 = xFE(n1); x2 = xFE(n2);
        bmao2 = 0.5*(x2-x1); bpao2 = 0.5*(x2+x1);
        nitil = bmao2*ni + bpao2;
        Aiter(n1,n1) = Aiter(n1,n1) + k*bmao2*dot(wi,...
            interp1([x1 x2],[uPrev(n1) uPrev(n2)],nitil)...
            .*interp1([x1 x2],[1 0],nitil).*interp1([x1 x2],[1 0],nitil));
        Aiter(n1,n2) = Aiter(n1,n2) + k*bmao2*dot(wi,...
            interp1([x1 x2],[uPrev(n1) uPrev(n2)],nitil)...
            .*interp1([x1 x2],[0 1],nitil).*interp1([x1 x2],[1 0],nitil));
        Aiter(n2,n1) = Aiter(n2,n1) + k*bmao2*dot(wi,...
            interp1([x1 x2],[uPrev(n1) uPrev(n2)],nitil)...
            .*interp1([x1 x2],[1 0],nitil).*interp1([x1 x2],[0 1],nitil));
        Aiter(n2,n2) = Aiter(n2,n2) + k*bmao2*dot(wi,...
            interp1([x1 x2],[uPrev(n1) uPrev(n2)],nitil)...
            .*interp1([x1 x2],[0 1],nitil).*interp1([x1 x2],[0 1],nitil));
    end
    
    %correction for boundary conditions
    BCcorr = BCcorrlin; BCcorr(end) = BCcorr(end) - Aiter(end-1,end);
    
    %overall A
    Aiter = Aiter(2:end-1,2:end-1) + Alin;
    
    Oenext = CFE*(Aiter\BFE);
    
    %solve linearized optimization problem
    qNext = (Oenext'*Oenext + aRtR)\(Oenext'*(yd-CFE*(Aiter\BCcorr)));
    
    %corresponding state
    uNext = Aiter\(BFE*qNext+BCcorr);
    
    %in case state went negative
    if min(uNext) < -utol;
        uNext(uNext < -utol) = -utol;
    end
    
    %to judge if solution sufficiently converged
    changeq = norm(qNext-qPrev);
    changeu = norm(uNext-uPrev(2:end-1));

    qPrev = qNext;
    uPrev = uNext;
    
    cnt = cnt + 1;
end
if changeu > umatchtol || changeq > qmatchtol
    disp(['Sad converge: |u_i-u_i+1|: ',...
        num2str(changeu),' |q_i-q_i+1|: ',num2str(changeq)])
end
A = Aiter;

q1 = qNext;
u1 = uNext; u1 = [0; u1; fEx(end)];
z1 = (CFE/A)'*(CFE*u1(2:end-1)-yd); z1 = [0; z1; 0];

%% solve inverse problem (iteratively solve optimality system)

qaltPrev = (Oe0'*Oe0 + aRtR)\(Oe0'*(yd-CFE*(Alin\BCcorrlin)));
ualtPrev = Alin\(BFE*qaltPrev+BCcorrlin);
zaltPrev = (CFE/Alin)'*(CFE*ualtPrev-yd);
changeu = 1; changeq = 1;
cnt = 0;
while cnt < 100 && (changeu > umatchtol || changeq > qmatchtol)
    Aiter = zeros(nElemFE+1, nElemFE+1);
    ualtPrev = [0; ualtPrev; fEx(end)];
    
    %linearize nonlinear part of A
    for elem = 1:nElemFE
        n1 = elem; n2 = elem+1;
        x1 = xFE(n1); x2 = xFE(n2);
        bmao2 = 0.5*(x2-x1); bpao2 = 0.5*(x2+x1);
        nitil = bmao2*ni + bpao2;
        Aiter(n1,n1) = Aiter(n1,n1) + k*bmao2*dot(wi,...
            interp1([x1 x2],[ualtPrev(n1) ualtPrev(n2)],nitil)...
            .*interp1([x1 x2],[1 0],nitil).*interp1([x1 x2],[1 0],nitil));
        Aiter(n1,n2) = Aiter(n1,n2) + k*bmao2*dot(wi,...
            interp1([x1 x2],[ualtPrev(n1) ualtPrev(n2)],nitil)...
            .*interp1([x1 x2],[0 1],nitil).*interp1([x1 x2],[1 0],nitil));
        Aiter(n2,n1) = Aiter(n2,n1) + k*bmao2*dot(wi,...
            interp1([x1 x2],[ualtPrev(n1) ualtPrev(n2)],nitil)...
            .*interp1([x1 x2],[1 0],nitil).*interp1([x1 x2],[0 1],nitil));
        Aiter(n2,n2) = Aiter(n2,n2) + k*bmao2*dot(wi,...
            interp1([x1 x2],[ualtPrev(n1) ualtPrev(n2)],nitil)...
            .*interp1([x1 x2],[0 1],nitil).*interp1([x1 x2],[0 1],nitil));
    end
    
    %correction for boundary conditions
    BCcorr = BCcorrlin; BCcorr(end) = BCcorr(end) - Aiter(end-1,end);
    
    %overall linearized A
    Aiter = Aiter(2:end-1,2:end-1) + Alin;
    
    np = 5; %number of parameters
    n = nElemFE - 1; %number of unknown state DOF
    
    %form optimality system
    bigjac = zeros(np+n+n);
    bigrhs = zeros(np+n+n,1);
    
    %state equation
    bigjac(1:n,1:np) = BFE;
    bigjac(1:n,np+1:np+n) = -Aiter;
    bigrhs(1:n) = -BCcorr;
    
    %gradient equation
    bigjac(n+1:n+np,1:np) = aRtR;
    bigjac(n+1:n+np,np+n+1:end) = BFE';
    
    %adjoint equation
    bigjac(n+np+1:end,np+1:np+n) = CFE'*CFE;
    bigjac(n+np+1:end,np+n+1:end) = -Aiter';
    bigrhs(n+np+1:end) = CFE'*yd;
    
    %solve optimality system
    meep = bigjac\bigrhs;
    qaltNext = meep(1:np);
    ualtNext = meep(np+1:np+n);
    zaltNext = meep(np+n+1:end);
    if min(ualtNext) < -utol;
        ualtNext(ualtNext < -utol) = -utol;
    end
    
    changeq = norm(qaltNext-qaltPrev);
    changeu = norm(ualtNext-ualtPrev(2:end-1));
    changez = norm(zaltNext-zaltPrev);
    
    qaltPrev = qaltNext;
    ualtPrev = ualtNext;
    zaltPrev = zaltNext;
    
    cnt = cnt + 1;
end
qalt = qaltNext;
ualt = ualtNext;
zalt = zaltNext;

ualt = [0; ualt; fEx(end)];
zalt = [0; zalt; 0];