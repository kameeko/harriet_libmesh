%multi-model adjoint poke

%low-fidelity model: u_xx - U*u_x = f(mu); u(0) = 0, u(1) = 1;
%high-fidelity model: u_xx - U*u_x - k*u*(1-u) = f(mu); u(0) = 0; u(1) = 1;
%f(mu) = sin(2*pi*x') - cos(4*pi*x') - 2;

%five basis functions HARDCODED into B,R for now

clear; close all;

nElemLF = 200; %for low- and mixed-fidelity
nElemHF = 1*nElemLF; %keep to whole multiples of nElemLF
nNodesEx = 201; %for data-producing "exact" solution
xEx = linspace(0,1,nNodesEx);
dxEx = 1/(nNodesEx-1);
xFELF = linspace(0,1,nElemLF+1);
xFEHF = linspace(0,1,nElemHF+1);

k = 10; %coefficient for reaction term
utol = 10^-10; %tolerance on how negative u can get
umatchtol = 10^-12; %acceptable mismatch between consecutive iterations of u before stopping
qmatchtol = 10^-12; %acceptable mismatch between consecutive iterations of q before stopping

%velocities
Ufd = 2*ones(1,nNodesEx); 
ULF = 2*ones(1,nElemLF+1); 
UHF = 2*ones(1,nElemHF+1); 
Uenr = 2*ones(1,nElemLF+1); %for superadjoint

linPred = 0; %linear (1) or nonlinear (0) prediction

%use HF model in these intervals to estimate xHF
% mixints{1,1} = [0]; mixints{1,2} = [1];
%diff vs conv+diff
% mixints{1,1} = []; mixints{1,2} = []; 
% mixints{2,1} = [0.7]; mixints{2,2} = [0.9]; %further mixes by approximated blaming
% mixints{3,1} = [0 0.7]; mixints{3,2} = [0.1 0.9];
% mixints{4,1} = [0 0.7]; mixints{4,2} = [0.2 0.9];
% mixints{5,1} = [0 0.6]; mixints{5,2} = [0.25 1];
% mixints{6,1} = [0]; mixints{6,2} = [1];
%diff(+conv) vs conv+diff+react
% mixints{1,1} = []; mixints{1,2} = []; 
% mixints{2,1} = [0.7]; mixints{2,2} = [0.9]; %further mixes by approximated blaming
% mixints{3,1} = [0.1 0.7]; mixints{3,2} = [0.2 0.9];
% mixints{4,1} = [0 0.7]; mixints{4,2} = [0.2 0.9];
% mixints{5,1} = [0 0.6]; mixints{5,2} = [0.25 1];
% mixints{6,1} = [0]; mixints{6,2} = [1];
mixints{1,1} = [0]; mixints{1,2} = [1]; %DEBUG

alpha = 1e-8; %for regularization
aRtR = alpha*[1 0 0 0 0; 0 1/2 0 0 0; 0 0 1/2 0 0; 0 0 0 1/2 0; 0 0 0 0 1/2];

%sensor locations (no touching ends!)
spyLoc = [0.1 0.2];

%interval in which to integrate for prediction (no touching ends!)
x1p = 0.7;
x2p = 0.9;

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

ypGod = 0;
if linPred
    ypGod = trapz(xEx(xEx >= x1p & xEx <= x2p),uEx(xEx >= x1p & xEx <= x2p));
else
for ii = 1:nNodesEx-1
    x1 = xEx(ii); x2 = xEx(ii+1);
	if (x1 >= x1p && x2 <= x2p) || (x1 <= x1p && x2 >= x1p) || (x1 <= x2p && x2 >= x2p)
		x1 = max(x1,x1p); x2 = min(x2,x2p);
		bmao2 = 0.5*(x2-x1); bpao2 = 0.5*(x2+x1);
		nitil = bmao2*ni + bpao2;
		ypGod = ypGod + bmao2*dot(wi,...
			interp1([xEx(ii) xEx(ii+1)],[uEx(ii) uEx(ii+1)],nitil).^2);
	end
end 
end

% plot(xEx,uEx,xEx,(AExlin\fEx),xEx,diffOper\fEx)

%% low-fidelity model

[ALF, snatch] = getAlin(xFELF,ULF,interp1(xFEHF,UHF,xFELF),k,zeros(1,nElemLF));

BCcorr = zeros(nElemLF-1,1); 
BCcorr(end) = -snatch;

%form right-hand side
BLF = getB(xFELF);

xFELF = xFELF(2:end-1);

CFELF = zeros(length(spyLoc),nElemLF-1);
for ind = 1:length(spyLoc)
	x1 = max(xFELF(xFELF<=spyLoc(ind)));
    x2 = min(xFELF(xFELF>=spyLoc(ind)));
    if isempty(x1)
        CFELF(ind,1) = spyLoc(ind)/x2;
    elseif isempty(x2)
        CFELF(ind,end) = (1-spyLoc(ind))/(1-x1);
    elseif x1 == x2
        CFELF(ind,xFELF==x1) = 1;
    else
        CFELF(ind,xFELF==x1) = abs((x2-spyLoc(ind))/(x2-x1));
        CFELF(ind,xFELF==x2) = abs((spyLoc(ind)-x1)/(x2-x1));
    end
    
end

OeLF = CFELF*(ALF\BLF);

qLF = (OeLF'*OeLF + aRtR)\(OeLF'*(yd-CFELF*(ALF\BCcorr)));
uLF = ALF\(BLF*qLF+BCcorr);
zLF = (CFELF/ALF)'*(CFELF*uLF-yd);

xFELF = [0 xFELF 1];
H = OeLF'*OeLF + aRtR; 
w = ALF\BLF;
uLF = [0; uLF; fEx(end)];

if linPred
Iu_uLFqLF = zeros(1,nElemLF-1);
xFEtmp = xFELF(2:end-1);
dx = 1/nElemLF;
xind1 = ceil(x1p/dx);
xind2 = floor(x2p/dx);
Iu_uLFqLF(xind1+1:xind2-1) = dx;
Iu_uLFqLF(xind1) = 0.5*dx + ...
    0.5*abs(xFEtmp(xind1)-x1p)*(1+abs(x1p-xFEtmp(xind1-1))/dx);
Iu_uLFqLF(xind1-1) = 0.5*abs(xFEtmp(xind1)-x1p)*(abs(xFEtmp(xind1)-x1p)/dx);
Iu_uLFqLF(xind2) = 0.5*dx + ...
    0.5*abs(x2p-xFEtmp(xind2))*(1+abs(xFEtmp(xind2+1)-x2p)/dx);
Iu_uLFqLF(xind2+1) = 0.5*abs(x2p-xFEtmp(xind2))*(abs(x2p-xFEtmp(xind2))/dx);
else
Iu_uLFqLF = zeros(1,length(uLF));
for ii = 1:nElemLF
	x1 = xFELF(ii); x2 = xFELF(ii+1);
	if (x1 >= x1p && x2 <= x2p) || (x1 <= x1p && x2 >= x1p) || (x1 <= x2p && x2 >= x2p)
		a = x1; b = x2;
		x1 = max(x1,x1p); x2 = min(x2,x2p);
		bmao2 = 0.5*(x2-x1); bpao2 = 0.5*(x2+x1);
		nitil = bmao2*ni + bpao2;
		uatn = interp1([xFELF(ii) xFELF(ii+1)],[uLF(ii) uLF(ii+1)],nitil);
		Iu_uLFqLF(ii) = Iu_uLFqLF(ii) + ...
			bmao2*wi(1)*uatn(1)*(1-(nitil(1)-a)/(b-a)) + ...
			bmao2*wi(2)*uatn(2)*(1-(nitil(2)-a)/(b-a));
		Iu_uLFqLF(ii+1) = Iu_uLFqLF(ii+1) + ...
			bmao2*wi(1)*uatn(1)*((nitil(1)-a)/(b-a)) + ...
			bmao2*wi(2)*uatn(2)*((nitil(2)-a)/(b-a));
	end
end
Iu_uLFqLF = 2*Iu_uLFqLF(2:end-1);
end
g = -(Iu_uLFqLF*w)';

pLF = H\g;
vLF = w*pLF;
yLF = ALF'\(Iu_uLFqLF'+CFELF'*CFELF*vLF);

% figure; plot(xEx,diffOper\fEx,xFE,ALF\(BLF*-[-2;1;0;0;-1]+BCcorr)) %check

%% high-fidelity model

dx = 1/nElemHF;
xFEHF = linspace(0,1,nElemHF+1);

[AHFlin, snatch] = getAlin(xFEHF,interp1(xFELF,ULF,xFEHF),UHF,k,ones(1,nElemHF));
BCcorrlinHF = zeros(nElemHF-1,1); 
BCcorrlinHF(end) = -snatch;

BHF = getB(xFEHF);

xFEHF = xFEHF(2:end-1);
CFEHF = zeros(length(spyLoc),nElemHF-1);
for ind = 1:length(spyLoc)
	x1 = max(xFEHF(xFEHF<=spyLoc(ind)));
    x2 = min(xFEHF(xFEHF>=spyLoc(ind)));
    if isempty(x1)
        CFEHF(ind,1) = spyLoc(ind)/x2;
    elseif isempty(x2)
        CFEHF(ind,end) = (1-spyLoc(ind))/(1-x1);
    elseif x1 == x2
        CFEHF(ind,xFEHF==x1) = 1;
    else
        CFEHF(ind,xFEHF==x1) = abs((x2-spyLoc(ind))/(x2-x1));
        CFE(ind,xFEHF==x2) = abs((spyLoc(ind)-x1)/(x2-x1));
    end
    
end
xFEHF = [0 xFEHF 1];

%solve for qHF iteratively
OeHF0 = CFEHF*(AHFlin\BHF);
qPrev = (OeHF0'*OeHF0 + aRtR)\(OeHF0'*(yd-CFEHF*(AHFlin\BCcorrlinHF)));
uPrev = AHFlin\(BHF*qPrev+BCcorrlinHF);
% qPrev = qLF; uPrev = uLF(2:end-1); %DEBUG
changeu = 1; changeq = 1;
cnt = 0;
while cnt < 100 && (changeu > umatchtol || changeq > qmatchtol)
    Aiter = zeros(nElemHF+1, nElemHF+1);
    uPrev = [0; uPrev; fEx(end)];
    for elem = 1:nElemHF
        n1 = elem; n2 = elem+1;
        x1 = xFEHF(n1); x2 = xFEHF(n2);
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
    BCcorr = BCcorrlinHF; BCcorr(end) = BCcorr(end) - Aiter(end-1,end);
    Aiter = Aiter(2:end-1,2:end-1) + AHFlin;
    OeHFnext = CFEHF*(Aiter\BHF);
    
    qNext = (OeHFnext'*OeHFnext + aRtR)\(OeHFnext'*(yd-CFEHF*(Aiter\BCcorr)));
    uNext = Aiter\(BHF*qNext+BCcorr);
    if min(uNext) < -utol;
        uNext(uNext < -utol) = -utol;
    end
    
    changeq = norm(qNext-qPrev);
    changeu = norm(uNext-uPrev(2:end-1));

    qPrev = qNext;
    uPrev = uNext;
    
    cnt = cnt + 1;
end
if changeu > umatchtol || changeq > qmatchtol
    disp(['HF optimality conditions have sad converge: |u_i-u_i+1|: ',...
        num2str(changeu),' |q_i-q_i+1|: ',num2str(changeq)])
end
AHF = Aiter;

OeHF = CFEHF*(AHF\BHF);

qHF = qNext;
uHF = uNext; uHF = [0; uHF; fEx(end)];
zHF = (CFEHF/AHF)'*(CFEHF*uHF(2:end-1)-yd); zHF = [0; zHF; 0];

%to get Hessian
[A2u, A2z] = getA2s(xFEHF,k,uHF,zHF,ones(nElemHF,1));
Oetmp = CFEHF*((AHFlin + 2*A2u)\BHF);
H = aRtR + Oetmp'*Oetmp - ((AHFlin + 2*A2u)\BHF)'*(2*A2z*((AHFlin + 2*A2u)\BHF));

%bit of a'_u from nonlinear reaction term
auprimer = zeros(nElemHF+1,nElemHF+1);
for elem = 1:nElemHF
    n1 = elem; n2 = elem+1;
    x1 = xFEHF(n1); x2 = xFEHF(n2);
    
    bmao2 = 0.5*(x2-x1);
    bpao2 = 0.5*(x2+x1);
    nitil = bmao2*ni + bpao2;
    auprimer(n1,n1) = auprimer(n1,n1) + 2*k*bmao2*dot(wi,...
        interp1([x1 x2],[uHF(n1) uHF(n2)],nitil)...
        .*interp1([x1 x2],[1 0],nitil).*interp1([x1 x2],[1 0],nitil));
    auprimer(n1,n2) = auprimer(n1,n2) + 2*k*bmao2*dot(wi,...
        interp1([x1 x2],[uHF(n1) uHF(n2)],nitil)...
        .*interp1([x1 x2],[0 1],nitil).*interp1([x1 x2],[1 0],nitil));
    auprimer(n2,n1) = auprimer(n2,n1) + 2*k*bmao2*dot(wi,...
        interp1([x1 x2],[uHF(n1) uHF(n2)],nitil)...
        .*interp1([x1 x2],[1 0],nitil).*interp1([x1 x2],[0 1],nitil));
    auprimer(n2,n2) = auprimer(n2,n2) + 2*k*bmao2*dot(wi,...
        interp1([x1 x2],[uHF(n1) uHF(n2)],nitil)...
        .*interp1([x1 x2],[0 1],nitil).*interp1([x1 x2],[0 1],nitil));
end

auprimequ = AHFlin + auprimer(2:end-1,2:end-1);
w = auprimequ\BHF;

if linPred
Iu_uHFqHF = zeros(1,nElemHF-1);
xFEtmp = xFEHF(2:end-1);
dx = 1/nElemHF;
xind1 = ceil(x1p/dx);
xind2 = floor(x2p/dx);
Iu_uHFqHF(xind1+1:xind2-1) = dx;
Iu_uHFqHF(xind1) = 0.5*dx + ...
    0.5*abs(xFEtmp(xind1)-x1p)*(1+abs(x1p-xFEtmp(xind1-1))/dx);
Iu_uHFqHF(xind1-1) = 0.5*abs(xFEtmp(xind1)-x1p)*(abs(xFEtmp(xind1)-x1p)/dx);
Iu_uHFqHF(xind2) = 0.5*dx + ...
    0.5*abs(x2p-xFEtmp(xind2))*(1+abs(xFEtmp(xind2+1)-x2p)/dx);
Iu_uHFqHF(xind2+1) = 0.5*abs(x2p-xFEtmp(xind2))*(abs(x2p-xFEtmp(xind2))/dx);
else
Iu_uHFqHF = zeros(1,length(uHF));    
for ii = 1:nElemHF
	x1 = xFEHF(ii); x2 = xFEHF(ii+1);
	if (x1 >= x1p && x2 <= x2p) || (x1 <= x1p && x2 >= x1p) || (x1 <= x2p && x2 >= x2p)
		a = x1; b = x2;
		x1 = max(x1,x1p); x2 = min(x2,x2p);
		bmao2 = 0.5*(x2-x1); bpao2 = 0.5*(x2+x1);
		nitil = bmao2*ni + bpao2;
		uatn = interp1([xFEHF(ii) xFEHF(ii+1)],[uHF(ii) uHF(ii+1)],nitil);
		Iu_uHFqHF(ii) = Iu_uHFqHF(ii) + ...
			bmao2*wi(1)*uatn(1)*(1-(nitil(1)-a)/(b-a)) + ...
			bmao2*wi(2)*uatn(2)*(1-(nitil(2)-a)/(b-a));
		Iu_uHFqHF(ii+1) = Iu_uHFqHF(ii+1) + ...
			bmao2*wi(1)*uatn(1)*((nitil(1)-a)/(b-a)) + ...
			bmao2*wi(2)*uatn(2)*((nitil(2)-a)/(b-a));
	end
end
Iu_uHFqHF = 2*Iu_uHFqHF(2:end-1);
end
g = -(Iu_uHFqHF*w)';

pHF = H\g; 
vHF = w*pHF; 
vHF = [0; vHF; 0];

nauu = zeros(1,length(uHF)); %-a"_uu
for ii = 1:nElemHF
    x1 = xFEHF(ii); x2 = xFEHF(ii+1);
    bmao2 = 0.5*(x2-x1); bpao2 = 0.5*(x2+x1);
	nitil = bmao2*ni + bpao2;
    vatn = interp1([x1 x2],[vHF(ii) vHF(ii+1)],nitil);
    zatn = interp1([x1 x2],[zHF(ii) zHF(ii+1)],nitil);
    nauu(ii) = nauu(ii) - ...
        bmao2*wi(1)*vatn(1)*zatn(1)*(1-(nitil(1)-x1)/(x2-x1)) - ...
		bmao2*wi(2)*vatn(2)*zatn(2)*(1-(nitil(2)-x1)/(x2-x1));
    nauu(ii+1) = nauu(ii+1) - ...
        bmao2*wi(1)*vatn(1)*zatn(1)*((nitil(1)-x1)/(x2-x1)) - ...
		bmao2*wi(2)*vatn(2)*zatn(2)*((nitil(2)-x1)/(x2-x1));
end
nauu = 2*k*nauu(2:end-1);
yHF = auprimequ'\(Iu_uHFqHF'+nauu'+CFEHF'*CFEHF*vHF(2:end-1));
yHF = [0; yHF; 0];

%% prediction error (LF vs HF)

if linPred
    ypLF = Iu_uLFqLF*uLF(2:end-1);
    ypHF = Iu_uHFqHF*uHF(2:end-1);
else
    ypLF = 0;
    ypHF = 0;
    if length(uLF) < length(uHF)
        uLFproj = interp1(xFELF,uLF,xFEHF);
    else
        uLFproj = uLF;
    end
    for ii = 1:nElemHF
        x1 = xFEHF(ii); x2 = xFEHF(ii+1);
        if (x1 >= x1p && x2 <= x2p) || (x1 <= x1p && x2 >= x1p) || (x1 <= x2p && x2 >= x2p)
            x1 = max(x1,x1p); x2 = min(x2,x2p);
            bmao2 = 0.5*(x2-x1); bpao2 = 0.5*(x2+x1);
            nitil = bmao2*ni + bpao2;
            ypLF = ypLF + bmao2*dot(wi,...
                interp1([xFEHF(ii) xFEHF(ii+1)],[uLFproj(ii) uLFproj(ii+1)],nitil).^2);
            ypHF = ypHF + bmao2*dot(wi,...
                interp1([xFEHF(ii) xFEHF(ii+1)],[uHF(ii) uHF(ii+1)],nitil).^2);
        end
    end
end
ypdiff = ypHF - ypLF;

%% smoothies!

figure;
pcntConverted = zeros(1,size(mixints,1));
ypdiffs_est = zeros(1,size(mixints,1));
ypdiffs = zeros(1,size(mixints,1));
uStash = zeros(1,length(uLF));

markers = {'bo','ro','ko','mo','go','co'};
markersUp = {'b*','r*','k*','m*','g*','c*'};

for jj = 1:size(mixints,1)
    
xFE = linspace(0,1,nElemLF+1);

xmixl = mixints{jj,1};
xmixr = mixints{jj,2};

upgradeMe = zeros(1,nElemLF);
for ii = 1:length(xmixl)
    leftind = find(xFE >= xmixl(ii),1,'first');
    rightind = find(xFE <= xmixr(ii),1,'last');
    upgradeMe(leftind:rightind-1) = ones(1,rightind-leftind);
end

[AMFlin, snatch] = getAlin(xFE,ULF,interp1(xFEHF,UHF,xFELF),k,upgradeMe);
BCcorrlinMF = zeros(nElemLF-1,1); 
BCcorrlinMF(end) = -snatch; 

BMF = BLF;

%solve for qMF iteratively
OeMF0 = CFELF*(AMFlin\BMF);
qPrev = (OeMF0'*OeMF0 + aRtR)\(OeMF0'*(yd-CFELF*(AMFlin\BCcorrlinMF)));
uPrev = AMFlin\(BMF*qPrev+BCcorrlinMF);
changeu = 1; changeq = 1;
cnt = 0;
while cnt < 100 && (changeu > umatchtol || changeq > qmatchtol)
    Aiter = zeros(nElemLF+1, nElemLF+1);
    uPrev = [0; uPrev; fEx(end)];
    for elem = 1:nElemLF
        if upgradeMe(elem)
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
    end
    BCcorr = BCcorrlinMF; BCcorr(end) = BCcorr(end) - Aiter(end-1,end);
    Aiter = Aiter(2:end-1,2:end-1) + AMFlin;
    OeMFnext = CFELF*(Aiter\BMF);
    
    qNext = (OeMFnext'*OeMFnext + aRtR)\(OeMFnext'*(yd-CFELF*(Aiter\BCcorr)));
    uNext = Aiter\(BMF*qNext+BCcorr);
    if min(uNext) < -utol;
        uNext(uNext < -utol) = -utol;
    end
    
    changeq = norm(qNext-qPrev);
    changeu = norm(uNext-uPrev(2:end-1));
    
    qPrev = qNext;
    uPrev = uNext;
    
    cnt = cnt + 1;
end
if changeu > umatchtol || changeq > qmatchtol
    disp(['MF optimality conditions have sad converge: |u_i-u_i+1|: ',...
        num2str(changeu),' |q_i-q_i+1|: ',num2str(changeq)])
end
AMF = Aiter;
OeMF = CFELF*(AMF\BMF);

qMF = qNext;
uMF = uNext;
zMF = (CFELF/AMF)'*(CFELF*uMF-yd);

uMF = [0; uMF; fEx(end)];
zMF = [0; zMF; 0];

%to get Hessian
[A2u, A2z] = getA2s(xFE,k,uMF,zMF,upgradeMe);
Oetmp = CFELF*((AMFlin + 2*A2u)\BMF);
H = aRtR + Oetmp'*Oetmp - ((AMFlin + 2*A2u)\BMF)'*(2*A2z*((AMFlin + 2*A2u)\BMF));

%bit of a'_u from nonlinear reaction term
auprimer = zeros(nElemLF+1,nElemLF+1);
for elem = 1:nElemLF
    if upgradeMe(elem)
    n1 = elem; n2 = elem+1;
    x1 = xFE(n1); x2 = xFE(n2);
    
    bmao2 = 0.5*(x2-x1);
    bpao2 = 0.5*(x2+x1);
    nitil = bmao2*ni + bpao2;
    auprimer(n1,n1) = auprimer(n1,n1) + 2*k*bmao2*dot(wi,...
        interp1([x1 x2],[uMF(n1) uMF(n2)],nitil)...
        .*interp1([x1 x2],[1 0],nitil).*interp1([x1 x2],[1 0],nitil));
    auprimer(n1,n2) = auprimer(n1,n2) + 2*k*bmao2*dot(wi,...
        interp1([x1 x2],[uMF(n1) uMF(n2)],nitil)...
        .*interp1([x1 x2],[0 1],nitil).*interp1([x1 x2],[1 0],nitil));
    auprimer(n2,n1) = auprimer(n2,n1) + 2*k*bmao2*dot(wi,...
        interp1([x1 x2],[uMF(n1) uMF(n2)],nitil)...
        .*interp1([x1 x2],[1 0],nitil).*interp1([x1 x2],[0 1],nitil));
    auprimer(n2,n2) = auprimer(n2,n2) + 2*k*bmao2*dot(wi,...
        interp1([x1 x2],[uMF(n1) uMF(n2)],nitil)...
        .*interp1([x1 x2],[0 1],nitil).*interp1([x1 x2],[0 1],nitil));
    end
end

auprimequ = AMFlin + auprimer(2:end-1,2:end-1);
w = auprimequ\BMF;
if linPred
Iu_uMFqMF = zeros(1,nElemLF-1);
xFEtmp = xFE(2:end-1);
dx = 1/nElemLF;
xind1 = ceil(x1p/dx);
xind2 = floor(x2p/dx);
Iu_uMFqMF(xind1+1:xind2-1) = dx;
Iu_uMFqMF(xind1) = 0.5*dx + ...
    0.5*abs(xFEtmp(xind1)-x1p)*(1+abs(x1p-xFEtmp(xind1-1))/dx);
Iu_uMFqMF(xind1-1) = 0.5*abs(xFEtmp(xind1)-x1p)*(abs(xFEtmp(xind1)-x1p)/dx);
Iu_uMFqMF(xind2) = 0.5*dx + ...
    0.5*abs(x2p-xFEtmp(xind2))*(1+abs(xFEtmp(xind2+1)-x2p)/dx);
Iu_uMFqMF(xind2+1) = 0.5*abs(x2p-xFEtmp(xind2))*(abs(x2p-xFEtmp(xind2))/dx);
else
Iu_uMFqMF = zeros(1,length(uMF));
for ii = 1:nElemLF
	x1 = xFE(ii); x2 = xFE(ii+1);
	if (x1 >= x1p && x2 <= x2p) || (x1 <= x1p && x2 >= x1p) || (x1 <= x2p && x2 >= x2p)
		a = x1; b = x2;
		x1 = max(x1,x1p); x2 = min(x2,x2p);
		bmao2 = 0.5*(x2-x1); bpao2 = 0.5*(x2+x1);
		nitil = bmao2*ni + bpao2;
		uatn = interp1([xFE(ii) xFE(ii+1)],[uMF(ii) uMF(ii+1)],nitil);
		Iu_uMFqMF(ii) = Iu_uMFqMF(ii) + ...
			bmao2*wi(1)*uatn(1)*(1-(nitil(1)-a)/(b-a)) + ...
			bmao2*wi(2)*uatn(2)*(1-(nitil(2)-a)/(b-a));
		Iu_uMFqMF(ii+1) = Iu_uMFqMF(ii+1) + ...
			bmao2*wi(1)*uatn(1)*((nitil(1)-a)/(b-a)) + ...
			bmao2*wi(2)*uatn(2)*((nitil(2)-a)/(b-a));
	end
end
Iu_uMFqMF = 2*Iu_uMFqMF(2:end-1);
end
g = -(Iu_uMFqMF*w)';

pMF = H\g; 
vMF = w*pMF; 
vMF = [0; vMF; 0];

nauu = zeros(1,length(uMF)); %-a"_uu
for ii = 1:nElemLF
    if upgradeMe(ii)
    x1 = xFE(ii); x2 = xFE(ii+1);
    bmao2 = 0.5*(x2-x1); bpao2 = 0.5*(x2+x1);
	nitil = bmao2*ni + bpao2;
    vatn = interp1([x1 x2],[vMF(ii) vMF(ii+1)],nitil);
    zatn = interp1([x1 x2],[zMF(ii) zMF(ii+1)],nitil);
    nauu(ii) = nauu(ii) - ...
        bmao2*wi(1)*vatn(1)*zatn(1)*(1-(nitil(1)-x1)/(x2-x1)) - ...
		bmao2*wi(2)*vatn(2)*zatn(2)*(1-(nitil(2)-x1)/(x2-x1));
    nauu(ii+1) = nauu(ii+1) - ...
        bmao2*wi(1)*vatn(1)*zatn(1)*((nitil(1)-x1)/(x2-x1)) - ...
		bmao2*wi(2)*vatn(2)*zatn(2)*((nitil(2)-x1)/(x2-x1));
    end
end
nauu = 2*k*nauu(2:end-1);

yMF = auprimequ'\(Iu_uMFqMF'+nauu'+CFELF'*CFELF*vMF(2:end-1));
yMF = [0; yMF; 0];

% figure; plot(xEx,uEx,xFE(2:end-1),AMF\(BMF*-[-2;1;0;0;-1]+Bccorr)) %check
uStash(jj,:) = uMF';

%%%%%%%% M_HF(x_HF)-M_HF(x_MF) (adjoint poke) %%%%%%%%

nElemadj = 1*nElemLF;
dxenr = 1/nElemadj;
xenr = linspace(0,1,nElemadj+1);

%%% solve for superadjoint

[Aadjlin, snatch] = getAlin(xenr,interp1(xFELF,ULF,xenr),Uenr,k,ones(1,nElemadj));

uMFproj = interp1(xFE,uMF,xenr');
zMFproj = interp1(xFE,zMF,xenr');
vMFproj = interp1(xFE,vMF,xenr');
yMFproj = interp1(xFE,yMF,xenr');
uMFprojx = (uMFproj(2:end)-uMFproj(1:end-1))./(xenr(2:end)'-xenr(1:end-1)');
zMFprojx = (zMFproj(2:end)-zMFproj(1:end-1))./(xenr(2:end)'-xenr(1:end-1)');
vMFprojx = (vMFproj(2:end)-vMFproj(1:end-1))./(xenr(2:end)'-xenr(1:end-1)');
yMFprojx = (yMFproj(2:end)-yMFproj(1:end-1))./(xenr(2:end)'-xenr(1:end-1)');

BCcorrlinadj = zeros(nElemadj-1,1); 
BCcorrlinadj(end) = -snatch; 

[A2uadj, A2zadj, snatch] = getA2s(xenr,k,uMFproj,zMFproj,ones(nElemadj,1));

BCcorr = BCcorrlinadj; BCcorr(end) = BCcorr(end) - snatch;

Benr = getB(xenr);

xenr = xenr(2:end-1);
Cenr = zeros(length(spyLoc),nElemadj-1);
for ind = 1:length(spyLoc)
	x1 = max(xenr(xenr<=spyLoc(ind)));
    x2 = min(xenr(xenr>=spyLoc(ind)));
    if isempty(x1)
        Cenr(ind,1) = spyLoc(ind)/x2;
    elseif isempty(x2)
        Cenr(ind,end) = (1-spyLoc(ind))/(1-x1);
    elseif x1 == x2
        Cenr(ind,xenr==x1) = 1;
    else
        Cenr(ind,xenr==x1) = abs((x2-spyLoc(ind))/(x2-x1));
        Cenr(ind,xenr==x2) = abs((spyLoc(ind)-x1)/(x2-x1));
    end
end

n = nElemadj - 1;
np = 5;
Asuper = zeros(2*np+4*n);
Asuper(1:np,1:np) = aRtR;
Asuper(1:np,np+n+1:np+n+n) = Benr';
Asuper(np+1:np+n,np+1:np+n) = Cenr'*Cenr;
Asuper(np+1:np+n,np+n+1:np+n+n) = -Aadjlin' - 2*A2uadj';
Asuper(np+n+1:np+n+n,1:np) = Benr;
Asuper(np+n+1:np+n+n,np+1:np+n) = -Aadjlin-A2uadj;
Asuper(np+n+n+1:end-n-n,np+n+n+1:end-n-n) = aRtR;
Asuper(np+n+n+1:end-n-n,end-n+1:end) = Benr';
Asuper(end-n-n+1:end-n,end-n-n+1:end-n) = Cenr'*Cenr - 2*A2zadj';
Asuper(end-n-n+1:end-n,end-n+1:end) = -Aadjlin';
Asuper(end-n+1:end,np+n+n+1:end-n-n) = Benr;
Asuper(end-n+1:end,end-n-n+1:end-n) = -Aadjlin - 2*A2uadj;

xenr = [0 xenr 1];

if linPred
Iu_uMFqMFadj = zeros(1,nElemadj-1);
xFEtmp = xenr(2:end-1);
dx = 1/nElemadj;
xind1 = ceil(x1p/dx);
xind2 = floor(x2p/dx);
Iu_uMFqMFadj(xind1+1:xind2-1) = dx;
Iu_uMFqMFadj(xind1) = 0.5*dx + ...
    0.5*abs(xFEtmp(xind1)-x1p)*(1+abs(x1p-xFEtmp(xind1-1))/dx);
Iu_uMFqMFadj(xind1-1) = 0.5*abs(xFEtmp(xind1)-x1p)*(abs(xFEtmp(xind1)-x1p)/dx);
Iu_uMFqMFadj(xind2) = 0.5*dx + ...
    0.5*abs(x2p-xFEtmp(xind2))*(1+abs(xFEtmp(xind2+1)-x2p)/dx);
Iu_uMFqMFadj(xind2+1) = 0.5*abs(x2p-xFEtmp(xind2))*(abs(x2p-xFEtmp(xind2))/dx);
else
Iu_uMFqMFadj = zeros(1,nElemadj+1);
for ii = 1:nElemadj
	x1 = xenr(ii); x2 = xenr(ii+1);
	if (x1 >= x1p && x2 <= x2p) || (x1 <= x1p && x2 >= x1p) || (x1 <= x2p && x2 >= x2p)
		a = x1; b = x2;
		x1 = max(x1,x1p); x2 = min(x2,x2p);
		bmao2 = 0.5*(x2-x1); bpao2 = 0.5*(x2+x1);
		nitil = bmao2*ni + bpao2;
		uatn = interp1([xenr(ii) xenr(ii+1)],[uMFproj(ii) uMFproj(ii+1)],nitil);
		Iu_uMFqMFadj(ii) = Iu_uMFqMFadj(ii) + ...
			bmao2*wi(1)*uatn(1)*(1-(nitil(1)-a)/(b-a)) + ...
			bmao2*wi(2)*uatn(2)*(1-(nitil(2)-a)/(b-a));
		Iu_uMFqMFadj(ii+1) = Iu_uMFqMFadj(ii+1) + ...
			bmao2*wi(1)*uatn(1)*((nitil(1)-a)/(b-a)) + ...
			bmao2*wi(2)*uatn(2)*((nitil(2)-a)/(b-a));
	end
end
Iu_uMFqMFadj = 2*Iu_uMFqMFadj(2:end-1);
end

gamma = zeros(2*np+4*n,1);
gamma(np+1:np+n) = Cenr'*yd;
gamma(np+n+1:np+n+n) = -BCcorr;
gamma(2*(np+n)+1:2*(np+n)+n) = -Iu_uMFqMFadj';

XMproj = [qMF; uMFproj(2:end-1); zMFproj(2:end-1); ...
    pMF; vMFproj(2:end-1); yMFproj(2:end-1)];
MprimeXM = Asuper*XMproj-gamma;
MprimeXM = [MprimeXM(np+n+n+1:end); MprimeXM(1:np+n+n)];

superadj = Asuper'\(0.5*MprimeXM);
rho = Asuper*XMproj - gamma;

MHxHmNHxM = -dot(superadj,rho);
% MHxHmNHxMexp = -superadj.*rho;

%%% cell-by-cell error breakdown %%%

superadju = [0; superadj(np+1:np+n); 0];
superadjz = [0; superadj(np+n+1:np+n+n); 0];
superadjv = [0; superadj(end-n-n+1:end-n); 0];
superadjy = [0; superadj(end-n+1:end); 0];

%%%rho_u(xi_MF)(z-bit-of-superadj)

term1 = 0; %zero at boundaries

sadjzx = (superadjz(2:end)-superadjz(1:end-1))./(xenr(2:end)'-xenr(1:end-1)');
term2 = -dot(sadjzx.*uMFprojx,(xenr(2:end)'-xenr(1:end-1)'));
term2exp = -sadjzx.*uMFprojx.*(xenr(2:end)'-xenr(1:end-1)');

bits = zeros(nElemadj,5);
for ii = 1:nElemadj
    xim1 = xenr(ii);
    xi = xenr(ii+1);
    
    %from ymyh(x_i)
    al = 1/(xi-xim1);
    bl = -xim1/(xi-xim1);
    x1 = xim1; x2 = xi;
    bits(ii,1) = 0.5*(x2-x1); %phi(x) = 1;
    bits(ii,2) = (al*sin(2*pi*x2)-2*pi*(al*x2+bl)*cos(2*pi*x2) ...
        -(al*sin(2*pi*x1)-2*pi*(al*x1+bl)*cos(2*pi*x1))) ...
        /(4*pi^2); %phi(x) = sin(2*pi*x)
    bits(ii,3) = (al*cos(2*pi*x2)+2*pi*(al*x2+bl)*sin(2*pi*x2) ...
        -(al*cos(2*pi*x1)+2*pi*(al*x1+bl)*sin(2*pi*x1))) ...
        /(4*pi^2); %phi(x) = cos(2*pi*x)
    bits(ii,4) = (al*sin(4*pi*x2)-4*pi*(al*x2+bl)*cos(4*pi*x2) ...
        -(al*sin(4*pi*x1)-4*pi*(al*x1+bl)*cos(4*pi*x1))) ...
        /(16*pi^2); %phi(x) = sin(4*pi*x)
    bits(ii,5) = (al*cos(4*pi*x2)+4*pi*(al*x2+bl)*sin(4*pi*x2) ...
        -(al*cos(4*pi*x1)+4*pi*(al*x1+bl)*sin(4*pi*x1))) ...
        /(16*pi^2); %phi(x) = cos(4*pi*x)
    bits(ii,:) = bits(ii,:)*superadjz(ii+1);
    
    %from ymyh(x_{i-1})
    ar = -1/(xi-xim1);
    br = xi/(xi-xim1);
    x1 = xim1; x2 = xi;
    bitstmp = zeros(1,5);
    bitstmp(1) = 0.5*(x2-x1); %phi(x) = 1;
    bitstmp(2) = (ar*sin(2*pi*x2)-2*pi*(ar*x2+br)*cos(2*pi*x2) ...
        -(ar*sin(2*pi*x1)-2*pi*(ar*x1+br)*cos(2*pi*x1))) ...
        /(4*pi^2); %phi(x) = sin(2*pi*x)
    bitstmp(3) = (ar*cos(2*pi*x2)+2*pi*(ar*x2+br)*sin(2*pi*x2) ...
        -(ar*cos(2*pi*x1)+2*pi*(ar*x1+br)*sin(2*pi*x1))) ...
        /(4*pi^2); %phi(x) = cos(2*pi*x)
    bitstmp(4) = (ar*sin(4*pi*x2)-4*pi*(ar*x2+br)*cos(4*pi*x2) ...
        -(ar*sin(4*pi*x1)-4*pi*(ar*x1+br)*cos(4*pi*x1))) ...
        /(16*pi^2); %phi(x) = sin(4*pi*x)
    bitstmp(5) = (ar*cos(4*pi*x2)+4*pi*(ar*x2+br)*sin(4*pi*x2) ...
        -(ar*cos(4*pi*x1)+4*pi*(ar*x1+br)*sin(4*pi*x1))) ...
        /(16*pi^2); %phi(x) = cos(4*pi*x)
    bitstmp = bitstmp*superadjz(ii);
    bits(ii,:) = bits(ii,:) + bitstmp;
end
bits = bits*qMF;
term3 = sum(bits);

term4exp = zeros(nElemadj,1);
for ii = 1:nElemadj
    x1 = xenr(ii); x2 = xenr(ii+1);
    bmao2 = 0.5*(x2-x1); bpao2 = 0.5*(x2+x1);
    nitil = bmao2*ni + bpao2;
    term4exp(ii) = -uMFprojx(ii)*bmao2*dot(wi,...
        interp1([x1 x2],[Uenr(ii) Uenr(ii+1)],nitil).*...
        interp1([x1 x2],[superadjz(ii) superadjz(ii+1)],nitil));
end

term4 = sum(term4exp);

term5exp = zeros(nElemadj,1);
for ii = 1:nElemadj
    x1 = xenr(ii); x2 = xenr(ii+1);
    bmao2 = 0.5*(x2-x1); bpao2 = 0.5*(x2+x1);
    nitil = bmao2*ni + bpao2;
    term5exp(ii) = -k*bmao2*dot(wi,...
        interp1([x1 x2],[uMFproj(ii) uMFproj(ii+1)],nitil).*...
        interp1([x1 x2],[1-uMFproj(ii) 1-uMFproj(ii+1)],nitil).*...
        interp1([x1 x2],[superadjz(ii) superadjz(ii+1)],nitil));
end
term5 = sum(term5exp);

rhou_xiMF_sadjz = -(term1 + term2 + term3 + term4 + term5);
rhou_xiMF_sadjz_exp = -(term1 + term2exp + bits + term4exp + term5exp);

%%%rho_z(xi_MF)(u-bit-of-superadj)

term1 = -dot(Cenr*superadju(2:end-1),yd-Cenr*uMFproj(2:end-1));
term1exp = [Cenr'; 0 0]*((-Cenr*superadju(2:end-1)).*(yd-Cenr*uMFproj(2:end-1)));

sadjux = (superadju(2:end)-superadju(1:end-1))./(xenr(2:end)'-xenr(1:end-1)');

term2exp = zeros(nElemadj,1);
for ii = 1:nElemadj
    x1 = xenr(ii); x2 = xenr(ii+1);
    bmao2 = 0.5*(x2-x1); bpao2 = 0.5*(x2+x1);
    nitil = bmao2*ni + bpao2;
    term2exp(ii) = sadjux(ii)*bmao2*dot(wi,...
        interp1([x1 x2],[Uenr(ii) Uenr(ii+1)],nitil).*...
        interp1([x1 x2],[zMFproj(ii) zMFproj(ii+1)],nitil));
    term2exp(ii) = term2exp(ii) + k*bmao2*dot(wi,...
        interp1([x1 x2],[superadju(ii) superadju(ii+1)],nitil).*...
        interp1([x1 x2],[zMFproj(ii) zMFproj(ii+1)],nitil))...
        -2*k*bmao2*dot(wi,...
        interp1([x1 x2],[superadju(ii) superadju(ii+1)],nitil).*...
        interp1([x1 x2],[zMFproj(ii) zMFproj(ii+1)],nitil).*...
        interp1([x1 x2],[uMFproj(ii) uMFproj(ii+1)],nitil));
end
term2exp = term2exp + sadjux.*zMFprojx.*(xenr(2:end)'-xenr(1:end-1)');

term2 = sum(term2exp);

rhoz_xiMF_sadju = term1 + term2;
rhoz_xiMF_sadju_exp = term1exp + term2exp;

%%%rho_v(chi_MF)(y-bit-of-superadj)

bits = zeros(nElemadj,5);
for ii = 1:nElemadj
    xim1 = xenr(ii);
    xi = xenr(ii+1);
    
    %from ymyh(x_i)
    al = 1/(xi-xim1);
    bl = -xim1/(xi-xim1);
    x1 = xim1; x2 = xi;
    bits(ii,1) = 0.5*(x2-x1); %phi(x) = 1;
    bits(ii,2) = (al*sin(2*pi*x2)-2*pi*(al*x2+bl)*cos(2*pi*x2) ...
        -(al*sin(2*pi*x1)-2*pi*(al*x1+bl)*cos(2*pi*x1))) ...
        /(4*pi^2); %phi(x) = sin(2*pi*x)
    bits(ii,3) = (al*cos(2*pi*x2)+2*pi*(al*x2+bl)*sin(2*pi*x2) ...
        -(al*cos(2*pi*x1)+2*pi*(al*x1+bl)*sin(2*pi*x1))) ...
        /(4*pi^2); %phi(x) = cos(2*pi*x)
    bits(ii,4) = (al*sin(4*pi*x2)-4*pi*(al*x2+bl)*cos(4*pi*x2) ...
        -(al*sin(4*pi*x1)-4*pi*(al*x1+bl)*cos(4*pi*x1))) ...
        /(16*pi^2); %phi(x) = sin(4*pi*x)
    bits(ii,5) = (al*cos(4*pi*x2)+4*pi*(al*x2+bl)*sin(4*pi*x2) ...
        -(al*cos(4*pi*x1)+4*pi*(al*x1+bl)*sin(4*pi*x1))) ...
        /(16*pi^2); %phi(x) = cos(4*pi*x)
    bits(ii,:) = bits(ii,:)*superadjy(ii+1);
    
    %from ymyh(x_{i-1})
    ar = -1/(xi-xim1);
    br = xi/(xi-xim1);
    x1 = xim1; x2 = xi;
    bitstmp = zeros(1,5);
    bitstmp(1) = 0.5*(x2-x1); %phi(x) = 1;
    bitstmp(2) = (ar*sin(2*pi*x2)-2*pi*(ar*x2+br)*cos(2*pi*x2) ...
        -(ar*sin(2*pi*x1)-2*pi*(ar*x1+br)*cos(2*pi*x1))) ...
        /(4*pi^2); %phi(x) = sin(2*pi*x)
    bitstmp(3) = (ar*cos(2*pi*x2)+2*pi*(ar*x2+br)*sin(2*pi*x2) ...
        -(ar*cos(2*pi*x1)+2*pi*(ar*x1+br)*sin(2*pi*x1))) ...
        /(4*pi^2); %phi(x) = cos(2*pi*x)
    bitstmp(4) = (ar*sin(4*pi*x2)-4*pi*(ar*x2+br)*cos(4*pi*x2) ...
        -(ar*sin(4*pi*x1)-4*pi*(ar*x1+br)*cos(4*pi*x1))) ...
        /(16*pi^2); %phi(x) = sin(4*pi*x)
    bitstmp(5) = (ar*cos(4*pi*x2)+4*pi*(ar*x2+br)*sin(4*pi*x2) ...
        -(ar*cos(4*pi*x1)+4*pi*(ar*x1+br)*sin(4*pi*x1))) ...
        /(16*pi^2); %phi(x) = cos(4*pi*x)
    bitstmp = bitstmp*superadjy(ii);
    bits(ii,:) = bits(ii,:) + bitstmp;
end
bits = bits*pMF;
term1 = -sum(bits);

sadjyx = (superadjy(2:end)-superadjy(1:end-1))./(xenr(2:end)'-xenr(1:end-1)');

term2exp = zeros(nElemadj,1);
for ii = 1:nElemadj
    x1 = xenr(ii); x2 = xenr(ii+1);
    bmao2 = 0.5*(x2-x1); bpao2 = 0.5*(x2+x1);
    nitil = bmao2*ni + bpao2;
    term2exp(ii) = vMFprojx(ii)*bmao2*dot(wi,...
        interp1([x1 x2],[Uenr(ii) Uenr(ii+1)],nitil).*...
        interp1([x1 x2],[superadjy(ii) superadjy(ii+1)],nitil));
    term2exp(ii) = term2exp(ii) + k*bmao2*dot(wi,...
        interp1([x1 x2],[vMFproj(ii) vMFproj(ii+1)],nitil).*...
        interp1([x1 x2],[superadjy(ii) superadjy(ii+1)],nitil))...
        -2*k*bmao2*dot(wi,...
        interp1([x1 x2],[vMFproj(ii) vMFproj(ii+1)],nitil).*...
        interp1([x1 x2],[superadjy(ii) superadjy(ii+1)],nitil).*...
        interp1([x1 x2],[uMFproj(ii) uMFproj(ii+1)],nitil));
end
term2exp = term2exp + sadjyx.*vMFprojx.*(xenr(2:end)'-xenr(1:end-1)');

term2 = sum(term2exp);

rhov_chiMF_sadjy = term1 + term2;
rhov_chiMF_sadjy_exp = -bits + term2exp;

%%%rho_y(chi_MF)(v-bit-of-superadj)
term1exp = zeros(nElemadj,1);
for ii = 1:nElemadj
    x1 = xenr(ii); x2 = xenr(ii+1);
    if linPred
        if x1 >= x1p && x2 <= x2p
            term1exp(ii) = 0.5*(superadjv(ii)+superadjv(ii+1))*dxenr;
        elseif x1 < x1p && x2 > x1p
            left = interp1([xenr(ii) xenr(ii+1)],[superadjv(ii) superadjv(ii+1)],x1p);
            term1exp(ii) = 0.5*(left+superadjv(ii+1))*(xenr(ii+1)-x1p);
        elseif x1 < x2p && x2 > x2p
            right = interp1([xenr(ii) xenr(ii+1)],[superadjv(ii) superadjv(ii+1)],x2p);
            term1exp(ii) = 0.5*(right+superadjv(ii))*(x2p-xenr(ii));
        end
    else
        if (x1 >= x1p && x2 <= x2p) || (x1 <= x1p && x2 >= x1p) || (x1 <= x2p && x2 >= x2p)
            x1 = max(x1,x1p); x2 = min(x2,x2p);
            bmao2 = 0.5*(x2-x1); bpao2 = 0.5*(x2+x1);
            nitil = bmao2*ni + bpao2;
            term1exp(ii) = 2*bmao2*dot(wi,...
                interp1([xenr(ii) xenr(ii+1)],[superadjv(ii) superadjv(ii+1)],nitil).*...
                interp1([xenr(ii) xenr(ii+1)],[uMFproj(ii) uMFproj(ii+1)],nitil));
        end
    end
end
term1 = sum(term1exp);

term2 = dot(Cenr*vMFproj(2:end-1),Cenr*superadjv(2:end-1));
term2exp = [Cenr'; 0 0]*((Cenr*vMFproj(2:end-1)).*(Cenr*superadjv(2:end-1)));

sadjvx = (superadjv(2:end)-superadjv(1:end-1))./(xenr(2:end)'-xenr(1:end-1)');

term3exp = zeros(nElemadj,1);
for ii = 1:nElemadj
    x1 = xenr(ii); x2 = xenr(ii+1);
    bmao2 = 0.5*(x2-x1); bpao2 = 0.5*(x2+x1);
    nitil = bmao2*ni + bpao2;
    term3exp(ii) = sadjvx(ii)*bmao2*dot(wi,...
        interp1([x1 x2],[Uenr(ii) Uenr(ii+1)],nitil).*...
        interp1([x1 x2],[yMFproj(ii) yMFproj(ii+1)],nitil));
    term3exp(ii) = term3exp(ii) + k*bmao2*dot(wi,...
        interp1([x1 x2],[superadjv(ii) superadjv(ii+1)],nitil).*...
        interp1([x1 x2],[yMFproj(ii) yMFproj(ii+1)],nitil))...
        -2*k*bmao2*dot(wi,...
        interp1([x1 x2],[superadjv(ii) superadjv(ii+1)],nitil).*...
        interp1([x1 x2],[yMFproj(ii) yMFproj(ii+1)],nitil).*...
        interp1([x1 x2],[uMFproj(ii) uMFproj(ii+1)],nitil));
end
term3exp = term3exp + sadjvx.*yMFprojx.*(xenr(2:end)'-xenr(1:end-1)');

term3 = sum(term3exp);

term4exp = zeros(nElemadj,1);
for ii = 1:nElemadj
    x1 = xenr(ii); x2 = xenr(ii+1);
    bmao2 = 0.5*(x2-x1); bpao2 = 0.5*(x2+x1);
    nitil = bmao2*ni + bpao2;
    term4exp(ii) = -2*k*bmao2*dot(wi,...
        interp1([x1 x2],[superadjv(ii) superadjv(ii+1)],nitil).*...
        interp1([x1 x2],[vMFproj(ii) vMFproj(ii+1)],nitil).*...
        interp1([x1 x2],[zMFproj(ii) zMFproj(ii+1)],nitil));
end
term4 = sum(term4exp);

rhoy_chiMF_sadjv = term1 + term2 + term3 + term4;
rhoy_chiMF_sadjv_exp = term1exp + term2exp + term3exp + term4exp;

MHxHmNHxMexp = -(rhoy_chiMF_sadjv_exp + rhov_chiMF_sadjy_exp + ...
    rhoz_xiMF_sadju_exp + rhou_xiMF_sadjz_exp);

%%%%%%%% M_HF(x_MF) - M_MF(x_MF) %%%%%%%%
vMFx = (vMF(2:end)-vMF(1:end-1))./(xFE(2:end)'-xFE(1:end-1)');
uMFx = (uMF(2:end)-uMF(1:end-1))./(xFE(2:end)'-xFE(1:end-1)');
remainauexp = zeros(nElemLF,1); remainaexp = zeros(nElemLF,1);
UHFproj = interp1(xFEHF,UHF,xFE);
for ii = 1:nElemLF
    if ~upgradeMe(ii)
        x1 = xFE(ii); x2 = xFE(ii+1);
        bmao2 = 0.5*(x2-x1); bpao2 = 0.5*(x2+x1);
        nitil = bmao2*ni + bpao2;
%         remainauexp(ii) = k*bmao2*dot(wi,...
%             interp1([x1 x2],[vMF(ii) vMF(ii+1)],nitil).*...
%             interp1([x1 x2],[zMF(ii) zMF(ii+1)],nitil))...
%             -2*k*bmao2*dot(wi,...
%             interp1([x1 x2],[vMF(ii) vMF(ii+1)],nitil).*...
%             interp1([x1 x2],[zMF(ii) zMF(ii+1)],nitil).*...
%             interp1([x1 x2],[uMF(ii) uMF(ii+1)],nitil));
%         remainaexp(ii) = k*bmao2*dot(wi,...
%             interp1([x1 x2],[uMF(ii) uMF(ii+1)],nitil).*...
%             interp1([x1 x2],[1-uMF(ii) 1-uMF(ii+1)],nitil).*...
%             interp1([x1 x2],[yMF(ii) yMF(ii+1)],nitil));
        remainauexp(ii) = vMFx(ii)*bmao2*dot(wi,...
            interp1([x1 x2],[UHFproj(ii)-ULF(ii) UHFproj(ii+1)-ULF(ii+1)],nitil).*...
            interp1([x1 x2],[zMF(ii) zMF(ii+1)],nitil)) ...
            + k*bmao2*dot(wi,...
            interp1([x1 x2],[vMF(ii) vMF(ii+1)],nitil).*...
            interp1([x1 x2],[zMF(ii) zMF(ii+1)],nitil))...
            -2*k*bmao2*dot(wi,...
            interp1([x1 x2],[vMF(ii) vMF(ii+1)],nitil).*...
            interp1([x1 x2],[zMF(ii) zMF(ii+1)],nitil).*...
            interp1([x1 x2],[uMF(ii) uMF(ii+1)],nitil));
        remainaexp(ii) = uMFx(ii)*bmao2*dot(wi,...
            interp1([x1 x2],[UHFproj(ii)-ULF(ii) UHFproj(ii+1)-ULF(ii+1)],nitil).*...
            interp1([x1 x2],[yMF(ii) yMF(ii+1)],nitil)) ...
            + k*bmao2*dot(wi,...
            interp1([x1 x2],[uMF(ii) uMF(ii+1)],nitil).*...
            interp1([x1 x2],[1-uMF(ii) 1-uMF(ii+1)],nitil).*...
            interp1([x1 x2],[yMF(ii) yMF(ii+1)],nitil));
    end
end

remainau = sum(remainauexp);
remaina = sum(remainaexp);

%%%%%%%% the reckoning %%%%%%%%
ypdiffest = MHxHmNHxM + remainau + remaina
ypdiffestexp = sum(reshape(MHxHmNHxMexp,1,nElemLF)',2) + remainauexp + remainaexp;

ypdiffs_est(jj) = ypdiffest;

%actual prediction error
if linPred
    ypMF = Iu_uLFqLF*uMF(2:end-1);
else
    ypMF = 0;
    for ii = 1:nElemLF
        x1 = xFE(ii); x2 = xFE(ii+1);
        if (x1 >= x1p && x2 <= x2p) || (x1 <= x1p && x2 >= x1p) || (x1 <= x2p && x2 >= x2p)
            x1 = max(x1,x1p); x2 = min(x2,x2p);
            bmao2 = 0.5*(x2-x1); bpao2 = 0.5*(x2+x1);
            nitil = bmao2*ni + bpao2;
            ypMF = ypMF + bmao2*dot(wi,...
                interp1([xFE(ii) xFE(ii+1)],[uMF(ii) uMF(ii+1)],nitil).^2);
        end
    end
end
ypdiffs(jj) = ypHF - ypMF;

pcntConverted(jj) = mean(upgradeMe);

figure(1) %error indicator
midpts = 0.5*(xFE(1:end-1)+xFE(2:end));
hold on; plot(midpts,abs(ypdiffestexp),markers{jj},midpts(upgradeMe==1),abs(ypdiffestexp(upgradeMe==1)),markersUp{jj})

% figure(2)
% hold on; plot(xFE,zMF,markers{jj})
% figure(3)
% hold on; plot(xFE,uMF,markers{jj})
% figure(4)
% hold on; plot(xFE,yMF,markers{jj})
% figure(5)
% hold on; plot(xFE,vMF,markers{jj})
figure(6)
tootsieCenter = 0.5*(xFE(1:end-1)+xFE(2:end));
subplot(size(mixints,1),1,jj); 
plot(tootsieCenter(upgradeMe==1),ones(1,sum(upgradeMe)),'b*',tootsieCenter(upgradeMe==0),ones(1,sum(~upgradeMe)),'ro')
ylabel([num2str(100*mean(upgradeMe)),'% HF'])


end

labels = reshape(repmat(pcntConverted,2,1),2*length(pcntConverted),1);
figure(1); legend(cellstr(num2str(labels(2:end)))); xlabel('x'); ylabel('cell contribution to QoI error')
% figure(2); legend(cellstr(num2str(pcntConverted(:)))); xlabel('x'); ylabel('zMF');
% figure(3); legend(cellstr(num2str(pcntConverted(:)))); xlabel('x'); ylabel('uMF');
% figure(4); legend(cellstr(num2str(pcntConverted(:)))); xlabel('x'); ylabel('yMF');
% figure(5); legend(cellstr(num2str(pcntConverted(:)))); xlabel('x'); ylabel('vMF');

figure; 
% subplot(1,3,1); 
% plot(pcntConverted,abs(ypdiffs_est)./abs(ypHF),'-*')
% xlabel('% HF')
% ylabel('estimated absolute relative error in prediction')
% subplot(1,3,2); 
% plot(pcntConverted,abs(ypdiffs)./abs(ypHF),'-*')
% xlabel('% HF')
% ylabel('absolute relative error in prediction')
% subplot(1,3,3); 
plot(pcntConverted,abs(ypdiffs)./abs(ypHF),'-*',...
    pcntConverted,abs(ypdiffs_est)./abs(ypHF),'-*')
xlabel('% HF')
ylabel('absolute relative error in prediction')
legend('True','Estimated')


figure; plot(xFE,uStash');
legend(cellstr(num2str(pcntConverted(:))))
xlabel('x')
ylabel('inferred state')
