function integrand = state_integrand(tau,H,sigma2,n,Q,v,lambda,x1,x2,y1,y2,x,y)

vx = v(1);
vy = v(2);
sigma2x = sigma2(1);
sigma2y = sigma2(2);

nevalpts = length(x(:));

ex = erf((repmat(x(:),1,length(tau))-x1-repmat(vx*tau,nevalpts,1))./repmat(sqrt(2*sigma2x*tau.^H),nevalpts,1))...
    -erf((repmat(x(:),1,length(tau))-x2-repmat(vx*tau,nevalpts,1))./repmat(sqrt(2*sigma2x*tau.^H),nevalpts,1));
ey = erf((repmat(y(:),1,length(tau))-y1-repmat(vy*tau,nevalpts,1))./repmat(sqrt(2*sigma2y*tau.^H),nevalpts,1))...
    -erf((repmat(y(:),1,length(tau))-y2-repmat(vy*tau,nevalpts,1))./repmat(sqrt(2*sigma2y*tau.^H),nevalpts,1));
r = repmat(exp(-lambda*tau),nevalpts,1);
integrand = (Q/(4*n))*r.*ex.*ey;

end

