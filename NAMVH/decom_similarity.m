function [P,Q] = decom_similarty(Y, alpha0)
%each column is a sample

N = size(Y,2);
[alpha,beta] = estimate_scl(Y, alpha0);
% alpha = 2;beta = 1;

if(all(sum(Y,1)<=1))%only for multiclass data
    w = sum(Y,2);
    w = min(w)./w;
    Y = bsxfun(@times,Y,w);
end

P = [sqrt(alpha)*Y;sqrt(beta)*ones(1,N)];
Q = [sqrt(alpha)*Y;-sqrt(beta)*ones(1,N)];

end

function [alpha,beta] = estimate_scl(Y, alpha0)

N = size(Y,2);
ns = estimate_ns(Y);
nd = N^2-ns;
beta = alpha0*ns/nd;
alpha = (1+beta);

end

function ns = estimate_ns(Y)
%Y: each column is a label vector

N = size(Y,2);
if(all(sum(Y)==1))
    ns = sum(sum(Y,2).^2);
else
    if(N<10000)
        ns = nnz(Y'*Y);
    elseif(N<200000)
        Ysamp = Y(:,randsample(N,round(0.1*N)));
        ns = min(N^2,10^2*nnz(Ysamp'*Ysamp));
    else
        Ysamp = Y(:,randsample(N,round(0.02*N)));
        ns = min(N^2,50^2*nnz(Ysamp'*Ysamp));
    end
end

end