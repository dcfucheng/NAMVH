function [label, C, dis] = bin_kmeans(X, k, options)
%binary kmeans clustering modified on fkmeans
%by Kun Ding 2016-4-29
%X is +1/-1 binary matrix, each column is a sample

n = size(X,2);
B = compactbit_mex(X>0);

% option defaults
careful = 1;% random initialization
max_iter = 30;

if(nargin==3)
    if isfield(options,'careful')
        careful = options.careful;
    end
    if isfield(options,'max_iter')
        max_iter = options.max_iter;
    end
end

if careful
    idx = spreadseeds(B, k);
else
    idx = randsample(n,k);
end
C = X(:,idx);
Cb = B(:,idx);

% generate initial labeling of points
label = label_assign(Cb, B);

last = 0;
iter = 1;
while any(label ~= last)
    % remove empty clusters
    [~,~,label] = unique(label);
    
    % transform label into indicator matrix
    ind = sparse(1:n,label,1,n,k,n);
    
    % compute centroid of each cluster
    tmp = X*(ind*spdiags(1./sum(ind,1)',0,k,k));
    if(iter==1)
        C = sign(tmp);
    else
        C = sign_(tmp,C);
    end
    Cb = compactbit_mex(C>0);
    
    last = label';
    label = label_assign(Cb, B);
    
    iter = iter+1;
    
    if(iter>max_iter)
        break;
    end
end

if(nargout==3)
    dis = compute_energy(label, Cb, B);
end

end

function dis = compute_energy(label, Cb, B)

k = size(Cb,2);
n = size(B,2);
if(k*n*8/1024/1024/1024>2)%deal with at most 2GB memory
    bs = 5000;
    nb = ceil(n/bs);
    dis = 0;
    for i = 1:nb
        id = (i-1)*bs+1:min(n,i*bs);
        distances = hammDist_mex(Cb,B(:,id));
        ind = sub2ind([k,length(id)],label(:,id),1:length(id));
        dis = dis + sum(distances(ind));
    end
else
    ind = sub2ind([k,n],label,1:n);
    distances = hammDist_mex(Cb,B);
    dis = sum(distances(ind));
end

end

function label = label_assign(Cb, B)

k = size(Cb,2);
n = size(B,2);
if(k*n*8/1024/1024/1024>2)%deal with at most 2GB memory
    bs = 5000;
    nb = ceil(n/bs);
    label = zeros(1,n);
    for i = 1:nb
        ind = (i-1)*bs+1:min(n,i*bs);
        distances = hammDist_mex(Cb,B(:,ind));
        [~,label(ind)] = min(distances);
    end
else
    distances = hammDist_mex(Cb,B);
    [~,label] = min(distances);
end

end

function X = sign_(X, X1)

if(nargin<2)
    X = sign(X);
else
    pos = X~=0;
    X(pos) = sign(X(pos));
    pos = ~pos;
    X(pos) = X1(pos);
end

end

function idx = spreadseeds(B, k)
% X: n x d data matrix
% k: number of seeds
% reference: k-means++: the advantages of careful seeding.
% by David Arthur and Sergei Vassilvitskii
% Adapted from softseeds written by Mo Chen (mochen@ie.cuhk.edu.hk), 
% March 2009.

[d,n] = size(B);
idx = zeros(k,1);
S = zeros(d,k,'uint8');
D = inf(1,n);
idx(1) = ceil(n.*rand);
S(:,1) = B(:,idx(1));
for i = 2:k
    D = min(D,double(hammDist_mex(S(:,i-1),B)));
    idx(i) = find(cumsum(D)/sum(D)>rand,1);
    S(:,i) = B(:,idx(i));
end
end