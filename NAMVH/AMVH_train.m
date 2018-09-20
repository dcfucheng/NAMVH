%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function model = AMVH_train(X, Y, opts)
%
% Main training code for "AMVH:Asymmetric Multi-Valued Hashing", CVPR 2017
%
%optimize objective function:
% |X'*W*C*A-S|_F^2+lambda*|V'*C*A-Y|_F^2
% where C is +1/-1 binary matrix, each column of A has opts.nnz 1s
%
%inputs:
% X: data matrix, d*N
% Y: 0-1 label matrix, C*N
%
% opts: settings
%  K: number of bits
%  lambda: trade-off parameter between preserving pairwise and pointwise
%          label
%  use_kernel: 1 use kernel feature mapping
%  Anchor: randomly selected anchors for kernel feature mapping
%  sample_size: number of anchors
%  sigma1_scl: scaling factor of kernel width
%  n_cluster: size of binary dictionary
%  max_iter: maximal iterations
%  A_nnz: number of 1s for each column of A
%
%output:
% model: a structure containing the needed information used for test
%  time: training and testing time (s)
%  W: projection matrix
%  C: +1/-1 binary dictionary
%  A: 0-1 sparse representation
%  B: C*A
%  Xhat: W'*X
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic;

global EPS;
EPS = 1e-4;
rng(opts.s);

N = size(X,2);
K = opts.K;
n_cluster = opts.n_cluster;
assert(n_cluster<N);

%default parameters
lambda = 1;
u_tansfer = 0.5;
use_kernel = 1;
sigma1_scl = 1;
A_nnz = 1;
with_anchor = 1;

if(isfield(opts,'lambda'))
    lambda = opts.lambda*N/size(Y,1);
end

if(isfield(opts,'A_nnz'))
    A_nnz = opts.A_nnz;
    assert(A_nnz>0);
    assert(round(A_nnz)==A_nnz);
end

if(isfield(opts,'use_kernel'))
    use_kernel = opts.use_kernel;
end

if(isfield(opts,'sigma1_scl'))
    sigma1_scl = opts.sigma1_scl;
end

if(isfield(opts,'u_tansfer'))
    u_tansfer = opts.u_tansfer;
end

if(isfield(opts,'sample_size'))
    n_anchor = opts.sample_size;
end

if(isfield(opts,'with_anchor'))
    with_anchor = opts.with_anchor;
end

%label transformation
Y = sparse(Y);
[P,Q] = decom_similarity(Y, u_tansfer);


%feature mapping
if use_kernel
    if(isfield(opts,'Anchor'))
        Anchor = opts.Anchor;
    else
        if with_anchor
            ind_anchor = opts.ind_anchor;
            Anchor = X(:,ind_anchor);
        else
            Anchor = X(:,randsample(size(X,2), n_anchor));
        end
    end
    D2 = Euclid2(Anchor, Anchor, 'col', 0);
    squared_sigma = mean(D2(:))*sigma1_scl;
    D2 = Euclid2(Anchor, X, 'col', 0);
    X = exp(-D2/2/squared_sigma);
    model.anchor = Anchor;
    model.squared_sigma = squared_sigma;
    clear ind D2;
end


model.use_kernel = use_kernel;

%remove data mean
mean_vec = mean(X,2);
X = bsxfun(@minus, X, mean_vec);
model.mean_vec = mean_vec;

%normalize standard variance
std_vec = std(X,0,2);
X = bsxfun(@rdivide, X, std_vec+eps);
model.std_vec = std_vec;


%initialization
[A,B,C] = init_ABC(X, n_cluster, K, 'random');
%[A,B,C] = init_ABC(X, n_cluster, K, 'kmeans');
A0 = A;
B0 = B;
C0 = C;

W_left = ((X*X'+EPS*eye(size(X,1)))\X)*P'*Q;
V = (B*B'+EPS*eye(K))\(B*Y');
W_1 = inf(size(X,1),K);

obj = inf;
objs = [];


for iter = 1:opts.max_iter
    
    %W step
    W = W_step(W_left, C*A);
    
    %A step
    A = A_step(X, Y, P, Q, W, V, C, lambda, A_nnz);
    
    %C step
    Xhat = W'*X;
    C = C_step(Xhat, Y, P, Q, A, V, C, lambda, A_nnz);
    
    %V step
    B = C*A;
    V = (B*B'+EPS*eye(K))\(B*Y');
    
    %determine if the convergence is reached
    if(N<50000)%use the objective value for small problem N<50000
        obj_1 = obj;
        obj = comp_obj(Xhat, B, P, Q, V, Y, lambda);
        objs = [objs,obj];
        model.objs = objs;
        fprintf('iter=%d, obj=%f\n',iter,obj);
        
        if(iter>1&&abs(obj-obj_1)<1e-5)
            break;
        end
    else%use error betwwen W and W_1 for large N
        err = norm(W_1-W,2)/norm(W,2);
        W_1 = W;
        fprintf('iter=%d,err=%f\n',iter,err);
        
        if(iter>1&&err<1e-3)
            break;
        end
    end
    
end

[ind_nnz,~] = find(A);
ind_nnz = reshape(ind_nnz,A_nnz,[]);

model.time = toc;
model.W = W;
model.B = B;
model.C = C;
model.Ccb = compactbit_mex(C>0);
model.A = A;
model.V = V;
model.ind_nnz = ind_nnz;
model.A_nnz = A_nnz;
model.Xhat = Xhat;
model.P = P;
model.Q = Q;
model.A0 = A0;
model.B0 = B0;
model.C0 = C0;

if (use_kernel && with_anchor)
    model.ind_anchor = ind_anchor;
end
B_extend = zeros(A_nnz*K,N);
for i = 1:N
    B_extend(:,i) = reshape(C(:,ind_nnz(:,i)),[],1);
end
model.B_extend = compactbit_mex(B_extend>0);

end


function W = W_step(W_left, B)
global EPS;
[K,~] = size(B);
W = W_left*B'/(B*B'+EPS*eye(K));
end


function A = A_step(X, Y, P, Q, W, V, C, lambda, A_nnz)

n_cluster = size(C,2);
N = size(X,2);
if(A_nnz==1)%when A_nnz=1, both small and large problem can be solved
    if(n_cluster*N*8/1024/1024/1024<2)%small problem: at most 2GB
        R1 = X'*W*C;
        R2 = V'*C;
        U = -(R1'*R1 + lambda*R2'*R2)/2;
        F = R1'*P'*Q+lambda*R2'*Y;
        tmp = bsxfun(@plus, F, diag(U));
        [~,id_max] = max(tmp,[],1);
        A = sparse(id_max,repmat(1:N,A_nnz,1),ones(size(id_max)),n_cluster,N);
    else%big problem
        u = zeros(1,n_cluster);
        bs = 1000;
        nb = ceil(n_cluster/bs);
        for i = 1:nb
            ind = (i-1)*bs+1:min(n_cluster,i*bs);
            u(ind) = -(sum((X'*W*C(:,ind)).^2)+lambda*sum((V'*C(:,ind)).^2))/2;
        end
        u = u';
        
        bs = 5000;
        nb = ceil(N/bs);
        id_max = zeros(A_nnz,N);
        PR1 = P*(X'*W)*C;
        R2 = V'*C;
        for i = 1:nb
            ind = (i-1)*bs+1:min(N,i*bs);
            F = PR1'*Q(:,ind)+lambda*R2'*Y(:,ind);
            tmp = bsxfun(@plus, F, u);
            [~,a] = max(tmp,[],1);
            id_max(:,ind) = a;
        end
        A = sparse(id_max,repmat(1:N,A_nnz,1),ones(size(id_max)),n_cluster,N);
    end
else%for A_nnz>1, only small problem can be dealt
    if(n_cluster>10000)
        error('As there is not structure imposed on A, we can only solve small problem currently!');
    end
    R1 = X'*W*C;
    R2 = V'*C;
    U = R1'*R1 + lambda*R2'*R2;
    F = R1'*P'*Q + lambda*R2'*Y;
    id_max = zeros(A_nnz,N);
    U = -U/2;%convert to maximization
    id_max = solve_A_greedy_mex(U, F, A_nnz);
    id_max = double(id_max);
    A = sparse(id_max,repmat(1:N,A_nnz,1),ones(size(id_max)),n_cluster,N);
end
end

function C = C_step(Xhat, Y, P, Q, A, V, C, lambda, A_nnz)

global EPS;

K = size(C,1);
n_cluster = size(C,2);

R = (Xhat*P'*Q+lambda*V*Y)*A';
U = A*A';

invU = (U+EPS*eye(n_cluster))\eye(n_cluster);

for tt = 1:10
    C_1 = C;
    for k = 1:K
        sel = true(K,1);
        sel(k) = false;
        f = R(k,:) - Xhat(k,:)*Xhat(sel,:)'*C(sel,:)*U - lambda*V(k,:)*V(sel,:)'*C(sel,:)*U;
        if(A_nnz==1)
            C(k,:) = sign_(f,C(k,:));
        else
            C(k,:) = sign_(f*invU,C(k,:));
        end
    end
    err = nnz(C~=C_1);
    if(err<1e-10)
        break;
    end
end

end


function [A,B,C] = init_ABC(X, n_cluster, K, method)

N = size(X,2);

%B
W = ITQtrain1(X', K);
B = sign(W'*X);

%C
if(strcmp(method,'kmeans'))
    
    n_repeat = 5;
    energies = zeros(1,n_repeat);
    id_maxs = cell(1,n_repeat);
    Cs = cell(1,n_repeat);
    for i = 1:n_repeat
        [id_maxs{i}, Cs{i}, energies(i)] = bin_kmeans(B, n_cluster); %binary kmeans clustering
    end
    [~,ind] = min(energies);
    id_max = id_maxs{ind};
    C = Cs{ind};
else
    ind = randsample(N,n_cluster);
    C = B(:,ind);
    id_max = randsample(n_cluster,N,'true');
end

%A
A = sparse(double(id_max),1:N,ones(1,N),n_cluster,N);

end

function obj = comp_obj(Xhat, B, P, Q, V, Y, lambda)

N = size(B,2);
bs = 1000;
nb = ceil(N/bs);
obj = 0;
for i = 1:nb
    ind = (i-1)*bs+1:min(i*bs,N);
    obj = obj + sum(sum((Xhat(:,ind)'*B(:,ind)-P(:,ind)'*Q(:,ind)).^2));
end
obj = obj/N/N + lambda*sum(sum((V'*B-Y).^2))/N/size(Y,1);
end

