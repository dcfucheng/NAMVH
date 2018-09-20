%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function model = NAMVH_train(X, Y, opts, gpu_id)
%
% Main training code for "Nonlinear Asymmetric Multi-Valued Hashing", TPAMI2018
%
% optimize objective function:
% |Z'*C*A-S|_F^2+lambda*|V'*C*A-Y|_F^2+lambda2*|Z-F|_F^2
%
% where C is +1/-1 binary matrix, each column of A has opts.nnz 1s
%
%inputs:
% X: data matrix, d*N
% Y: 0-1 label matrix, C*N
%
%opts: settings
%  K: number of bits
%  lambda: trade-off parameter between preserving pairwise and pointwise label
%  lambda2: hyper-parameter 
%  n_cluster: the size of binary dictionary
%  max_iter: maximal iterations
%  A_nnz: number of 1s for each column of A
%  s: random seed
%
%output:
% model: a structure containing the needed information used for test
%  time: training and testing time (s)
%  the learned network model
%  C: +1/-1 binary dictionary
%  A: 0-1 sparse representation
%  B: C*A
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic;

global EPS;
EPS = 1e-4;
rng(opts.s);

N = size(Y,2);
K = opts.K;
n_cluster = opts.n_cluster;
assert(n_cluster<N);

%default parameters
lambda = 10;
lambda2 = 10;
A_nnz = 1;
u_tansfer = 0.5;
loss_w = 10;

if(isfield(opts,'lambda'))
    lambda = opts.lambda*N/size(Y,1);
end

if(isfield(opts,'lambda2'))
    lambda2 = opts.lambda2*N/K;
end

if(isfield(opts,'A_nnz'))
    A_nnz = opts.A_nnz;
    assert(A_nnz>0);
    assert(round(A_nnz)==A_nnz);
end

% similarity construction
Y = sparse(Y);
[P,Q] = decom_similarity(Y, u_tansfer);

%initialization
%[A,B,C] = init_ABC(X, n_cluster, K, 'random', A_nnz); 
[A,B,C] = init_ABC(X, n_cluster, K, 'kmeans'); 
A0 = A;
B0 = B;
C0 = C;

V = (B*B'+EPS*eye(K))\(B*Y');

Z = (B*B' + lambda2*eye(K) + EPS*eye(K))\(B*Q'*P);

Z_1 = Z;

obj = inf;
objs = [];

ZZ = cell(1,opts.max_iter);
FF = cell(1,opts.max_iter);

order_h5 = randperm(N);

X_img = reshape(full(X),1,1,size(X,1),size(X,2));

% main loop
for iter = 1:opts.max_iter
    
	%F step
    Z2 = Z*loss_w;
  	generate_train_hdf5(X_img(:,:,:,order_h5), Z2(:,order_h5), opts); 
	fprintf('HDF5 train files gre. %d itr done \n',iter);
    
    train_DNN(opts.models_path, opts, gpu_id);
    fprintf('Train DNN. %d iter done \n',iter);
    
	F2 = forward_DNN(X_img, opts.models_path, opts, gpu_id);
    F = F2/loss_w;
	FF{iter} = F;
	fprintf('Forward DNN. %d iter done \n',iter);

    %A step
    A = A_step(Z, Y, P, Q, V, C, lambda, A_nnz);
    fprintf('A_step, %d iter done \n',iter);
    
    %C step
    C = C_step(Z, Y, P, Q, A, V, C, lambda, A_nnz);
    fprintf('C_step, %d iter done \n',iter);
    
    %V step
    B = C*A;
    V = (B*B'+EPS*eye(K))\(B*Y');
    fprintf('V_step, %d iter done \n',iter);
    
	%Z step
	Z = Z_step(B, P, Q, F, lambda2);
	ZZ{iter} = Z;
	fprintf('Z_step, %d iter done \n',iter);
    
    %determine if the convergence is reached
    if(N<50000)%use the objective value for small problem
        obj_1 = obj;
        obj = comp_obj(Z, B, P, Q, V, Y, lambda);
        objs = [objs,obj];
        model.objs = objs;
        fprintf('iter=%d, obj=%f\n',iter,obj);
        
        if(iter>1&&abs(obj-obj_1)<1e-5)
            break;
        end
    else%use error betwwen W and W_1 for large N
        err = norm(Z_1-Z,2)/norm(Z,2);
        Z_1 = Z;
        fprintf('iter=%d,err=%f\n',iter,err);
        
        if(iter > 1 && err < 1e-4)
            break;
        end
    end
    
end

[ind_nnz,~] = find(A);
ind_nnz = reshape(ind_nnz,A_nnz,[]);

model.time = toc;
model.B = B;
model.C = C;
model.Ccb = compactbit_mex(C>0);
model.A = A;
model.V = V;
model.ind_nnz = ind_nnz;
model.A_nnz = A_nnz;
model.Z = Z;
model.F = F;
model.ZZ = ZZ;
model.FF = FF;
model.P = P;
model.Q = Q;
model.A0 = A0;
model.B0 = B0;
model.C0 = C0;


B_extend = zeros(A_nnz*K,N);
for i = 1:N
    B_extend(:,i) = reshape(C(:,ind_nnz(:,i)),[],1);
end
model.B_extend = compactbit_mex(B_extend>0);

end


%%%
%%% function 
%%%
function Z = Z_step(B, P, Q, F, lambda2)
global EPS;
K = size(B,1);

Z = (B*B' + lambda2*eye(K) + EPS*eye(K))\(B*Q'*P + lambda2*F);
end

function A = A_step(Z, Y, P, Q, V, C, lambda, A_nnz)

n_cluster = size(C,2);
N = size(Y,2);
if(A_nnz==1)%when A_nnz=1, both small and large problem can be solved
    if(n_cluster*N*8/1024/1024/1024<2)%small problem: at most 2GB
        R1 = Z'*C;
        R2 = V'*C;
        U = -(R1'*R1 + lambda * (R2'*R2))/2;
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
            u(ind) = -(sum((Z'*C(:,ind)).^2)+lambda*sum((V'*C(:,ind)).^2))/2;
        end
        u = u';
        
        bs = 5000;
        nb = ceil(N/bs);
        id_max = zeros(A_nnz,N);
        PR1 = P*(Z')*C;
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
    R1 = Z'*C;
    R2 = V'*C;
    U = R1'*R1 + lambda*(R2'*R2);
    F = R1'*P'*Q+lambda*R2'*Y;
    id_max = zeros(A_nnz,N);
    
    U = -U/2;%convert to maximization
    id_max = solve_A_greedy_mex(U, F, A_nnz);
    id_max = double(id_max);
    A = sparse(id_max,repmat(1:N,A_nnz,1),ones(size(id_max)),n_cluster,N);
end

end


function C = C_step(Z, Y, P, Q, A, V, C, lambda, A_nnz)

global EPS;

K = size(C,1);
n_cluster = size(C,2);

R = (Z*P'*Q+lambda*V*Y)*A';
U = A*A';

%%c_method,'inv')
invU = (U+EPS*eye(n_cluster))\eye(n_cluster);

for tt = 1:10
    C_1 = C;
    for k = 1:K
        sel = true(K,1);
        sel(k) = false;
        f = R(k,:) - Z(k,:)*Z(sel,:)'*C(sel,:)*U - lambda*V(k,:)*V(sel,:)'*C(sel,:)*U;
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

%B
N = size(X,2);
W = ITQtrain1(X', K);
B = sign(W'*X);

%C
if(strcmp(method,'kmeans'))
    n_repeat = 5;
    energies = zeros(1,n_repeat);
    id_maxs = cell(1,n_repeat);
    Cs = cell(1,n_repeat);
    for i = 1:n_repeat
        [id_maxs{i}, Cs{i}, energies(i)] = bin_kmeans(B, n_cluster);%binary kmeans clustering
    end
    [~,ind] = min(energies);
    id_max = id_maxs{ind};
    C = Cs{ind};
else
    ind = randsample(N,n_cluster);
    C = B(:,ind);
    id_max = randsample(n_cluster,N,true);
end

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
obj = obj + lambda*sum(sum((V'*B-Y).^2));

end

