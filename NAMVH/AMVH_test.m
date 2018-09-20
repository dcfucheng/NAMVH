%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [IX, time] = AMVH_test(X, model, use_sign, use_mex, splits)
%
% Main testing code for "AMVH:Asymmetric Multi-Valued Hashing", CVPR 2017 
%
%inputs:
% X: test data matrix, d*N
% model: the learned shallow model
%
% model: settings
%  C: dictionary
%  B: the learned codes
%  B_extend: the concatenation of the selected atoms of C
%  A_nnz: number of 1s for each column of A
%  ind_nnz: the index of 1 in A 
%
% use_sign = 1 use binary codes for query 
% use_sign = 0 use binary codes for query 
%
% use_mex = 1 use lookup table to calculate distance
% use_mex = 0 use matrix multiplication to calculate distance
%
% splits = 10 transfer the big problem into some small problems  
%
%output:
% IX: sort matrix, number_uery * number_database
% time: testing time
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic;

%feature mapping
if model.use_kernel
    D2 = Euclid2(model.anchor, X, 'col', 0);
    X = exp(-D2/2/model.squared_sigma);
end

%normalize standard variance
X = bsxfun(@minus, X, model.mean_vec);
X = bsxfun(@rdivide, X, model.std_vec+eps);

A_nnz = model.A_nnz;
N_test = size(X,2);
N_retri = size(model.B,2);
% K = size(model.C,1);

%sorting
if(use_sign)%use binary codes
    Btest = compactbit_mex(model.W'*X>0);
%     clear X
    if(A_nnz==1)
        Bretri = compactbit_mex(model.B>0);
        D = hammDist_mex(Btest,Bretri);

%         D = zeros(N_test,N_retri,'uint16');
%         for i = 1:N_test
%             D(i,:) = dist(i,model.ind_nnz);
%         end
%         time = toc;

        %D = disttest_omp2(dist,model.ind_nnz,N_test,N_retri);        
    else
        D = hammDist_mex(repmat(Btest,A_nnz,1),model.B_extend);
        
%         Xhat = sign(model.W'*X);
%         sim = Xhat'*model.C;
%         S = simtest_omp(sim,model.ind_nnz,N_test,N_retri);
%         clear Xs
%         time = toc;
%         if (N_retri>50000) %%big problem
%             num_one_split = N_test/splits;
%             IX =[];
%             for sp = 1:splits
%                 temp_S = S(num_one_split*(sp-1)+1:num_one_split*sp,:);
%                 [~,tempIX] = sort(temp_S,2,'descend');
%                 IX = [IX;tempIX];
%             end
%             clear S
%         else
%             [~,IX] = sort(S,2,'descend');
%             clear S
%         end
%         
        
    end
    
    
    time = toc;
    if (N_retri>500000) %%big problem
        num_one_split = ceil(N_test/splits);
        IX =[];
        for sp = 1:splits
            temp_D = D(num_one_split*(sp-1)+1:min(num_one_split*sp,N_test),:);
            [~,tempIX] = sort(temp_D,2,'ascend');
            IX = [IX;tempIX];
			fprintf('Test sort %d splits.\n', sp);
        end
        clear D
    else
        [~,IX] = sort(D,2,'ascend');
        clear D
    end

    
else
    if(use_mex)
        Xhat = model.W'*X;
        sim = Xhat'*model.C;
        S = simtest_omp(sim,model.ind_nnz,N_test,N_retri);
        clear X
        time = toc;
        if (N_retri>500000) %%big problem
            num_one_split = ceil(N_test/splits);
            IX =[];
            for sp = 1:splits
                temp_S = S(num_one_split*(sp-1)+1:min(num_one_split*sp,N_test),:);
                [~,tempIX] = sort(temp_S,2,'descend');
                IX = [IX;tempIX];
				fprintf('Test sort %d splits.\n', sp);
            end
            clear S
        else
            [~,IX] = sort(S,2,'descend');
            clear S
        end
    else
        Xhat = model.W'*X;        
        S = Xhat'*model.B;
        clear X        
        time = toc;
        if (N_retri>500000) %%big problem
            num_one_split = ceil(N_test/splits);
            IX =[];
            for sp = 1:splits
                temp_S = S(num_one_split*(sp-1)+1:min(num_one_split*sp,N_test),:);
                [~,tempIX] = sort(temp_S,2,'descend');
                IX = [IX;tempIX];
				fprintf('Test sort %d splits.\n', sp);
            end
            clear S
        else
            [~,IX] = sort(S,2,'descend');
            clear S
        end
    end
    
end

end