%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [IX, time] = NAMVH_test(X, model, opts, use_sign, use_mex, splits, gpu_id)
%
% Main testing code for "Nonlinear Asymmetric Multi-Valued Hashing", TPAMI2018
%
%inputs:
% X: test data matrix, d*N
% model: the learned deep model
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

A_nnz = model.A_nnz;
N_test = size(X,2);
N_retri = size(model.B,2);

X_Img = reshape(full(X),1,1,size(X,1),size(X,2));
clear X

tic;

%sorting
if(use_sign)%use binary codes
	
    Xhat = forward_DNN(X_Img, opts.models_path, opts, gpu_id);
	fprintf('Test Forward DNN done.\n');
    Btest = compactbit_mex(Xhat>0);

    if(A_nnz==1)
        Bretri = compactbit_mex(model.B>0);
        D = hammDist_mex(Btest,Bretri);        
    else
        D = hammDist_mex(repmat(Btest,A_nnz,1),model.B_extend);
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
	fprintf('Test sort done.\n');
else
    if(use_mex)
        Xhat = forward_DNN(X_Img, opts.models_path, opts, gpu_id);
		fprintf('Test Forward DNN done.\n');
        sim = Xhat'*model.C;
        S = simtest_omp(sim,model.ind_nnz,N_test,N_retri);

        time = toc;
        if (N_retri>500000) %%big problem
            num_one_split = ceil(N_test/splits);
            IX =[];
            for sp = 1:splits
                temp_S = S(num_one_split*(sp-1)+1:min(num_one_split*sp,N_test),:);
                [~,tempIX] = sort(temp_S,2,'descend');
                IX = [IX;tempIX];
				fprintf('Test sort done %d splits.\n', sp);
            end
            clear S
        else
            [~,IX] = sort(S,2,'descend');
            clear S
        end
		fprintf('Test sort done.\n');
    else
        Xhat = forward_DNN(X_Img, opts.models_path, opts, gpu_id);
		fprintf('Test Forward DNN done.\n');		
        S = Xhat'*model.B; 
  
        time = toc;
        if (N_retri>500000) %%big problem
            num_one_split = ceil(N_test/splits);
            IX =[];
            for sp = 1:splits
                temp_S = S(num_one_split*(sp-1)+1:min(num_one_split*sp,N_test),:);
                [~,tempIX] = sort(temp_S,2,'descend');
                IX = [IX;tempIX];
				fprintf('Test sort done %d splits.\n', sp);
            end
            clear S
        else
            [~,IX] = sort(S,2,'descend');
            clear S
        end
		fprintf('Test sort done.\n');
    end
    
end

end