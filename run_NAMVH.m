%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the code for paper 'Nonlinear Asymmetric Multi-Valued Hashing, TPAMI2018'
% Written by Cheng Da (cheng.da@nlpr.ia.ac.cn)
% Last modified: 2018-09-18
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

diary NAMVH_log.txt

clc;
close all;

addpath(genpath('./utilities/'));
addpath(genpath('./NAMVH/'));
addpath(genpath('./caffe_cudnn/matlab'));

opts.data_name = 'ESP_GAME'; 
opts.net_name = 'DNN';

load_path = sprintf('./data/%s.mat', opts.data_name);
load(load_path)

data = get_data(data, data.splits{1});
S = data.Ytest'*data.Yretri>0;

n_anchor = 1000;
nbits = [8]; %%% 8,16,32,64
topK_MAP = 2000;
topKs = 1:50:500;
top_NDCG = 100;
ann = [1,10]; 
maps = zeros(1,2);
ndcg = zeros(1,2);
Pres = cell(1,2);
Tr_times = zeros(1,2);

warning('off'); 
gpu_id = 0;

for i = 1:length(ann)

fprintf('NAMVH_DNN on %s:\n',opts.data_name);
opts.A_nnz = ann(i);
opts.K = nbits;
opts.lambda = 10;
opts.lambda2 = 10;
opts.max_iter = 10;
opts.n_cluster = 256;
load s;
opts.s = s;

% linux
opts.data_path = './data/';
opts.models_path = './results/solvers'; 
opts.save_path = './results/models'; 

model = NAMVH_train(data.Xretri, data.Yretri, opts, gpu_id);
Tr_times(i) = model.time;

[IX1, test_time1] = NAMVH_test(data.Xtest, model, opts, 1, 1, 10, gpu_id);
[IX0, test_time0] = NAMVH_test(data.Xtest, model, opts, 0, 0, 10, gpu_id);


maps(i,1) = fastMAP(S, IX1', topK_MAP);
maps(i,2) = fastMAP(S, IX0', topK_MAP);
Pres{i,1} = topK_Pre(S, IX1', topKs);
Pres{i,2} = topK_Pre(S, IX0', topKs);
ndcg(i,1) = NDCG_k(data.Yretri,data.Ytest,IX1',top_NDCG);
ndcg(i,2) = NDCG_k(data.Yretri,data.Ytest,IX0',top_NDCG);


caffe.reset_all();
opts

end
%clear data IX0 IX1 S;

save_path = sprintf('result_NAMVH_%s.mat', opts.data_name);
save(save_path, 'maps','-v7.3');

maps

diary off



