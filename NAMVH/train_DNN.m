function train_DNN(models_path, opts, gpu_id)
%train DNN by caffe
%
%inputs:
% save_path: path to save models
% data_name: name of this dataset
% models_path: prototxt

caffe.set_device(gpu_id);
caffe.set_mode_gpu();

solver_file = sprintf('%s/%s/DC_%s_solver_%s_%d.prototxt', models_path, opts.data_name, opts.data_name, opts.net_name, opts.K);    

solver = caffe.Solver(solver_file);

%solver.net.copy_from(weights);

solver.solve();

%save caffe model
train_net = solver.net;
model_name = sprintf('%s/DC_%s_%s_%d.caffemodel', opts.save_path, opts.data_name, opts.net_name, opts.K);
train_net.save(model_name);

caffe.reset_all();
end