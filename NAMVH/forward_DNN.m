function F = forward_DNN(ImgData, model_path, opts, gpu_id) 
%
%forward_DNN
%
%inputs:
% deploy_path: path to deploy
% weights_path:path to weights

caffe.set_device(gpu_id);
caffe.set_mode_gpu();
    
    deploy_path = sprintf('%s/%s/DC_%s_deploy_%s_%d.prototxt', model_path, opts.data_name, opts.data_name ,opts.net_name, opts.K);
    weights = sprintf('%s/DC_%s_%s_%d.caffemodel', opts.save_path, opts.data_name, opts.net_name, opts.K);
    net = caffe.Net(deploy_path, weights,'test');
    num_trains = size(ImgData,4);
    bs = 1000;
    
    net.blobs('data').reshape([1, 1, size(ImgData,3), bs]);
    net.reshape();

    F = cell(1,ceil(num_trains/bs));
    tic;
    for i = 1:ceil(num_trains/bs)
        if i ~= ceil(num_trains/bs)
            img = ImgData(:,:,:,(i-1)*bs+1:i*bs);
            tmp = net.forward({img});  
        else
            img = ImgData(:,:,:,(i-1)*bs+1:end);
            net.blobs('data').reshape([1, 1, size(ImgData,3), size(ImgData,4)-(i-1)*bs]);
            tmp = net.forward({img});
        end
        F{i} = tmp{1};
    end
    F = double(cell2mat(F));
    toc
    
    caffe.reset_all();
end