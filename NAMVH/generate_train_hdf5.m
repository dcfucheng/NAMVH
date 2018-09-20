function generate_train_hdf5(ImgData, Z, opts)
%generate hdf5 files
%
%inputs:
% data: 4-D matrix
% label: label K*N bits
% data_path: path to save data
% data_name: name of this dataset

chunksz=1000;
hdf5_file_name = sprintf('%s/%s_train_%s_%d.h5',opts.data_path, opts.data_name, opts.net_name, opts.K);
hdf5_txt_name = sprintf('%s/%s_trainlist_%s_%d.txt',opts.data_path, opts.data_name, opts.net_name, opts.K);

mat2hdf5(ImgData, Z, chunksz, hdf5_file_name,'single',5);

FILE=fopen(hdf5_txt_name, 'w');
fprintf(FILE, '%s', hdf5_file_name);
fclose(FILE);
fprintf('HDF5 filename listed in %s \n', hdf5_txt_name);

end
