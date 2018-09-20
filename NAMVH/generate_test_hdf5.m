function generate_test_hdf5(ImgData, Z, data_path, data_name)
%generate hdf5 files
%
%inputs:
% data: 4-D matrix
% label: label K*N bits
% data_path: path to save data
% data_name: name of this dataset
chunksz=100;
hdf5_file_name = sprintf('%s_%s_test_.h5',data_path, data_name);
hdf5_txt_name = sprintf('%s_%s_testlist.txt',data_path, data_name);

mat2hdf5(ImgData, Z, chunksz, file_name,'single',5);

FILE=fopen(hdf5_txt_name, 'w');
fprintf(FILE, '%s', file_name);
fclose(FILE);
fprintf('HDF5 filename listed in %s \n', hdf5_txt_name);

end
