function mat2hdf5(data_disk,label_disk,chunksz,filename,data_type,compress_level)
num_total_samples=size(data_disk,4);
created_flag=false;
totalct=0;

if~exist('data_type','var')
    data_type = 'single';
end

if(~exist('compress_level','var'))
    compress_level = 0;
end

for batchno=1:ceil(num_total_samples/chunksz)
%     fprintf('batch no. %d\n', batchno);
    last_read=(batchno-1)*chunksz;
    
    % to simulate maximum data to be held in memory before dumping to hdf5 file
    batchdata=data_disk(:,:,:,last_read+1:min(last_read+chunksz,num_total_samples));
    
    %%%batchlabs=label_disk(:,:,:,last_read+1:min(last_read+chunksz,num_total_samples));
    batchlabs=label_disk(:,last_read+1:min(last_read+chunksz,num_total_samples));
    
    % store to hdf5
    
    startloc=struct('dat',[1,1,1,totalct+1], 'lab', [1,totalct+1]);
    %%%startloc=struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    
    curr_dat_sz=DC_store2hdf5(filename, batchdata, batchlabs, ~created_flag, startloc, chunksz, data_type, compress_level);
    created_flag=true;% flag set so that file is created only once
    totalct=curr_dat_sz(end);% updated dataset size (#samples)
end

% display structure of the stored HDF5 file
h5disp(filename);

end