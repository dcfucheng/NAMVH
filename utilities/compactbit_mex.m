function cb = compactbit_mex(b, wordsize)
% each column is a sample
% cb = compacted string of bits (using words of 'wordsize' bits)
% default wordsize is 8

if (nargin<2)
  wordsize = 8;
end

if(~islogical(b))
    warning('b must be logical, convert it to logical by b=b>0;');
    b = b>0;
end

if (wordsize == 8)
  cb = compactbit_mex_8(b,8);
elseif (wordsize == 16)
  cb = compactbit_mex_16(b,16);
elseif (wordsize == 32)
  cb = compactbit_mex_32(b,32);
elseif (wordsize == 64)
  cb = compactbit_mex_64(b,64);
else
  error('unrecognized wordsize');
end

end