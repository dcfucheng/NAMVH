function [a,IX] =  DC_sort(dist,type,splits)
N_test = size(dist,1);
N_retri = size(dist,2);
if (N_retri>500000) %%big problem
    num_one_split = ceil(N_test/splits);
    IX =[];
    for sp = 1:splits
        temp_D = dist(num_one_split*(sp-1)+1:min(num_one_split*sp,N_test),:);
        [a,tempIX] = sort(temp_D,2,type);
        IX = [IX;tempIX];
		fprintf('DC_sort %d splits.\n', sp);
    end    
else
    [a,IX] = sort(dist,2,type);
end

end