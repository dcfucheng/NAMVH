function Precision = topK_Pre(S, IX, topK)

ntest = size(S,1);

retrieved_good_pairs = zeros(1,length(topK));
retrieved_pairs = zeros(1,length(topK));
for i = 1:ntest%for each query
    if mod(i,500)==0
       fprintf('%d...',i);
    end
    ind1 = IX(:,i);
    sim = S(i,:);
    for r = 1:length(topK)
        ind = ind1(1:topK(r));
        retrieved_pairs(r) = retrieved_pairs(r) + length(ind);
        retrieved_good_pairs(r) = retrieved_good_pairs(r) + sum(sim(:,ind));
    end
end
fprintf('\n');

Precision = retrieved_good_pairs./(retrieved_pairs+eps);

end