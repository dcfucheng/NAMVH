function map = fastMAP(S, IX, topK)
% average precision (AP) calculation
% traingnd,testgnd: each column is a label vector
% IX: ranked list, ntrain*ntest matrix
% topK: only compute MAP on returned top topK neighbours

[numtrain, numtest] = size(IX);

apall = zeros(1,numtest);
for i = 1 : numtest
    y = IX(:,i);
    x=0;
    p=0;
    new_label=zeros(1,numtrain);
    new_label(S(i,:))=1;
    
    num_return_NN = min(topK,numtrain);
    for j=1:num_return_NN
        if new_label(y(j))==1
            x=x+1;
            p=p+x/j;
        end
    end  
    if p==0
        apall(i)=0;
    else
        apall(i)=p/x;
    end
end
map = mean(apall);
end