function data = get_data(data, split)

ind_train = split.ind_train;
ind_test = split.ind_test;
ind_retri = split.ind_retri;

data.Xtrain = data.X(:,ind_train);
data.Xtest = data.X(:,ind_test);
data.Xretri = data.X(:,ind_retri);

data.Ytrain = data.Y(:,ind_train);
data.Ytest = data.Y(:,ind_test);
data.Yretri = data.Y(:,ind_retri);

end