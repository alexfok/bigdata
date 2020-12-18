data = load("EX2q4_data.mat")
gscatter(data.Xtrain(:,1), data.Xtrain(:,2), data.Ytrain, 'rb', 'o', 5);
