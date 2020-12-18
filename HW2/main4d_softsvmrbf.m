data = load("EX2q4_data.mat");
Xtrain = data.Xtrain;
Xtest = data.Xtest;
Ytrain = data.Ytrain;
Ytest = data.Ytest;

sigmas = [0.01, 0.5, 1];

lambda = 100;
sigma = 0.01;
alpha = softsvmrbf(lambda, sigma, Xtrain, Ytrain);
new_y = softsvmrbf_predict(Xtrain, Xtrain, Ytrain, alpha, sigma);
predictedY = sign(new_y);
tbl = table(predictedY, Ytrain, alpha);
figure
heatmap(tbl, 'predictedY','Ytrain');

sigma = 0.5;
alpha = softsvmrbf(lambda, sigma, Xtrain, Ytrain);
new_y = softsvmrbf_predict(Xtrain, Xtrain, Ytrain, alpha, sigma);
predictedY = sign(new_y);
tbl = table(predictedY, Ytrain, alpha);
figure
heatmap(tbl, 'predictedY','Ytrain');

sigma = 1;
alpha = softsvmrbf(lambda, sigma, Xtrain, Ytrain);
new_y = softsvmrbf_predict(Xtrain, Xtrain, Ytrain, alpha, sigma);
predictedY = sign(new_y);
tbl = table(predictedY, Ytrain, alpha);
figure
heatmap(tbl, 'predictedY','Ytrain');
