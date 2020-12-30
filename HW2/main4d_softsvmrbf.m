data = load("EX2q4_data.mat");
Xtrain = data.Xtrain;
Xtest = data.Xtest;
Ytrain = data.Ytrain;
Ytest = data.Ytest;

[X,Y] = meshgrid(-10:0.5:10);

sigmas = [0.01, 0.5, 1];

lambda = 100;
sigma = 0.01;
alpha = softsvmrbf(lambda, sigma, Xtrain, Ytrain);
new_y = softsvmrbf_predict(Xtrain, Xtrain, Ytrain, alpha, sigma);
predictedY = sign(new_y);

maximum = max(abs(alpha));
alpha = alpha/maximum;
alpha = discretize(alpha, -2:0.05:2);
tbl = table(alpha);
figure
heatmap(tbl, 'alpha', 'alpha');

sigma = 0.5;
alpha = softsvmrbf(lambda, sigma, Xtrain, Ytrain);
new_y = softsvmrbf_predict(Xtrain, Xtrain, Ytrain, alpha, sigma);
predictedY = sign(new_y);
maximum = max(abs(alpha));
alpha = alpha/maximum;
alpha = discretize(alpha, -2:0.05:2);
tbl = table(alpha);
figure
heatmap(tbl, 'alpha', 'alpha');

sigma = 1;
alpha = softsvmrbf(lambda, sigma, Xtrain, Ytrain);
new_y = softsvmrbf_predict(Xtrain, Xtrain, Ytrain, alpha, sigma);
predictedY = sign(new_y);
maximum = max(abs(alpha));
alpha = alpha/maximum;
alpha = discretize(alpha, -2:0.05:2);
tbl = table(alpha);
figure
heatmap(tbl, 'alpha', 'alpha');
