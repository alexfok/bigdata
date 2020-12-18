data = load("EX2q4_data.mat");
Xtrain = data.Xtrain;
Xtest = data.Xtest;
Ytrain = data.Ytrain;
Ytest = data.Ytest;
lambdas = [1, 10, 100];
sigmas = [0.01, 0.5, 1];
kFold = 5;
c = cvpartition(length(Ytrain), 'kFold', kFold);
params = zeros(9,3);
foldError = zeros(kFold, 1);
for i = 1:length(lambdas)
    l = lambdas(i);
    for j = 1:length(sigmas)
        s = sigmas(j);
        for k = 1:kFold
            x_train_fold = Xtrain(c.training(k),:);
            y_train_fold = Ytrain(c.training(k));
            alpha = softsvmrbf(l, s, x_train_fold, y_train_fold);
            x_test_fold = Xtrain(c.test(k),:);
            y_test_fold = Ytrain(c.test(k));
            new_y = softsvmrbf_predict(x_test_fold, x_train_fold, y_train_fold, alpha, s);
            foldError(k) = mean(sign(new_y) ~= y_test_fold);
        end
        ind = i * length(sigmas) + j - length(sigmas);
        params(ind, 1) = mean(foldError);
        params(ind, 2) = l;
        params(ind, 3) = s;
        disp(params(ind, :));
    end
end

disp(params);
ylabel('Error value');
xlabel('Parameter combinations');
plot(params(:,1));
[val, bestResultIndex] = min(params(:,1));
alpha = softsvmrbf(params(bestResultIndex, 2), params(bestResultIndex, 3), Xtrain, Ytrain);
new_y = softsvmrbf_predict(Xtest, Xtrain, Ytrain, alpha, params(bestResultIndex, 3));
error = mean(sign(new_y) ~= Ytest);
fprintf("Error: %f with lambda: %f and sigma: %f\n",error, params(bestResultIndex, 2), params(bestResultIndex, 3));
