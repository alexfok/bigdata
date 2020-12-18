

data = load("EX2q4_data.mat");
Xtrain = data.Xtrain;
Xtest = data.Xtest;
Ytrain = data.Ytrain;
Ytest = data.Ytest;
lambdas = [1, 10, 100];
kFold = 5;
c = cvpartition(length(Ytrain), 'kFold', kFold);
params = zeros(3,2);
for i = 1:length(lambdas)
    l = lambdas(i);
    foldError = 0;
    for k = 1:kFold
        x_train_fold = Xtrain(c.training(k),:);
        y_train_fold = Ytrain(c.training(k));
        w = softsvm(l, x_train_fold, y_train_fold);
        x_test_fold = Xtrain(c.test(k),:);
        y_test_fold = Ytrain(c.test(k));
        new_y = Xtest*w;
        foldError = foldError + mean(sign(new_y) ~= Ytest);
    end
    params(i, 1) = foldError/kFold;
    params(i, 2) = l;
    disp(params(i, :));
end

disp(params);
ylabel('Error value');
xlabel('Number of parameter combinations');
plot(params(:,1));
[val, bestResultIndex] = min(params(:,1));
w = softsvm(params(bestResultIndex, 2), Xtrain, Ytrain);
new_y = Xtest*w;
error = mean(sign(new_y) ~= Ytest);
fprintf("Error: %f with lambda: %f\n",error, params(bestResultIndex, 2));

