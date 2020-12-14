% Experiment 2
repeates = 10;
% Load data file
cifar_data = load('EX2q2_data.mat');

n = [0:15];
%n = [3,6,9,12,15];
lambdas = zeros(length(n), 1);

iterTrainErrors = zeros(repeates, 1);
iterTestErrors = zeros(repeates, 1);

meanTrainErrors = zeros(length(n), 1);
meanTestErrors = zeros(length(n), 1);
minMaxTrainErrors = zeros(length(n), 2);

for i=1:length(n)
    for j=repeates
        % lambda - algorithm parameter
        lambdas(i) = 10^n(i);
        fprintf("Calculating softsvm, iteration: %d\n", i);
        [Xtrain, Ytrain] = gensmall_sample(cifar_data.Xtrain, cifar_data.Ytrain, 100);
        [Xtest, Ytest] = gensmall_sample(cifar_data.Xtest, cifar_data.Ytest, 100);
        tic;
        w = softsvm(lambdas(i), Xtrain, Ytrain);
        t = toc;
        fprintf("Finished calculating softsvm, iteration: %d, time: %f\n", i, t);
        % Calculate iteration error
        iterTrainErrors(j) = mean(sign(Xtrain*w) ~= Ytrain);
        iterTestErrors(j) = mean(sign(Xtest*w) ~= Ytest);
    end
    meanTrainErrors(i) = mean(iterTrainErrors);
    meanTestErrors(i) = mean(iterTestErrors);
    minMaxTrainErrors(i,:) = [min(iterTrainErrors), max(iterTrainErrors)];
end

% Calculate lambdas and plot it on logarithmic scale
% disp("minTrainErrors: ")
% disp(minTrainErrors);
disp("meantrainErrors: ")
disp(meantrainErrors);

figure
plot(n, meanTrainErrors);
xlabel("log(Lambda)");
ylabel("Average sample error");
hold('on');
plot(n, meanTestErrors);
legend('train error', 'test error', 'Location','southeast')
hold('off');

figure
bar(n, minMaxTrainErrors);
xlabel("log(Lambda)");
ylabel("min Max Train Errors");
