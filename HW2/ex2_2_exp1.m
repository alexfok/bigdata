repeates = 10;
% Load data file
cifar_data = load('EX2q2_data.mat');

n = [0:15];
%n = [3,6,9,12,15];
lambdas = zeros(length(n), 1);

% Experiment 1
meantestErrors = zeros(length(n), 1);
%minMaxErrors = zeros(length(n), 2);
minMaxTestErrors = zeros(length(n), 2);

meantrainErrors = zeros(length(n), 1);
minMaxTrainErrors = zeros(length(n), 2);

Xtest = cifar_data.Xtest;
Ytest = cifar_data.Ytest;
Xtrain = cifar_data.Xtrain;
Ytrain = cifar_data.Ytrain;
% [Xtest, Ytest] = gensmall_sample(cifar_data.Xtest, cifar_data.Ytest, 500)
% [Xtrain, Ytrain] = gensmall_sample(cifar_data.Xtrain, cifar_data.Ytrain, 500)


for i=1:length(n)
    % lambda - algorithm parameter
    lambdas(i) = 10^n(i);
    fprintf("Calculating softsvm, iteration: %d\n", i);
    tic;
    w = softsvm(lambdas(i), Xtrain, Ytrain);
    t = toc;
    fprintf("Finished calculating softsvm, iteration: %d, time: %f\n", i, t);
    % Calculate iteration error
    
%    testIterErrors = sign(Xtest*w) ~= Ytest;
    meantestErrors(i) = mean(sign(Xtest*w) ~= Ytest);
    disp("meantestErrors(i):");
    disp(meantestErrors(i));
%    minMaxTestErrors(i,:) = [min(testIterErrors), max(testIterErrors)];
    
%    trainIterErrors = sign(Xtrain*w) ~= Ytrain;
    meantrainErrors(i) = mean(sign(Xtrain*w) ~= Ytrain);
    disp("meantrainErrors(i):");
    disp(meantrainErrors(i));
%    minMaxTrainErrors(i,:) = [min(trainIterErrors), max(trainIterErrors)];
    
end
%meanErrors = mean(errors);
%minMaxErrors = [min(meanErrors), max(meanErrors)];

% TODO:
% Calculate lambdas and plot it on logarithmic scale
disp("Avg: ")
disp(meantestErrors);
disp("Min Max: ")
disp(minMaxTestErrors);

disp("Avg: ")
disp(meantrainErrors);
disp("Min Max: ")
disp(minMaxTrainErrors);

% figure
% loglog(lambdas, meantestErrors, lambdas, meantrainErrors)
% xlabel("Lambda");
% ylabel("Average sample error");
% legend('Test Error','Train Error','Location','southeast')

figure
a1 = plot(n, meantestErrors);
hold('on');
a2 = plot(n, meantrainErrors);
xlabel("log(Lambda)");
ylabel("Average sample error");
M1 = "Test Error";
M2 = "Train Error";
legend([a1;a2], M1, M2, 'Location','southeast')
hold('off');

% figure
% a1 = bar(n, minMaxTestErrors);
% hold('on');
% a2 = bar(n, minMaxTrainErrors);
% xlabel("Lambda");
% ylabel("Sampel size minMaxErrors");
% M1 = "minMaxTestErrors";
% M2 = "minMaxTrainErrors";
% %legend([a1;a2], M1, M2)
% hold('off');
