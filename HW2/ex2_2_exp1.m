% Load data file
cifar_data = load('EX2q2_data.mat');

n = [0:15];
lambdas = zeros(length(n), 1);

% Experiment 1
meantestErrors = zeros(length(n), 1);
minMaxTestErrors = zeros(length(n), 2);

meantrainErrors = zeros(length(n), 1);
minMaxTrainErrors = zeros(length(n), 2);

Xtest = cifar_data.Xtest;
Ytest = cifar_data.Ytest;
Xtrain = cifar_data.Xtrain;
Ytrain = cifar_data.Ytrain;


for i=1:length(n)
    % lambda - algorithm parameter
    lambdas(i) = 10^n(i);
    fprintf("Calculating softsvm, iteration: %d\n", i);
    tic;
    w = softsvm(lambdas(i), Xtrain, Ytrain);
    t = toc;
    fprintf("Finished calculating softsvm, iteration: %d, time: %f\n", i, t);
    % Calculate iteration error
    
    meantestErrors(i) = mean(sign(Xtest*w) ~= Ytest);
    disp("meantestErrors(i):");
    disp(meantestErrors(i));
    meantrainErrors(i) = mean(sign(Xtrain*w) ~= Ytrain);
    disp("meantrainErrors(i):");
    disp(meantrainErrors(i));
   
end
%meanErrors = mean(errors);
%minMaxErrors = [min(meanErrors), max(meanErrors)];

% Calculate lambdas and plot it on logarithmic scale
disp("Avg: ")
disp(meantestErrors);

disp("Avg: ")
disp(meantrainErrors);

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
