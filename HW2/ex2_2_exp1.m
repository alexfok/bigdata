% Load data file
cifar_data = load('EX2q2_data.mat');

n = [0:15];
lambdas = zeros(length(n), 1);

% Experiment 1
meanTestErrors = zeros(length(n), 1);
meanTrainErrors = zeros(length(n), 1);
meanMaxTrainErrors = zeros(length(n), 2);
meanMaxTestErrors = zeros(length(n), 2);

Xtest = cifar_data.Xtest;
Ytest = cifar_data.Ytest;
Xtrain = cifar_data.Xtrain;
Ytrain = cifar_data.Ytrain;


for i=1:length(n)
    % lambda - algorithm parameter
    lambdas(i) = 10^n(i);
    % Calculate and measure time of softsvm
    fprintf("Calculating softsvm, iteration: %d\n", i);
    tic;
    w = softsvm(lambdas(i), Xtrain, Ytrain);
    t = toc;
    fprintf("Finished calculating softsvm, iteration: %d, time: %f\n", i, t);

    % Calculate lambda iteration errors
    meanTestErrors(i) = mean(sign(Xtest*w) ~= Ytest);
    disp("meantestErrors(i):");
    disp(meanTestErrors(i));
    meanTrainErrors(i) = mean(sign(Xtrain*w) ~= Ytrain);
    disp("meantrainErrors(i):");
    disp(meanTrainErrors(i));
    
    meanMaxTestErrors(i,:) = [mean(meanTestErrors(i)), max(meanTestErrors(i))];
    meanMaxTrainErrors(i,:) = [mean(meanTrainErrors(i)), max(meanTrainErrors(i))];
   
end

disp("Avg: ")
disp(meanTestErrors);

disp("Avg: ")
disp(meanTrainErrors);

% Plot mean iteration errors
figure
a1 = plot(n, meanTestErrors);
hold('on');
a2 = plot(n, meanTrainErrors);
xlabel("log(Lambda)");
ylabel("Average sample error");
M1 = "Test Error";
M2 = "Train Error";
legend([a1;a2], M1, M2, 'Location','southeast')
hold('off');

% Graph mean and max errors - for test data
figure
bar(meanMaxTestErrors);
xlabel("log(Lambda)");
ylabel("Mean Max Test Errors");

% Graph mean and max errors - for train data
figure
bar(meanMaxTrainErrors);
xlabel("log(Lambda)");
ylabel("Mean Max Train Errors");
