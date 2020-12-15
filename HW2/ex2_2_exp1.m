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

    % Calculate iteration errors
    meanTestErrors(i) = mean(sign(Xtest*w) ~= Ytest);
    disp("meantestErrors(i):");
    disp(meanTestErrors(i));
    meanTrainErrors(i) = mean(sign(Xtrain*w) ~= Ytrain);
    disp("meantrainErrors(i):");
    disp(meanTrainErrors(i));
    
    meanMaxTestErrors(i,:) = [mean(meanTestErrors(i)), max(meanTestErrors(i))];
    meanMaxTrainErrors(i,:) = [mean(meanTrainErrors(i)), max(meanTrainErrors(i))];
   
end
% Calculate accumulated errors
% minMaxTestErrors = [min(meanTestErrors), mean(meanTestErrors), max(meanTestErrors)];
% minMaxTrainErrors = [min(meanTrainErrors), mean(meanTrainErrors), max(meanTrainErrors)];

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

figure
bar(meanMaxTestErrors);
xlabel("log(Lambda)");
ylabel("Mean Max Test Errors");

figure
bar(meanMaxTrainErrors);
xlabel("log(Lambda)");
ylabel("Mean Max Train Errors");

% % Plot accumulated errors
% figure
% x = categorical(["Min", "Mean", "Max"]);
% x = reordercats(x,["Min", "Mean", "Max"]);
% bar(x, minMaxTestErrors);
% xlabel("train and test errors");
% ylabel("Min Mean Max sample error");
% hold('on');
% bar(x, minMaxTrainErrors);
% legend('test errors', 'train errors', 'Location','southeast')
% hold('off');
