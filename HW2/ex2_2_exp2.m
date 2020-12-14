repeates = 10;
% Load data file
cifar_data = load('EX2q2_data.mat');

n = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15];

meanErrors = zeros(length(n), 1);
minMaxErrors = zeros(length(n), 2);
minErrors = zeros(length(n), 1);

errors = zeros(repeates, 1);
for i=1:length(n)
    % lambda - algorithm parameter
    lambda = 10^n(i);
    % Generate test w for validation
    [smallTestX, smallTestY] = gensmall_sample(cifar_data.Xtest, cifar_data.Ytest, 100)
    WTest = softsvm(lambda, smallTestX, smallTestY);
    for j=1:repeates
        % draw sample size 100
        [smallX, smallY] = gensmall_sample(cifar_data.Xtrain, cifar_data.Ytrain, 100)
        WTrain = softsvm(lambda, smallX, smallY);
        errors(j) = mean(WTest ~= WTrain);
    end
    meanErrors(i) = mean(errors);
    minMaxErrors(i,:) = [min(errors), max(errors)];
end
%meanErrors = mean(errors);
%minMaxErrors = [min(errors), max(errors)];

disp("Avg: ")
disp(meanErrors);
disp("Min Max: ")
disp(minMaxErrors);

figure
plot(n, meanErrors);
xlabel("Lambda");
ylabel("Average sample size error");

figure
bar(n, minMaxErrors);
xlabel("Lambda");
ylabel("Sampel size error");
