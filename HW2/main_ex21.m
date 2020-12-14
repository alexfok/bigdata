% input sizes:
m = 20;
d = 4;

% lambda - algorithm parameter
lambda = 3

% Xtrain - samples matrix - mXd
Xtrain=[rand(m,d)]

% Ytrain - labels vector - m
Ytrain=[]
% Generate labels
labels = [-1 1];
for i = 1:m
    rnd = randi([1 2]);
    Ytrain(i) = labels(rnd);
end
Ytrain=Ytrain'

w = softsvm(lambda, Xtrain, Ytrain)
