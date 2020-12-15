function w = softsvm(lambda, Xtrain, Ytrain)
% lambda - algorithm parameter
% Xtrain - samples matrix - mXd
% Ytrain - labels vector - m
[m,d] = size(Xtrain);

% Create u:
u = [zeros(1,d) ones(1,m)*1/m];

% Create v:
v = [zeros(m,1); ones(m,1)];

% Create H:
t1 = eye(d);
t2 = zeros(d,m);
t3 = zeros(m,d);
t4 = zeros(m);
H = 2*lambda*[t1 t2;t3 t4];

% Create A:
t1 = zeros(m,d);
t2 = eye(m);
t3 = Ytrain .* Xtrain;
t4 = eye(m);
A = [t1 t2;t3 t4];

% Adjust H with small constant value - to overcome Matlab "problem is not convex" issue
small_const = repmat(2,[1,size(Xtrain,1) + size(Xtrain,2)]);
H1 = diag(small_const,0);
H = H + H1;

% Calculate quatratic program
w = quadprog(H, u, -A, -v);
% Extract w from quadprog result
w = w(1:d);
