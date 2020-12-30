function alpha = softsvmrbf(lambda, sigma, Xtrain, Ytrain) 
    % Building Gaussian Kernel
    Kernel = rbf_kernel(Xtrain, Xtrain, sigma);
    % Calculating Alphas
    N = size(Xtrain, 1);
    H = lambda * diag(Ytrain) * Kernel * diag(Ytrain);
    f = -ones(N, 1);
    alpha=quadprog(H,f);
end
