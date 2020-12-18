function new_Y = softsvm_predict(new_X, Xtrain, Ytrain, alpha, sigma)
    kernel = rbf_kernel(Xtrain, new_X, sigma);
    new_Y = sum(diag(alpha.*Ytrain) * kernel, 1)';
end

