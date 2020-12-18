function Kernel = rbf_kernel(X1,X2,sigma)
    % Building Gaussian Kernel
    Kernel = zeros(size(X1,1),size(X2,1));
    for i = 1:size(X1,1)
        for j = 1:size(X2,1)
            Kernel(i,j) = exp(-sigma*norm(X1(i,:)-X2(j,:))^2);
        end
    end
end

