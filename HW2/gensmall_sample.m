function [X,Y] = gensmall_sample(data, labels, samplesize)
[m,d] = size(data);
perm = randperm(m);
sample_idxes = perm(1:samplesize);
X = data(sample_idxes,:);
Y = labels(sample_idxes);
