function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
pairs = [1, 0.1; 
         1, 0.3;
         3, 0.1;
         3, 0.3;];

%for i = 1:size(pairs, 1)
%  p = pairs(i, :);
%  model= svmTrain(X, y, p(1), @(x1, x2) gaussianKernel(x1, x2, p(2))); 
%  predictions = svmPredict(model, Xval);
%  errs = mean(double(predictions ~= yval));
%
%  fprintf('\nParams: %f, %f; errs: %f \n', p(1), p(2), errs);
%  visualizeBoundary(X, y, model);
%  pause;
%end

C = 1;
sigma = 0.1;







% =========================================================================

end
