function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%


% You need to return the following variables correctly.
X_poly = zeros(numel(X), p);

% ====================== YOUR CODE HERE ======================
% Instructions: Given a vector X, return a matrix X_poly where the p-th 
%               column of X contains the values of X to the p-th power.
%
% 

%Trying broadcasting method for faster execution instead of using a "for loop".	   
p_v = 1:1:p;

    %Using built in broadcasting function
X_poly = bsxfun(@power, X, p_v);

    %Alternatively bitwise operator can be used which is mostly same as the bsxfun function.
%X_poly = X.^p_v;

%"for loop" as shown below can also be used but it is better to go for vectorised implementations.
%for j = 1 : p
%    X_poly(:,j) = X.^j;
%endfor



% =========================================================================

end
