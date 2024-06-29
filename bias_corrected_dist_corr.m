function [bcR, p, T, df] = bias_corrected_dist_corr(x, y)
% BCDISTCORR computes the bias corrected distance correlation
%
%   [BCR,P,T,DF] = bias_corrected_dist_corr(X,Y), where X (size n-by-p) and Y (size
%   n-by-q) are n random samples of observation.  The function returns the
%   bias corrected distance correlation BCR and the corresponding p-value
%   P, as well as the student t statistics T and its degree of freedom DF.
%   Note that the The t-test of independence is unbiased for every n ? 4
%   and any significance level.
%
%   This implementation is based on Székely, G. J., & Rizzo, M. L. (2013).
%   The distance correlation t-test of independence in high dimension.
%   Journal of Multivariate Analysis, 117, 193-213.
%
%   Date: 05.12.2020  v.1.0
%   Revised: 24.01.2021 v.1.1
%            13.03.2023 v.1.2
        
%   Author: Dr.Stavros I. Dimitriadis(stidimitriadis@gmail.com)
%%  Website:https://www.researchgate.net/profile/Stavros_Dimitriadis

assert(rows(x)==rows(y));
n = rows(x);
X = Astar(x);    % section 2.4
Y = Astar(y);    % section 2.4
XY = modified_distance_covariance(X, Y);
XX = modified_distance_covariance(X, X);
clear X;
YY = modified_distance_covariance(Y, Y);
clear Y;
bcR = XY/sqrt(XX*YY); % equation 2.10
M = n*(n-3)/2; 
T=0;
p=0;
df=0;
%T = sqrt(M-1) * bcR / sqrt(1-bcR^2); % equation 3.7
%df = M-1; %degrees of freedom
%p = 1 - tcdf(T, df); % Student test distribution
%fprintf('bias-corrected R = %.3f, p-value=%.3f, T(%d)=%.4f\n',...
%    bcR, p, df, T);
end


function XY = modified_distance_covariance(X, Y)
n = rows(X);
XY = sum(sum(bsxfun(@times, X, Y)))...
    - (n/(n-2))*diag(X)'*diag(Y);
end

function A = Astar(x)
d = pdist2(x,x);
n = rows(x);
m = mean(d);
M = mean(d(:));
% A = d - m'*ones(1,n);
A = bsxfun(@minus, d, bsxfun(@mtimes, m', ones(1,n)));
% A = A - ones(n,1)*m;
A = bsxfun(@minus, A, bsxfun(@mtimes, ones(n,1), m));
% A = A + M;
A = bsxfun(@plus, A, M); 
A = A - d/n;
A(1:n+1:end) = m - M;
A = (n/(n-1))*A;
end

function r = rows(x)
    r = size(x,1);
end