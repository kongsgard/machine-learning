function [w,Phi] = newton_update(X_0,X_1,N,degree)
%NEWTON_UPDATE Iteratively updates the feature vector w using the newton
%update
%   If equal priors is assumed, this function can be used as an
%   implementation of the MAP rule.

    % Define Phi = [1, x1, x2, x1^2, x1*x2, x2^2, ...]';
    Phi = map_feature(X_0,X_1,degree);
    
    t = [zeros(1,N), ones(1,N)]';

    [n_monomials,~] = size(Phi);
    w = zeros(n_monomials,1);

    for i = 1:N
        y = (sigmf(w'*Phi,[1 0]))';
        R = diag(y.*(1-y));
        w = w - inv(Phi*R*Phi')*Phi*(y-t);
    end
end

