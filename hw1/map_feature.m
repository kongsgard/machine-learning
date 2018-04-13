function phi = map_feature(feat1, feat2, degree)
% MAP_FEATURE    Feature mapping function
%
%   map_feature(feat1, feat2) maps the two input features
%   to higher-order features as defined in the following way:
%   phi = [1, x1, x2, x1^2, x1*x2, x2^2, x1^3, x1^2*x2, x1*x2^2, x2^3, ...]';
%
%   Inputs feat1, feat2 must be the same size

    [m,~] = size(feat1);
    if m ~= 1
        x1 = [feat1(:,1); feat2(:,1)];
        x2 = [feat1(:,2); feat2(:,2)];
    else
        x1 = feat1;
        x2 = feat2;
    end
      
    phi = ones(size(x1(:,1)));
    for i = 1:degree
        for j = 0:i
            phi(:, end+1) = (x1.^(i-j)).*(x2.^j);
        end
    end
end