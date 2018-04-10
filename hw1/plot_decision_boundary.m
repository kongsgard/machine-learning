function plot_decision_boundary(N,degree,w)
%PLOT_DECISION_BOUNDARY Plots a decision boundary defined as the contour
%where w'*phi equals zero

    x1 = linspace(-5, 5, N);
    x2 = linspace(-5, 5, N);

    z = zeros(length(x1), length(x2));
    for i = 1:length(x1)
        for j = 1:length(x2)
            z(i,j) = w'*map_feature(x1(i),x2(j),degree);
        end
    end
    z = z';

    contour(x1,x2,z,[0,0],'LineWidth',2);
end
