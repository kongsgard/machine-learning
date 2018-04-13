function t = classify_with_map(X)
%CLASSIFY_WITH_MAP Binary classification of X

    class_0_probability = calculate_probability(X,0);
    class_1_probability = calculate_probability(X,1);

    t = zeros(1,size(X,2));
    for i = 1:size(X,2)
        if (log(class_0_probability(i)) > log(class_1_probability(i)))
            t(i) = 0;
        else
            t(i) = 1;
        end
    end
end
