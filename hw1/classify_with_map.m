function t = classify_with_map(X,N)
%CLASSIFY_WITH_MAP Binary classification of X

    class_0_probability = calculate_probability(X,N,0);
    class_1_probability = calculate_probability(X,N,1);

    t = zeros(1,N);
    for i = 1:N
        if (log(class_0_probability(i)) > log(class_1_probability(i)))
            t(i) = 0;
        else
            t(i) = 1;
        end
    end
end

