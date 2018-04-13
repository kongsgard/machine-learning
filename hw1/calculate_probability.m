function p = calculate_probability(X,class)
%CALCULATE_PROBABILITY Calculate probability of gaussian distribution

    load parameters.mat

    d = 2; % 2D data
    p = zeros(1,size(X,2));
    for i = 1:size(X,2)
        x = X(:,i);
        if class == 0
            p(i) = 1/((2*pi)^(d/2) * det(C_0)^(1/2)) * exp(-(1/2)*(x-m_0)'*inv(C_0)*(x-m_0));
        elseif class == 1
            p_A = 1/((2*pi)^(d/2) * det(C_A)^(1/2)) * exp(-(1/2)*(x-m_A)'*inv(C_A)*(x-m_A));
            p_B = 1/((2*pi)^(d/2) * det(C_B)^(1/2)) * exp(-(1/2)*(x-m_B)'*inv(C_B)*(x-m_B));
            p(i) = pi_A*p_A + pi_B*p_B;
        end
    end
end