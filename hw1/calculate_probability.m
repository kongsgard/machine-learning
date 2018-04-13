function p = calculate_probability(X,N,class)
%CALCULATE_PROBABILITY Calculate probability of gaussian distribution

    load parameters.mat
    m = m'; m_A = m_A'; m_B = m_B'; % Defined differently in the samples generation

    d = 2; % 2D data
    p = zeros(1,N);
    for i = 1:N
        x = X(1,:)';
        if class == 0
            p(i) = 1/((2*pi)^(d/2) * det(C)^(1/2)) * exp(-(1/2)*(x-m)'*inv(C)*(x-m));
        elseif class == 1
            p_A = 1/((2*pi)^(d/2) * det(C_A)^(1/2)) * exp(-(1/2)*(x-m)'*inv(C_A)*(x-m));
            p_B = 1/((2*pi)^(d/2) * det(C_B)^(1/2)) * exp(-(1/2)*(x-m)'*inv(C_B)*(x-m));
            p(i) = pi_A*p_A + pi_B*p_B;
        end
    end
end

