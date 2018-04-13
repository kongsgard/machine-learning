function [p] = getProbability(data, class)
load('parameters.mat')

p = zeros(1,size(data,2));

for i=1:size(data,2)
    x = data(:,i);
    if     class == 0
        p(i) = 1/((2*pi)^1*det(C_0 )^0.5)*exp(-1/2*(x-m_0 )'*inv(C_0 )*(x-m_0 ));
    elseif class == 1
        p_A  = 1/((2*pi)^1*det(C_1A)^0.5)*exp(-1/2*(x-m_1A)'*inv(C_1A)*(x-m_1A));
        p_B  = 1/((2*pi)^1*det(C_1B)^0.5)*exp(-1/2*(x-m_1B)'*inv(C_1B)*(x-m_1B));
        p(i) = pi_1A*p_A + pi_1B*p_B;
    else
        p(i) = 0;
    end
end

end