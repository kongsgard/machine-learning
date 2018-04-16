function [c0_incorrect_probability, c1_incorrect_probability] = classify_generated_samples_from_feature_vector(N,Phi,w)
%CLASSIFY_GENERATED_SAMPLES Find the percentage of correct classifications
%of the generated samples using the feature vector w
    
    c0_correct = 0;
    c1_correct = 0;

    for i = 1:N
        z = w'*Phi(:,i);
        if (i <= N && z < 0)
            % Sample from Class 0 classified as Class 0
            c0_correct = c0_correct + 1;
        end
        if (i > N/2 && z > 0)
            % Sample from Class 1 classified as Class 1
            c1_correct = c1_correct + 1;
        end     
    end

    c0_incorrect_probability = (N/2 - c0_correct)/(N/2);
    c1_incorrect_probability = (N/2 - c1_correct)/(N/2);
end