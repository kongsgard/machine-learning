function correct_classification_percentage = classify_generated_samples_from_feature_vector(N,Phi,w)
%CLASSIFY_GENERATED_SAMPLES Find the percentage of correct classifications
%of the generated samples using the feature vector w
    
    c0_correct = 0;
    c1_correct = 0;

    for i = 1:2*N
        z = w'*Phi(:,i);
        if (z < 0 && i <= N)
            % Sample from Class 0 classified as Class 0
            c0_correct = c0_correct + 1;
        end
        if (z > 0 && i > N)
            % Sample from Class 1 classified as Class 1
            c1_correct = c1_correct + 1;
        end     
    end

    correct_classification_percentage = ((c0_correct + c1_correct) / (2*N))*100;
end

