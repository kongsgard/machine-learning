%% 1) Generate 2D synthetic data for binary classification
%https://www.mathworks.com/help/stats/simulate-data-from-a-gaussian-mixture-model.html
N = 100; % Number of samples

% Class 0
m_0      = [0, 0]';

theta    = 0;
lambda_1 = 2; u_1 = [cos(theta), sin(theta)]'; lambda_2 = 1; u_2 = [-sin(theta), cos(theta)]';
C_0      = lambda_1*(u_1*u_1') + lambda_2*(u_2*u_2');

gm_0 = gmdistribution(m_0',C_0);

rng('default'); % For reproducibility
X_0 = random(gm_0,N)';
%figure(1), clf; hold on;
%gmPDF = @(x,y)pdf(gm_0,[x y]); ezcontour(gmPDF);
%scatter(X_0(1,:),X_0(2,:),100,'.') % Scatter plot with points of size 100
%title('Simulated Data, Class 0','FontSize',20);

% Class 1
% Component A:
pi_A     = 1/3;
m_A      = [-2, 1]';

theta    = -3/4 * pi;
lambda_1 = 2; u_1 = [cos(theta), sin(theta)]'; lambda_2 = 1/4; u_2 = [-sin(theta), cos(theta)]';
C_A      = lambda_1*(u_1*u_1') + lambda_2*(u_2*u_2');
% Component B:
pi_B     = 2/3;
m_B      = [3, 2]';

theta    = pi/4;
lambda_1 = 3; u_1 = [cos(theta), sin(theta)]'; lambda_2 = 1; u_2 = [-sin(theta), cos(theta)]';
C_B      = lambda_1*(u_1*u_1') + lambda_2*(u_2*u_2');

% Gaussian Mixture of Component A and Component B
gm_1 = gmdistribution([m_A'; m_B'], cat(3,C_A,C_B), [pi_A, pi_B]);

rng('default'); % For reproducibility
X_1 = random(gm_1,N)';
%figure(2); clf; hold on;
%gmPDF = @(x,y)pdf(gm_1,[x y]); ezcontour(gmPDF);
%scatter(X_1(1,:),X_1(2,:),100,'.') % Scatter plot with points of size 100
%title('Simulated Data, Class 1','FontSize',20);

save parameters.mat m_0 C_0 pi_A m_A C_A pi_B m_B C_B

%% 2) Classification with the MAP Decision Rule
X = [X_0 X_1];
t = classify_with_map(X);

u = linspace(-5, 5, N);
v = linspace(-5, 5, N);

z = zeros(length(u), length(v));

for i = 1:length(u)
    for j = 1:length(v)
        x(:,j) = [u(i) v(j)]';
    end
    class_0_probability = calculate_probability(x,0);
	class_1_probability = calculate_probability(x,1);
    z(:,i) = class_1_probability - class_0_probability;
end

figure('Name','Binary Classification'); clf; hold on;
scatter(X_0(1,:),X_0(2,:),100,'.');
scatter(X_1(1,:),X_1(2,:),100,'.');
contour(u,v,z,[0,0],'LineWidth',2);
legend({'$t = 0$','$t = 1$','Decision Boundary'},'Interpreter','Latex','FontSize',20,'Location','SouthEast');


%% 3) Estimate the conditional probability of incorrect classification for each class
% From the classification in the previous section, we see that we get about
% a 80% correct classification. The expected error probability is therefore
% 20%.
% The number of samples should be at least 10 times larger than the inverse
% of the expected error probability, that is N > 10*5
% Therefore, a good choice is N=200.

% save('class_samples.mat','X_0','X_1');

%% 4) Kernelized logistic regression
N = 200; % Define the old N above to be N/2 (half of this new N)
l = 10^-1;
lambda = 1;

X = [X_0 X_1];
K = zeros(N,2);
for i = 1:N
    for j = 1:N
        K(i,j) = exp(-norm(X(:,i)-X(:,j))^2/(2*l^2));
    end
end

t = [zeros(1,N/2), ones(1,N/2)]';

a = zeros(N,1);
for i = 1:10
    y = (sigmf(a'*K,[1 0]))';
    R = diag(y.*(1-y));
    E_map_grad = K * (y - t + lambda*a);
    H = K*R*K + lambda*K;
    
    a_old = a;
    a = a - H\E_map_grad;
    
    if abs(a - a_old) < 0.01
        fprintf('Newton iterations converged in %d steps.\n',i);
        break;
    end
end

%% 5) Plot training data points and show the decision boundaries
u = linspace(-5, 5, N);
v = linspace(-5, 5, N);
z = zeros(N,N);

X = [X_0 X_1];
K = zeros(N,2);
for i = 1:N
    for j = 1:N
        x = [u(i) v(j)]';
        K(i,j) = exp(-norm(X(:,i)-x)^2/(2*l^2));
    end
end

for i = 1:N
    z(:,i) = a'*K(:,i);
end

figure('Name','Binary Classification'); clf; hold on;
scatter(X_0(1,:),X_0(2,:),100,'.');
scatter(X_1(1,:),X_1(2,:),100,'.');
contour(u,v,z,[0,0],'LineWidth',2);
legend({'$t = 0$','$t = 1$','Decision Boundary'},'Interpreter','Latex','FontSize',20,'Location','SouthEast');

%% 7) Non-kernelized logistic regression
degree = 3; % The feature vector will include all monomials up to the degree'th power.

[w,Phi] = newton_update([X_0 X_1],N,degree);

figure('Name','Binary Classification'); clf; hold on;
scatter(X_0(1,:),X_0(2,:),100,'.');
scatter(X_1(1,:),X_1(2,:),100,'.');
plot_decision_boundary(N,degree,w);
legend({'$t = 0$','$t = 1$','Decision Boundary'},'Interpreter','Latex','FontSize',20,'Location','SouthEast');

% Classification
fprintf('Percentage of correct classification of generated samples: %2.2f\n', classify_generated_samples_from_feature_vector(N,Phi,w));

% No overfitting observed



