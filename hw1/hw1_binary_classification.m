%% 1) Generate 2D synthetic data for binary classification
%https://www.mathworks.com/help/stats/simulate-data-from-a-gaussian-mixture-model.html
N = 200; % Number of samples

% Class 0
m        = [0, 0];

theta    = 0;
lambda_1 = 2; u_1 = [cos(theta), sin(theta)]'; lambda_2 = 1; u_2 = [-sin(theta), cos(theta)]';
C        = lambda_1*(u_1*u_1') + lambda_2*(u_2*u_2');

gm_0 = gmdistribution(m,C);

rng('default'); % For reproducibility
X_0 = random(gm_0,N);
figure(1), clf; hold on;
%gmPDF = @(x,y)pdf(gm_0,[x y]);
%ezcontour(gmPDF);
scatter(X_0(:,1),X_0(:,2),100,'.') % Scatter plot with points of size 100
title('Simulated Data, Class 0','FontSize',20);

% Class 1
% Component A:
pi_A     = 1/3;
m_A      = [-2, 1];

theta    = -3/4 * pi;
lambda_1 = 2; u_1 = [cos(theta), sin(theta)]'; lambda_2 = 1/4; u_2 = [-sin(theta), cos(theta)]';
C_A      = lambda_1*(u_1*u_1') + lambda_2*(u_2*u_2');
% Component B:
pi_B     = 2/3;
m_B      = [3, 2];

theta    = pi/4;
lambda_1 = 3; u_1 = [cos(theta), sin(theta)]'; lambda_2 = 1; u_2 = [-sin(theta), cos(theta)]';
C_B      = lambda_1*(u_1*u_1') + lambda_2*(u_2*u_2');

% Gaussian Mixture of Component A and Component B
gm_1 = gmdistribution([m_A; m_B], cat(3,C_A,C_B), [pi_A, pi_B]);

rng('default'); % For reproducibility
X_1 = random(gm_1,N);
%figure(2); clf; hold on;
%gmPDF = @(x,y)pdf(gm_1,[x y]);
%ezcontour(gmPDF);
scatter(X_1(:,1),X_1(:,2),100,'.') % Scatter plot with points of size 100
title('Simulated Data, Class 1','FontSize',20);

%% 2) MAP Decision Rule
% $$\hat{t}_{\mathrm{MAP}}(x) = \arg \max_{C_k} \left[ \ln P(x | C_k) + \ln
% P(C_k) \right]$$
%
% Let $\{ C_0, C_1, ..., C_k \}$ be the classes, and let $x$ be a vector of features


% Define Phi = [1, x1, x2, x1^2, x2^2, x1*x2]';

Phi = [ones(1,2*N)                                      % 1*N matrix
       X_0' X_1';                                       % 2*N matrix
       X_0.^2' X_1.^2';                                 % 2*N matrix
       (X_0(:,1).*X_0(:,2))' (X_1(:,1).*X_1(:,2))';     % 1*N matrix
       ];
t = [zeros(1,N), ones(1,N)]';
w = zeros(6,1);

for i = 1:N
    y = (sigmf(w'*Phi,[1 0]))';
    R = diag(y.*(1-y));
    w = w - inv(Phi*R*Phi')*Phi*(y-t);
end

% Plot the contour where w*phi equals zero
u = linspace(-5, 5, 200);
v = linspace(-5, 5, 200);

z = zeros(length(u), length(v));
for i = 1:length(u)
    for j = 1:length(v)
        phi = [1;
             u(i);
             v(j);
             u(i)^2;
             v(j)^2;
             u(i)*v(j);
            ];
        z(i,j) = w'*phi;
        %z(i,j) = w'*map_feature(u(i), v(j),2);
    end
end
z = z';

contour(u, v, z, [0,0], 'LineWidth', 2)
legend('y = 0', 'y = 1', 'Decision boundary')


