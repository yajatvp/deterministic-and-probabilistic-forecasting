function f = opt_sigma(xi, mu)
% estimate optimal sigma for min. pin ball loss using fmincon
LB              = [0];
UB              = [500];
options         = optimoptions('fmincon','StepTolerance',...
    1e-07,'OptimalityTolerance',1e-07,'Algorithm','sqp','MaxIterations',200);


f = fmincon(@(sigma) pinball(sigma, xi, mu), [100], [], [], [], [], LB, UB, [], options);