function f = pinball(sigma, xi, mu)
% quantile
q = norminv([0.01:0.01:0.99], mu, sigma);

% pin ball loss calculation
f = 0;
for i = 1:99
    if xi < q(i)
        f = f + (1-(i/100))*(q(i)-xi);
    else
        f = f - (i/100)*(q(i)-xi);
    end
end
