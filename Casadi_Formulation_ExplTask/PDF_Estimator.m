function [Phi_hat_x_1_next, Phi_hat_x_2_next, Estim_sol] = PDF_Estimator(Phi_hat_x_1_last, Phi_hat_x_2_last, X_e, V_Xe, Par_PDF)

x_1 = Par_PDF.x_1;
x_2 = Par_PDF.x_2;
a = Par_PDF.Meas_mean;

% Likelihood function for Theta given measurements (functions projected on x_1 and x_2)
V_bar = V_Xe - a + 0.004; % 0.0005; for large distanced defects | 0.01 regular ones
% Meas = Meas - min(Meas);

exp_x1 = sum( V_bar.*X_e(:,1) ) / sum(V_bar);
var_x1 = sum( V_bar.*(X_e(:,1).^2) ) / sum(V_bar) - exp_x1^2;
P_V_k_x1 = (1/(sqrt(2*pi*var_x1)))*exp( -(x_1 - exp_x1).^2/(2*var_x1));

exp_x2 = sum( V_bar.*X_e(:,2)) / sum(V_bar);
var_x2 = sum( V_bar.*(X_e(:,2).^2)) / sum(V_bar) - exp_x2^2;
P_V_k_x2 = (1/(sqrt(2*pi*var_x2)))*exp( -(x_2 - exp_x2).^2/(2*var_x2));

% Previous PDF
P_theta_x1 = Phi_hat_x_1_last;
P_theta_x2 = Phi_hat_x_2_last;

% Next PDF
P_theta_x1_next = P_V_k_x1.*P_theta_x1;
exp_x1_next = sum( P_theta_x1_next.*x_1) / sum(P_theta_x1_next);
var_x1_next = sum( P_theta_x1_next.*(x_1.^2)) / sum(P_theta_x1_next) - exp_x1_next^2;
Phi_hat_x_1_next = normpdf(x_1, exp_x1_next, sqrt(var_x1_next));

P_theta_x2_next = P_V_k_x2.*P_theta_x2;
exp_x2_next = sum( P_theta_x2_next.*x_2) / sum(P_theta_x2_next);
var_x2_next = sum( P_theta_x2_next.*(x_2.^2)) / sum(P_theta_x2_next) - exp_x2_next^2;
Phi_hat_x_2_next = normpdf(x_2, exp_x2_next, sqrt(var_x2_next));

Estim_sol.V_bar = V_bar;
Estim_sol.exp_x1_V_Xe = exp_x1;
Estim_sol.var_x1_V_Xe = var_x1;
Estim_sol.exp_x2_V_Xe = exp_x2;
Estim_sol.var_x2_V_Xe = var_x2;
Estim_sol.exp_x1_hat = exp_x1_next;
Estim_sol.var_x1_hat = var_x1_next;
Estim_sol.exp_x2_hat = exp_x2_next;
Estim_sol.var_x2_hat = var_x2_next;

end