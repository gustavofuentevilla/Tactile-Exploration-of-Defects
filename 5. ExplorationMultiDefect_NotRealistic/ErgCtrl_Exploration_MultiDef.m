% Primera ejecución
close all
clear
clc
M = Function.load('M.casadi');

% Resto de ejecuciones
% close all
% clearvars -except M
% clc

%% Parámetros del espacio de búsqueda U = [L_1_l, L_1_u] \times [L_2_l, L_2_u]

n = 2; % Número de dimensiones espaciales

L_1_l = 0.5;
dx_1 = 0.02;
L_1_u = 1.5;

L_2_l = 0.5;
dx_2 = 0.02;
L_2_u = 1.5;

% Dimensiones \mathbf{x} = [x_1 x_2]^T
x_1 = (L_1_l:dx_1:L_1_u)';
x_2 = (L_2_l:dx_2:L_2_u)';

%vector de límites inferior y superiores de las dimensiones
L_i_l = [L_1_l, L_2_l];
L_i_u = [L_1_u, L_2_u];

[x_1_grid, x_2_grid] = meshgrid(x_1, x_2);

%Espacio de búsqueda discretizado
Omega = [reshape(x_1_grid,[],1), reshape(x_2_grid,[],1)]; 

%% Gaussian Mixture distribution (PDF REAL)

%Media del Gaussiano
% Mu_1 = [0.75, 1.35];
% Mu_2 = [1.3, 1];
% Mu_3 = [0.9, 0.8];
% 
% Mu = [Mu_1; Mu_2; Mu_3];
% 
% % Matriz de Covarianza
% Cov_1 = [0.0005, 0;
%          0, 0.0005];
% Cov_2 = Cov_1;
% Cov_3 = Cov_1;
% 
% Sigma = cat(3,Cov_1,Cov_2,Cov_3);
% 
% %Peso sobre el Gaussiano
% proporciones = [1/3, 1/3, 1/3];
% 
% gm_dist = gmdistribution(Mu, Sigma, proporciones);
% 
% 
% %PDF de referencia REAL
% Phi_x = pdf(gm_dist, Omega);

%% Real PDF

n_def = 3;

mu_1 = [0.8, 1.2];
sigma_1 = [0.0005, 0;
         0, 0.0005];

mu_2 = [1.3, 1];
sigma_2 = sigma_1;

mu_3 = [0.9, 0.8];
sigma_3 = sigma_1;

w_pdf = 1/n_def;

Phi_x = w_pdf*mvnpdf(Omega, mu_1, sigma_1) + w_pdf*mvnpdf(Omega, mu_2, sigma_2) +...
        w_pdf*mvnpdf(Omega, mu_3, sigma_3);

%% Uniform PDF as an initial guess

Phi_hat_x_1 = unifpdf(x_1, L_1_l, L_1_u);
Phi_hat_x_2 = unifpdf(x_2, L_2_l, L_2_u);

[Phi_hat_x_1_grid, Phi_hat_x_2_grid] = meshgrid(Phi_hat_x_1, Phi_hat_x_2);
Phi_hat_x = prod([reshape(Phi_hat_x_1_grid,[],1), reshape(Phi_hat_x_2_grid,[],1)], 2);

%% Cálculo de los coeficientes de Fourier para la PDF de referencia

% Coeficientes por dimensión
K = 12;

% Conjunto de valores para k_i
k_1 = (0:K-1)';
k_2 = (0:K-1)';

[k_1_grid, k_2_grid] = meshgrid(k_1, k_2);

% Conjunto de vectores índice
K_cal = [reshape(k_1_grid,1,[]); reshape(k_2_grid,1,[])];

Par_struct.K = K;
Par_struct.n = n;
Par_struct.K_cal = K_cal;
Par_struct.Omega = Omega;
Par_struct.dx_1 = dx_1;
Par_struct.dx_2 = dx_2;
Par_struct.L_i_l = L_i_l;
Par_struct.L_i_u = L_i_u;

[phi_k_reg, f_k_reg, h_k_reg] = FourierCoef_RefPDF(Phi_hat_x, Par_struct);

%% Reconstrucción de la PDF de referencia para verificación de coeficientes

% Phi_hat_x_reconstructed = zeros(height(Omega),1);
% 
% for i = 1:height(phi_k_reg)
%     Phi_hat_x_reconstructed = Phi_hat_x_reconstructed + phi_k_reg(i)*f_k_reg(:,i);
% end
% 
% % Graficación
% 
% figure(1)
% subplot(2,2,1);
% surf(x_1_grid, x_2_grid, reshape(Phi_hat_x, length(x_2), length(x_1)))
% xlim([L_1_l, L_1_u])
% ylim([L_2_l, L_2_u])
% title("Initial Uniform PDF",'Interpreter','latex')
% xlabel('$x_1$','Interpreter','latex')
% ylabel('$x_2$','Interpreter','latex')
% zlabel('$\hat{\Phi}(\mathbf{x})$','Interpreter','latex')
% %axis equal
% grid on
% 
% subplot(2,2,2);
% contour(x_1_grid, x_2_grid, reshape(Phi_hat_x, length(x_2), length(x_1)))
% xlim([L_1_l, L_1_u])
% ylim([L_2_l, L_2_u])
% title("Initial Uniform PDF",'Interpreter','latex')
% xlabel('$x_1$','Interpreter','latex')
% ylabel('$x_2$','Interpreter','latex')
% axis equal
% grid on
% 
% subplot(2,2,3);
% surf(x_1_grid, x_2_grid, reshape(Phi_hat_x_reconstructed, length(x_2), length(x_1)))
% xlim([L_1_l, L_1_u])
% ylim([L_2_l, L_2_u])
% title("Reconstruction",'Interpreter','latex')
% xlabel('$x_1$','Interpreter','latex')
% ylabel('$x_2$','Interpreter','latex')
% zlabel('$\hat{\Phi}(\mathbf{x})$','Interpreter','latex')
% %axis equal
% grid on
% 
% subplot(2,2,4);
% contour(x_1_grid, x_2_grid, reshape(Phi_hat_x_reconstructed, length(x_2), length(x_1)))
% xlim([L_1_l, L_1_u])
% ylim([L_2_l, L_2_u])
% title("Reconstruction",'Interpreter','latex')
% xlabel('$x_1$','Interpreter','latex')
% ylabel('$x_2$','Interpreter','latex')
% axis equal
% grid on


%% Condiciones Iniciales y parámetros

N = 125; % Número de muestras por iteración
t_f = 10;           %Tiempo final por iteración
T_s = t_f/N;                  % Tiempo de muestreo
% t_f = N*T_s;     % Tiempo de simulación
t = (0:T_s:t_f)';   %Vector de tiempo por iteración

% Peso sobre controles %1e-3
R = [7e-4, 0;
     0, 7e-4];

% Peso sobre métrica ergódica
gamma = 1;

% Estado inicial z = [z_1; z_2; z_3; z_4] = [x_1; x_1_dot; x_2; x_2_dot]
z_0 = [0.5; 0; 0.5; 0]; 

%Pre-cálculo de Lambda
p = 2; %norma 2
Lambda_k = (1 + vecnorm(K_cal, p, 1)').^(-(n + 1)/2);


%% Casadi Problem Setup
import casadi.*

% Ecuaciones x_1_ddot = u_1;    x_2_ddot = u_2;
% z = MX.sym('z', 4); %states z = [z(1); z(2); z(3); z(4)] = [x_1; x_1_dot; x_2; x_2_dot]
% u = MX.sym('u', 2); %controls u = [u(1); u(2)]
% 
% %ODE construction: z_dot = [z(2); u(1); z(4); u(2)]
% z_dot = [z(2); u(1); z(4); u(2)];
% f = Function('f', {z,u}, {z_dot}, {'z', 'u'}, {'z_dot'});
% 
% %DAE problem structure
% 
% intg_options = struct;
% intg_options.tf = T_s;   %Integration time (one step ahead)
% intg_options.simplify = true;
% intg_options.number_of_finite_elements = 4; %intermediate steps on the integration horizon
% 
% dae = struct;
% 
% dae.x = z;  % states (formalized)
% dae.p = u;  % parameter, fixed during integration horizon (just one step ahead)
% dae.ode = f(z,u); % symbolic dynamics
% 
% intg = integrator('intg', 'rk', dae, intg_options); %RK4 integration method
% 
% %One step integration (numerical)
% % res = intg('x0', z_0, 'p', [0.5; 0]);  %z_0 initial condition, p = u controls
% % z_next = res.xf;
% 
% %One step integration (symbolic)
% res = intg('x0', z, 'p', u);
% z_next = res.xf;
% 
% F = Function('F', {z,u}, {z_next}, {'z','u'}, {'z_next'});
% 
% %% Multiple Shooting for one prediction horizon with N+1 samples
% opti = casadi.Opti();
% 
% z = opti.variable(4, N+1);
% u = opti.variable(2, N);
% z_0_sym = opti.parameter(4, 1);  % parameter (not optimized over): initial condition
% phi_k_sym = opti.parameter(K^n, 1);
% %u_d_sym = opti.parameter(2, 1);
% 
% % Symbolic Fourier functions, coefficients and ergodic metric with casadi
% X_e_sym = [z(1,:)', z(3,:)'];     %Position [x_1, x_2] for all N samples
% 
% c_k_sym = 0;
% f_k_traj_sym = opti.variable(K^n,1);
% J = 0;
% for i = 1:N
%     for j = 1:K^n
%         %problems using(.*) with casadi when K_cal is a matrix
%         temp = cos(K_cal(:,j)'.*pi.*(X_e_sym(i,:) - L_i_l)./(L_i_u - L_i_l));   
%         %problems using prod() function
%         f_k_traj_sym(j,1) = temp(1)*temp(2)/h_k_reg(j);
%     end
%     c_k_sym = c_k_sym + (f_k_traj_sym*T_s)/(t_f);%/i*T_s /t_f
%     Varepsilon_sym = sum( Lambda_k.*(c_k_sym - phi_k_sym).^2 );
% 
%     % Objetive function
%     J = J + gamma*Varepsilon_sym + u(:,i)'*R*u(:,i); %(u(:,i) - u_d)   u(:,i)'*R*u(:,i)
% end
% 
% % f_k_traj_sym = prod(cos( K_cal'.*pi.*(X_e_sym(i,:) - L_i_l)./(L_i_u - L_i_l) ), 2) ./ h_k_reg ;
% % c_k_sym = c_k_sym + (f_k_traj_act*T_s)/t_f ;
% % Varepsilon_act = sum( Lambda_k .* (c_k_act - phi_k_reg).^2 ); 
% 
% opti.minimize( J );
% 
% % Equality Constraints
% for k = 1:N
%     opti.subject_to( z(:,k+1) == F( z(:,k),u(:,k) ) );
% end
% opti.subject_to( z(:,1) == z_0_sym );
% 
% % Inequality Constraints
% opti.subject_to( -50 <= u <= 50 );
% opti.subject_to( L_1_l <= z(1,:) <= L_1_u );    % x_1 boundaries
% opti.subject_to( L_1_l <= z(3,:) <= L_1_u );    % x_2 boundaries
% 
% % Solver definition
% % p_opts = struct('expand',true);
% % s_opts = struct('max_iter',100);
% solver_opts = struct;
% solver_opts.expand = true;
% solver_opts.ipopt.max_iter = 2000;
% opts.ipopt.print_level = 0;
% %opts.ipopt.acceptable_tol = 1e-8;
% %opts.ipopt.acceptable_obj_change_tol = 1e-6;
% 
% opti.solver('ipopt', solver_opts);
% 
% %opti.solver('sqpmethod', struct('qpsol','qrqp'));
% 
% %setting the initial conditions
% opti.set_value(z_0_sym, z_0);
% opti.set_value(phi_k_sym, phi_k_reg);
% %opti.set_value(u_d_sym, u_d);
% 
% %Solving the optimization problem over the horizon N
% solution = opti.solve();
% 
% %Function mapping from initial condition z0 to the optimal control action
% %(first u of the N control actions). M contains IPOPT method embedded and
% %integration method RK4
% M = opti.to_function('M', {z_0_sym, phi_k_sym}, {z, u}, {'z_0','phi_k'}, {'z','u'});

%% Saving and loading casadi function object
% M.save('M.casadi');

% M = Function.load('M.casadi');

%% vector to add more points on the trajectory and get more data from sensor

t_spline = (0:0.01:t_f)'; %Time vector por spline in one iteration

%% Loop for the Search task

n_iter = 8;

% Registers
z_reg = zeros(N+1, 4, n_iter);
u_reg = zeros(N, 2, n_iter);
X_e_reg = zeros(N+1, 2, n_iter);
X_e_dot_reg = zeros(N+1, 2, n_iter);
phi_k_REG = zeros(K^n, 1, n_iter);
Phi_hat_x_reg = zeros(height(Omega), 1, n_iter);
X_e_spline_reg = zeros(length(t_spline), 2, n_iter);
V_Xe_reg = zeros(length(t_spline), 1, n_iter);

% Initializations
z_act = z_0;
phi_k_act = phi_k_reg;
Phi_hat_x_1_act = [Phi_hat_x_1, Phi_hat_x_1, Phi_hat_x_1];
Phi_hat_x_2_act = [Phi_hat_x_2, Phi_hat_x_2, Phi_hat_x_2];

% Measurement parameters
a = 5;
b = 0.2;
c = 0.01; %0.05 %Amplitud del ruido en la medición: Importante porque la estimación
% del siguiente PDF depende fuertemente de la amplitud de medición, si el
% ruido es muy grande el estimador puede fallar.
n_points = length(t_spline);

% Parameters for PDF Estimator
% Par_PDF.mu_1 = mu_1;
% Par_PDF.sigma_1 = sigma_1;
% Par_PDF.mu_2 = mu_2;
% Par_PDF.sigma_2 = sigma_2;
%Par_PDF.N = N;
Par_PDF.x_1 = x_1;
Par_PDF.x_2 = x_2;
Par_PDF.Meas_mean = a;
Par_PDF.n_def = n_def;

Phi_hat_x_1_next = zeros(length(x_1), n_def);
Phi_hat_x_2_next = zeros(length(x_2), n_def);
Phi_hat_x_j = zeros(length(x_1)*length(x_2), n_def);

for i = 1:n_iter

    [Z, U] = M(z_act, phi_k_act); %Soluciones
    Z = full(Z)';
    U = full(U)';

    % Trajectory X_e(t) = [x_1, x_2]
    X_e = [Z(:, 1), Z(:, 3)];
    X_e_dot = [Z(:, 2), Z(:, 4)];

    % Spline: Adding points to the trajectory to get more data from the sensor
    % and pass it to the estimator
    x_1e_spline = spline(t, X_e(:,1), t_spline);
    x_2e_spline = spline(t, X_e(:,2), t_spline);
    X_e_spline = [x_1e_spline, x_2e_spline];

    % Measurement model V
    %Upsilon = a + b*pdf(gm_dist, X_e_spline); %Real PDF
    Upsilon_1 = w_pdf*mvnpdf(X_e_spline, mu_1, sigma_1);
    Upsilon_2 = w_pdf*mvnpdf(X_e_spline, mu_2, sigma_2);
    Upsilon_3 = w_pdf*mvnpdf(X_e_spline, mu_3, sigma_3);

    Upsilon = b*[Upsilon_1, Upsilon_2, Upsilon_3];
    h_Theta = sum(Upsilon, 2);

    delta = c*randn(n_points, 1); %Gaussian Noise with Variance c^2
    
    V_Xe_i = a + Upsilon + delta;
    V_Xe = a + h_Theta + delta;

    % Registers
    z_reg(:,:,i) = Z;
    u_reg(:,:,i) = U;
    X_e_reg(:,:,i) = X_e;
    X_e_dot_reg(:,:,i) = X_e_dot;
    phi_k_REG(:,:,i) = phi_k_reg;
    Phi_hat_x_reg(:,:,i) = Phi_hat_x;
    X_e_spline_reg(:,:,i) = X_e_spline;
    V_Xe_reg(:,:,i) = V_Xe;

    % PDF Estimation
    for j = 1:n_def
        [Phi_hat_x_1_next(:,j), Phi_hat_x_2_next(:,j)] = PDF_Estimator( ...
            Phi_hat_x_1_act(:,j), Phi_hat_x_2_act(:,j), X_e_spline, V_Xe_i(:,j), Par_PDF );
    
        [Phi_hat_x_1_next_grid, Phi_hat_x_2_next_grid] = meshgrid( ...
            Phi_hat_x_1_next(:,j), Phi_hat_x_2_next(:,j) );

        Phi_hat_x_j(:,j) = prod([reshape(Phi_hat_x_1_next_grid,[],1), reshape(Phi_hat_x_2_next_grid,[],1)], 2);
    end

    Phi_hat_x = (1/n_def)*sum(Phi_hat_x_j, 2);

    % Compute new Fourier coefficients for \hat{Phi}(x)
    [phi_k_reg, ~, ~] = FourierCoef_RefPDF(Phi_hat_x, Par_struct);

    % Update parameters for next iteration
    z_act = Z(end,:)';           % Initial condition for state
    phi_k_act = phi_k_reg;      % New target coefficients
    Phi_hat_x_1_act = Phi_hat_x_1_next;
    Phi_hat_x_2_act = Phi_hat_x_2_next;

end

%plot(X_e_reg(:,1,1), X_e_reg(:,2,1), '*'); xlim([0.5, 1.5]);ylim([0.5, 1.5]); axis equal

%% Testing

% P_V_k_x1 = zeros(length(x_1), n_def);
% P_V_k_x2 = zeros(length(x_2), n_def);
% Phi_hat_x_1_next_i = zeros(length(x_1), n_def);
% Phi_hat_x_2_next_i = zeros(length(x_2), n_def);
% 
% % Previous PDF
% P_theta_x1 = Phi_hat_x_1;
% P_theta_x2 = Phi_hat_x_2;
% 
% % Likelihood function for Theta given measurements (functions projected on x_1 and x_2)
% for i = 1:n_def
%     Meas = V_Xe_i(:,i) - a;
%     Meas = Meas - min(Meas);
% 
%     exp_x1 = sum( Meas.*X_e_spline(:,1) ) / sum(Meas);
%     var_x1 = sum( Meas.*(X_e_spline(:,1).^2) ) / sum(Meas) - exp_x1^2;
%     P_V_k_x1(:,i) = (1/(sqrt(2*pi*var_x1)))*exp( -(x_1 - exp_x1).^2/(2*var_x1));
% 
%     exp_x2 = sum( Meas.*X_e_spline(:,2)) / sum(Meas);
%     var_x2 = sum( Meas.*(X_e_spline(:,2).^2)) / sum(Meas) - exp_x2^2;
%     P_V_k_x2(:,i) = (1/(sqrt(2*pi*var_x2)))*exp( -(x_2 - exp_x2).^2/(2*var_x2));
% 
%     % Next PDF
%     P_theta_x1_next_i = P_V_k_x1(:,i).*P_theta_x1;
%     exp_x1_next_i = sum( P_theta_x1_next_i.*x_1) / sum(P_theta_x1_next_i);
%     var_x1_next_i = sum( P_theta_x1_next_i.*(x_1.^2)) / sum(P_theta_x1_next_i) - exp_x1_next_i^2;
%     Phi_hat_x_1_next_i(:,i) = normpdf(x_1, exp_x1_next_i, sqrt(var_x1_next_i));
% 
%     P_theta_x2_next_i = P_V_k_x2(:,i).*P_theta_x2;
%     exp_x2_next_i = sum( P_theta_x2_next_i.*x_2) / sum(P_theta_x2_next_i);
%     var_x2_next_i = sum( P_theta_x2_next_i.*(x_2.^2)) / sum(P_theta_x2_next_i) - exp_x2_next_i^2;
%     Phi_hat_x_2_next_i(:,i) = normpdf(x_2, exp_x2_next_i, sqrt(var_x2_next_i));
% end
% 
% Phi_hat_x_1_next = (1/n_def)*sum(Phi_hat_x_1_next_i, 2);
% Phi_hat_x_2_next = (1/n_def)*sum(Phi_hat_x_2_next_i, 2);
% 
% [Phi_hat_x_1_next_grid, Phi_hat_x_2_next_grid] = meshgrid(Phi_hat_x_1_next, Phi_hat_x_2_next);
% temp = prod([reshape(Phi_hat_x_1_next_grid,[],1), reshape(Phi_hat_x_2_next_grid,[],1)], 2);
% 
% figure(6)
% plot(x_1, P_V_k_x1)
% 
% figure(7)
% plot(x_1, Phi_hat_x_1_next_i)
% hold on
% plot(x_1, sum(Phi_hat_x_1_next_i,2))
% hold off
% 
% figure(8)
% contour(x_1_grid, x_2_grid, reshape(temp, length(x_2), length(x_1)))
% xlim([L_1_l, L_1_u])
% ylim([L_2_l, L_2_u])
% 
% figure(9)
% surf(x_1_grid, x_2_grid, reshape(temp, length(x_2), length(x_1)))
% xlim([L_1_l, L_1_u])
% ylim([L_2_l, L_2_u])

%% Reconstrucción de distribución empírica y métrica ergódica

Varepsilon_reg = zeros(length(t), 1, n_iter);
C_x_reg = zeros(height(Omega), length(t), n_iter);

for r = 1:n_iter
    c_k = zeros(K^n, 1);
    C_x = zeros(height(Omega), 1);
    for i = 1:length(t)

        % Compute Fourier Functions and coefficients on the new position
        f_k_traj = prod(cos( K_cal'.*pi.*(X_e_reg(i,:,r) - L_i_l)./(L_i_u - L_i_l) ), 2) ./ h_k_reg ;
        c_k = c_k + (f_k_traj*T_s)/(t_f) ; %i*T_s %t_f

        %Ergodic metric
        Varepsilon = sum( Lambda_k .* (c_k - phi_k_REG(:,:,r)).^2 );

        %Empirical distribution reconstruction
        C_x_i = zeros(height(Omega), 1);
        for j = 1:K^n
            C_x_i = C_x_i + c_k(j)*f_k_reg(:,j);
        end

        %Se suman todas las distribuciones generadas en cada muestra
        C_x = C_x + C_x_i;

        %Se registra
        C_x_reg(:,i,r) = C_x;
        Varepsilon_reg(i,:,r) = Varepsilon;

    end

end



%% Gráficas

t_total = zeros(n_iter*(length(t)-1) + 1, 1);
X_e_total = zeros(n_iter*(length(t)-1) + 1, 2);
X_e_dot_total = zeros(n_iter*(length(t)-1) + 1, 2);
Varepsilon_total = zeros(n_iter*(length(t)-1) + 1, 1);
u_total = zeros(n_iter*(length(t)-1) + 1, 2);

t_spline_total = zeros(n_iter*(length(t_spline)-1) + 1, 1);
X_e_spline_total = zeros(n_iter*(length(t_spline)-1) + 1, 2);
V_Xe_total = zeros(n_iter*(length(t_spline)-1) + 1, 1);

for i = 1:n_iter

    id_init = ((i - 1)*(length(t)-1) + 1); %1,101,..
    id_last = (i*(length(t)-1) + 1);        %101, 201,...

    t_total( id_init:id_last ) = (i - 1)*t(end) + t;
    X_e_total( id_init:id_last, : ) = X_e_reg(:,:,i);
    X_e_dot_total( id_init:id_last, : ) = X_e_dot_reg(:,:,i);
    Varepsilon_total( id_init:id_last, : ) = Varepsilon_reg(:,:,i);
    u_total( id_init:id_last, : ) = [u_reg(:,:,i); [NaN, NaN]];
    
    id_init_spline = ((i - 1)*(length(t_spline)-1) + 1);
    id_last_spline = (i*(length(t_spline)-1) + 1);
    t_spline_total( id_init_spline:id_last_spline ) = (i - 1)*t_spline(end) + t_spline;
    X_e_spline_total( id_init_spline:id_last_spline, : ) = X_e_spline_reg(:,:,i);
    V_Xe_total( id_init_spline:id_last_spline, : ) = V_Xe_reg(:,:,i);

end

figure(1)
subplot(3,1,1)
plot(t_total, X_e_total, 'LineWidth', 1.5)
title("Position States",'Interpreter','latex')
xlabel('Time [s]','Interpreter','latex')
ylabel('Position [m]','Interpreter','latex')
legend('$x_1$', '$x_2$','Interpreter','latex')
grid on
subplot(3,1,2)
plot(t_total, X_e_dot_total, 'LineWidth', 1.5)
title("Velocity States",'Interpreter','latex')
xlabel('Time [s]','Interpreter','latex')
ylabel('Velocity [m/s]','Interpreter','latex')
legend('$\dot{x}_1$', '$\dot{x}_2$','Interpreter','latex')
grid on
subplot(3,1,3)
plot(t_total, u_total, 'LineWidth', 1.5) %stairs
title("Control actions",'Interpreter','latex')
xlabel('Time [s]','Interpreter','latex')
ylabel('Force [N]','Interpreter','latex')
legend('$u_1$', '$u_2$','Interpreter','latex')
grid on

figure(2)
plot(t_total, Varepsilon_total, "LineWidth",1.5)
title("Ergodic Metric",'Interpreter','latex')
xlabel('Time [s]','Interpreter','latex')
ylabel('$\varepsilon \left( \mathbf{X_e}(t), \Phi(\mathbf{x}) \right) $','Interpreter','latex')
grid on

figure(3)
subplot(3,3,1)
contour(x_1_grid, x_2_grid, reshape(Phi_x, length(x_2), length(x_1)))
xlim([L_1_l, L_1_u])
ylim([L_2_l, L_2_u])
title("Real PDF",'Interpreter','latex')
xlabel('$x_1$ [m]','Interpreter','latex')
ylabel('$x_2$ [m]','Interpreter','latex')
axis equal
grid on
legend('$\Phi(\mathbf{x})$','Interpreter','latex','Location','northeastoutside')
for i = 1:n_iter
    subplot(3,3,i+1)
    contour(x_1_grid, x_2_grid, reshape(Phi_hat_x_reg(:,:,i), length(x_2), length(x_1)))
    xlim([L_1_l, L_1_u])
    ylim([L_2_l, L_2_u])
    title("Estimated PDF, iteration " + i,'Interpreter','latex')
    xlabel('$x_1$ [m]','Interpreter','latex')
    ylabel('$x_2$ [m]','Interpreter','latex')
    axis equal
    grid on
    hold on
    plot(X_e_reg(:,1,i), X_e_reg(:,2,i),'LineWidth',1.2)
    plot(X_e_reg(1,1,i), X_e_reg(1,2,i),'ksq','MarkerSize',7,'LineWidth',1.5)
    legend('$\hat{\Phi}(\mathbf{x})$','$\mathbf{X_e}(t)$', '$\mathbf{X_e}(0)$',...
        'Interpreter','latex','Location','northeastoutside')
    hold off
end

%% Gráficas con adición de puntos Spline

figure(4)
plot(t_spline_total, X_e_spline_total, 'LineWidth', 1.5)
title("Position States",'Interpreter','latex')
xlabel('Time [s]','Interpreter','latex')
ylabel('Position [m]','Interpreter','latex')
legend('$x_1$', '$x_2$','Interpreter','latex')
grid on

figure(5)
subplot(3,1,1)
plot(t_spline_total, V_Xe_total, 'LineWidth', 1.5)
title("Sensor measurement on time",'Interpreter','latex')
xlabel('Time [s]','Interpreter','latex')
ylabel('Force [N]','Interpreter','latex')
legend('$V(t)$','Interpreter','latex')
grid on

for i = 1:n_iter
    
    subplot(3,1,2)
    hold on
    plot(X_e_spline_reg(:,1,i), V_Xe_reg(:,:,i), '.', 'MarkerSize', 12, 'DisplayName', "$V(x_1)$, iteration " + i)
    title("Sensor measurement on spatial domain",'Interpreter','latex')
    xlabel('$x_1$ [m]','Interpreter','latex')
    ylabel('Force [N]','Interpreter','latex')
    grid on
    hold off
    legend('Interpreter','latex')
    subplot(3,1,3)
    hold on
    plot(X_e_spline_reg(:,2,i), V_Xe_reg(:,:,i), '.', 'MarkerSize', 12, 'DisplayName', "$V(x_2)$, iteration " + i)
    title("Sensor measurement on spatial domain",'Interpreter','latex')
    xlabel('$x_2$ [m]','Interpreter','latex')
    ylabel('Force [N]','Interpreter','latex')
    grid on
    hold off
    legend('Interpreter','latex')
    
end


% figure(6)
% subplot(3,3,1)
% contour(x_1_grid, x_2_grid, reshape(Phi_x, length(x_2), length(x_1)))
% xlim([L_1_l, L_1_u])
% ylim([L_2_l, L_2_u])
% title("Real PDF",'Interpreter','latex')
% xlabel('$x_1$ [m]','Interpreter','latex')
% ylabel('$x_2$ [m]','Interpreter','latex')
% axis equal
% grid on
% legend('$\Phi(\mathbf{x})$','Interpreter','latex','Location','northeastoutside')
% for i = 1:n_iter
%     subplot(3,3,i+1)
%     contour(x_1_grid, x_2_grid, reshape(Phi_hat_x_reg(:,:,i), length(x_2), length(x_1)))
%     xlim([L_1_l, L_1_u])
%     ylim([L_2_l, L_2_u])
%     title("Estimated PDF, iteration " + i,'Interpreter','latex')
%     xlabel('$x_1$ [m]','Interpreter','latex')
%     ylabel('$x_2$ [m]','Interpreter','latex')
%     axis equal
%     grid on
%     hold on
%     plot(X_e_spline_reg(:,1,i), X_e_spline_reg(:,2,i),'LineWidth',1.2)
%     plot(X_e_spline_reg(1,1,i), X_e_spline_reg(1,2,i),'ksq','MarkerSize',7,'LineWidth',1.5)
%     legend('$\hat{\Phi}(\mathbf{x})$','$\mathbf{X_e}(t)$', '$\mathbf{X_e}(0)$',...
%         'Interpreter','latex','Location','northeastoutside')
%     hold off
% end

%% Detailed charts about every iteration

% for i = 1:n_iter
% 
%     figure(i+10)
%     subplot(2,3,1);
%     contour(x_1_grid, x_2_grid, reshape(Phi_x, length(x_2), length(x_1)))
%     xlim([L_1_l, L_1_u])
%     ylim([L_2_l, L_2_u])
%     title("Real PDF",'Interpreter','latex')
%     xlabel('$x_1$ [m]','Interpreter','latex')
%     ylabel('$x_2$ [m]','Interpreter','latex')
%     axis equal
%     grid on
%     legend('$\Phi(\mathbf{x})$','Interpreter','latex','Location','northeastoutside')
%     subplot(2,3,2);
%     contour(x_1_grid, x_2_grid, reshape(Phi_hat_x_reg(:,:,i), length(x_2), length(x_1)))
%     xlim([L_1_l, L_1_u])
%     ylim([L_2_l, L_2_u])
%     title("Estimated PDF, iteration " + i,'Interpreter','latex')
%     xlabel('$x_1$ [m]','Interpreter','latex')
%     ylabel('$x_2$ [m]','Interpreter','latex')
%     axis equal
%     grid on
%     hold on
%     plot(X_e_reg(:,1,i), X_e_reg(:,2,i),'LineWidth',1.2)
%     plot(X_e_reg(1,1,i), X_e_reg(1,2,i),'ksq','MarkerSize',7,'LineWidth',1.5)
%     legend('$\hat{\Phi}(\mathbf{x})$','$\mathbf{X_e}(t)$', '$\mathbf{X_e}(0)$',...
%         'Interpreter','latex','Location','northeastoutside')
%     hold off
%     subplot(2,3,3);
%     contour(x_1_grid, x_2_grid, reshape(C_x_reg(:,end,i), length(x_2), length(x_1)))
%     xlim([L_1_l, L_1_u])
%     ylim([L_2_l, L_2_u])
%     title("Empirical distribution reconstruction",'Interpreter','latex')
%     xlabel('$x_1$ [m]','Interpreter','latex')
%     ylabel('$x_2$ [m]','Interpreter','latex')
%     axis equal
%     grid on
%     hold on
%     plot(X_e_reg(:,1,i), X_e_reg(:,2,i),'LineWidth',1.2)
%     plot(X_e_reg(1,1,i), X_e_reg(1,2,i),'ksq','MarkerSize',7,'LineWidth',1.5)
%     legend('$C(\mathbf{x})$','$\mathbf{X_e}(t)$', '$\mathbf{X_e}(0)$',...
%         'Interpreter','latex','Location','northeastoutside')
%     hold off
%     subplot(2,3,4);
%     surf(x_1_grid, x_2_grid, reshape(Phi_x, length(x_2), length(x_1)))
%     xlim([L_1_l, L_1_u])
%     ylim([L_2_l, L_2_u])
%     %title("Real PDF",'Interpreter','latex')
%     xlabel('$x_1$','Interpreter','latex')
%     ylabel('$x_2$','Interpreter','latex')
%     zlabel('$\Phi(\mathbf{x})$','Interpreter','latex')
%     grid on
%     subplot(2,3,5);
%     surf(x_1_grid, x_2_grid, reshape(Phi_hat_x_reg(:,:,i), length(x_2), length(x_1)))
%     xlim([L_1_l, L_1_u])
%     ylim([L_2_l, L_2_u])
%     %title("Reference PDF",'Interpreter','latex')
%     xlabel('$x_1$','Interpreter','latex')
%     ylabel('$x_2$','Interpreter','latex')
%     zlabel('$\hat{\Phi}(\mathbf{x})$','Interpreter','latex')
%     grid on
%     subplot(2,3,6);
%     surf(x_1_grid, x_2_grid, reshape(C_x_reg(:,end,i), length(x_2), length(x_1)))
%     xlim([L_1_l, L_1_u])
%     ylim([L_2_l, L_2_u])
%     %title("Empirical distribution reconstruction",'Interpreter','latex')
%     xlabel('$x_1$','Interpreter','latex')
%     ylabel('$x_2$','Interpreter','latex')
%     zlabel('$C(\mathbf{x})$','Interpreter','latex')
%     grid on
% 
% end


%% MPC

% % Registers
% f_k_traj_reg = zeros(K^n, length(t));
% c_k_reg = zeros(K^n, length(t));
% Varepsilon_reg = zeros(length(t), 1);
% Z_reg = zeros(length(t), size(z,1));
% U_reg = zeros(length(t), size(u,1));
% 
% % Initializations
% z_act = z_0;
% c_k_act = c_k_0;
% f_k_traj_act = f_k_traj_0;
% Varepsilon_act = Varepsilon_0;
% 
% for i = 1:length(t)
% 
%     % Optimal control input computation given actual state z
%     u = full( M(z_act) );
% 
%     % Saving in registers
%     f_k_traj_reg(:,i) = f_k_traj_act;
%     c_k_reg(:,i) = c_k_act;
%     Varepsilon_reg(i) =  Varepsilon_act;
%     Z_reg(i,:) = z_act;
%     U_reg(i,:) = u;
% 
%     % Apply optimal control u and simulate system
%     z_act = full( F(z_act,u) );
% 
%     % Compute Fourier Functions and coefficients on the new position
%     X_e = [z_act(1), z_act(3)];   % Position [x_1, x_2]
%     f_k_traj_act = prod(cos( K_cal'.*pi.*(X_e - L_i_l)./(L_i_u - L_i_l) ), 2) ./ h_k_reg ;
%     c_k_act = c_k_act + (f_k_traj_act*T_s)/t_f ;
% 
%     %Ergodic metric
%     Varepsilon_act = sum( Lambda_k .* (c_k_act - phi_k_reg).^2 ); 
% 
% end
% 
% %% Charts MPC results
% 
% figure(3)
% subplot(1,2,1)
% plot(t, Z_reg, 'LineWidth', 1.5)
% title("States",'Interpreter','latex')
% xlabel('Time [s]','Interpreter','latex')
% ylabel('Position [m], velocity[m/s]','Interpreter','latex')
% legend('$x_1$', '$\dot{x}_1$', '$x_2$', '$\dot{x}_2$','Interpreter','latex')
% grid on
% subplot(1,2,2)
% stairs(t, U_reg, 'LineWidth', 1.5)
% title("Control actions",'Interpreter','latex')
% xlabel('Time [s]','Interpreter','latex')
% ylabel('Force [N]','Interpreter','latex')
% legend('$u_1$', '$u_2$','Interpreter','latex')
% grid on
% 
% figure(4)
% plot(t, Varepsilon_reg, "LineWidth",1.5)
% title("Ergodic Metric",'Interpreter','latex')
% xlabel('Time [s]','Interpreter','latex')
% ylabel('$\varepsilon \left( \mathbf{X_e}(t), \Phi(\mathbf{x}) \right) $','Interpreter','latex')
% grid on