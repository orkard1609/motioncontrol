% Parameters
M = 1.0;       % cart mass
mp = 0.1;      % pendulum mass
l = 0.5;       % pendulum length
g = 9.81;      % gravity
dt = 0.02;
sim_time = 10.0;
num_steps = round(sim_time / dt);

% Linearized and discretized
A = [0, 1, 0, 0;
     0, 0, -(mp*g)/M, 0;
     0, 0, 0, 1;
     0, 0, (M+mp)*g/(M*l), 0];
B = [0; 1/M; 0; -1/(M*l)];
Ad = eye(4) + A * dt;
Bd = B * dt;

% MPC parameters
N = 20; % prediction horizon
Q = diag([1, 1, 10, 1]); % state cost
R = 0.01; % input cost

% Build MPC matrices
n = size(Ad,1);     % state dimension
nu = size(Bd,2);    % input dimension

% Phi = [Ad; Ad^2; ... Ad^N]
Phi = zeros(N*n, n);
current_A = Ad;
for i = 1:N
    Phi((i-1)*n+1:i*n, :) = current_A;
    current_A = current_A * Ad;
end

% Gamma (toeplitz block)
Gamma = zeros(N*n, N*nu);
for i = 1:N
    for j = 1:i
        if i - j == 0
            Gamma((i-1)*n+1:i*n, (j-1)*nu+1:j*nu) = Bd;
        else
            Gamma((i-1)*n+1:i*n, (j-1)*nu+1:j*nu) = Ad * Gamma((i-2)*n+1:(i-1)*n, (j-1)*nu+1:j*nu);
        end
    end
end

% Block diagonal Qb, Rb
Qb = kron(eye(N), Q);
Rb = kron(eye(N), R);

% H = Gamma^T Qb Gamma + Rb
H = Gamma' * Qb * Gamma + Rb;

% Simulation
state = zeros(4, 1);  
state(1) = 0.0;  % x
state(2) = 0.0;  % dx
state(3) = 0.1;  % theta
state(4) = 0.0;  % dtheta

states = zeros(4, num_steps);
states(:,1) = state;

for i = 2:num_steps
    % Scalars
    x = double(state(1));
    dx = double(state(2));
    theta = double(state(3));
    dtheta = double(state(4));
    
    % MPC control
    state_col = reshape(state, 4, 1);
    gvec = Gamma' * Qb * Phi * state_col;
    U_opt = -H \ gvec;
    u = double(U_opt(1));
    
    % Dynamics
    sin_theta = sin(theta);
    cos_theta = cos(theta);
    denom = M + mp * sin_theta^2;
    
    ddx = (u + mp * l * (dtheta^2 * sin_theta - g * sin_theta * cos_theta)) / denom;
    ddtheta = (g * sin_theta - cos_theta * ddx) / l;
    
    % Euler integration
    x_new = x + dx * dt;
    dx_new = dx + ddx * dt;
    theta_new = theta + dtheta * dt;
    dtheta_new = dtheta + ddtheta * dt;
    
    % Store
    states(1,i) = x_new;
    states(2,i) = dx_new;
    states(3,i) = theta_new;
    states(4,i) = dtheta_new;
    
    % Update state
    state(1) = x_new;
    state(2) = dx_new;
    state(3) = theta_new;
    state(4) = dtheta_new;
end

% Plot
time = 0:dt:sim_time-dt;
figure('Position', [100 100 800 600]);
subplot(2,1,1);
plot(time, states(1,:), 'DisplayName', 'Cart Position (x)');
hold on;
plot(time, states(3,:), 'DisplayName', 'Pendulum Angle (theta)');
legend;
grid on;
subplot(2,1,2);
plot(time, states(2,:), 'DisplayName', 'Cart Velocity (dx)');
hold on;
plot(time, states(4,:), 'DisplayName', 'Angular Velocity (dtheta)');
legend;
grid on;
