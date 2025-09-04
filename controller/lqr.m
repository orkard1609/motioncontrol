% Parameters (same as above)
M = 1.0;
m = 0.1;
l = 0.5;
g = 9.81;
dt = 0.02;
sim_time = 10.0;
num_steps = round(sim_time / dt);

% Linearized matrices (A, B)
A = [0, 1, 0, 0;
     0, 0, -(m*g)/M, 0;
     0, 0, 0, 1;
     0, 0, (M+m)*g/(M*l), 0];
B = [0; 1/M; 0; -1/(M*l)];

% Discretize (approximate for small dt)
Ad = eye(4) + A * dt;
Bd = B * dt;

% LQR parameters
Q = diag([1, 1, 10, 1]); % state cost (emphasize theta)
R = 1; % input cost

% Solve discrete Riccati equation from scratch (iterative method)
P = zeros(4,4); % start with P=0
for iter = 1:1000 % iterate until convergence
    P_next = Q + Ad' * P * Ad - Ad' * P * Bd * inv(R + Bd' * P * Bd) * Bd' * P * Ad;
    if norm(P_next - P, 'fro') < 1e-6
        break;
    end
    P = P_next;
end

% LQR gain K = (R + B^T P B)^-1 B^T P A
K = inv(R + Bd' * P * Bd) * (Bd' * P * Ad);

% Simulation
state = [0.0; 0.0; 0.1; 0.0];
states = zeros(4, num_steps);
states(:,1) = state;

for i = 2:num_steps
    u = -K * state; % LQR control u = -K x
    
    dot_state = inverted_pendulum_dynamics(state, u, M, m, l, g);
    state = state + dot_state * dt;
    states(:,i) = state;
end

% Plot (same as above)
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

% Nonlinear dynamics (same as above)
function dot_state = inverted_pendulum_dynamics(state, u, M, m, l, g)
    x = state(1); dx = state(2); theta = state(3); dtheta = state(4);
    sin_theta = sin(theta);
    cos_theta = cos(theta);
    denom = M + m * sin_theta^2;
    ddx = (u + m * l * (dtheta^2 * sin_theta - g * sin_theta * cos_theta)) / denom;
    ddtheta = (g * sin_theta - cos_theta * ddx) / l;
    dot_state = [dx; ddx; dtheta; ddtheta];
end