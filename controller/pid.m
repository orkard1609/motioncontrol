% Parameters
M = 1.0;  % cart mass (kg)
m = 0.1;  % pendulum mass (kg)
l = 0.5;  % pendulum length to center of mass (m)
g = 9.81; % gravity (m/s^2)
dt = 0.02; % time step (s)
sim_time = 10.0; % simulation time (s)
num_steps = round(sim_time / dt);

% PID Controller (from scratch)
kp = 50.0; ki = 1.0; kd = 10.0; setpoint = 0.0;
integral = 0.0;
prev_error = 0.0;

% Simulation
state = [0.0; 0.0; 0.1; 0.0]; % initial: x=0, dx=0, theta=0.1 rad, dtheta=0
states = zeros(4, num_steps);
states(:,1) = state;

for i = 2:num_steps
    theta = state(3);
    error = setpoint - theta;
    integral = integral + error * dt;
    derivative = (error - prev_error) / dt;
    u = kp * error + ki * integral + kd * derivative; % control force
    prev_error = error;
    
    dot_state = inverted_pendulum_dynamics(state, u, M, m, l, g);
    state = state + dot_state * dt; % Euler integration
    states(:,i) = state;
end

% Plot results
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

% Nonlinear dynamics function
function dot_state = inverted_pendulum_dynamics(state, u, M, m, l, g)
    x = state(1); dx = state(2); theta = state(3); dtheta = state(4);
    sin_theta = sin(theta);
    cos_theta = cos(theta);
    denom = M + m * sin_theta^2;
    ddx = (u + m * l * (dtheta^2 * sin_theta - g * sin_theta * cos_theta)) / denom;
    ddtheta = (g * sin_theta - cos_theta * ddx) / l;
    dot_state = [dx; ddx; dtheta; ddtheta];
end