% Parameters
M = 1.0;  % cart mass (kg)
m = 0.1;  % pendulum mass (kg)
l = 0.5;  % pendulum length to center of mass (m)
g = 9.81; % gravity (m/s^2)
dt = 0.02; % time step (s)
sim_time = 50.0; % simulation time (s)
num_steps = round(sim_time / dt); % num step/num of data points

% Simulation
% initial: 
% 1. Cart position: x=0
% 2. Cart velocity: dx=0
% 3. Pendulum angle: theta=0 rad
% 3. Pendulum velocity: dtheta=0
state = [0.0; 0.0; 0.0; 0.0]; 
states = zeros(4, num_steps); % init array of zeros with size of "state x num_steps"
states(:,1) = state; % assign first state into the frist column of the array

% PID Controller (from scratch)
kp = 15; ki = 30; kd = 5; setpoint = 1.0;
integral = 0.0;
prev_error = 0.0;

for i = 2:num_steps
    theta = state(3);
    error = setpoint - theta;
    integral = integral + error * dt;
    derivative = (error - prev_error) / dt;
    u = kp * error + ki * integral + kd * derivative; % control force
    prev_error = error;
    
    % Add new state into the array
    dot_state = inverted_pendulum_dynamics(state, u, M, m, l, g);
    state = state + dot_state * dt; % Euler integration
    states(:,i) = state;
end

% Plot results
time = 0:dt:sim_time-dt;
figure('Position', [100 100 800 600]);

subplot(2,2,1);
plot(time, states(1,:), 'DisplayName', 'Cart Position (x)');
hold on;
legend;
grid on;

subplot(2,2,2);
plot(time, states(3,:), 'DisplayName', 'Pendulum Angle (theta)');
hold on;
legend;
grid on;

subplot(2,2,3);
plot(time, states(2,:), 'DisplayName', 'Cart Velocity (dx)');
hold on;
legend;
grid on;

subplot(2,2,4);
plot(time, states(4,:), 'DisplayName', 'Angular Velocity (dtheta)');
hold on;
legend;
grid on;

% Nonlinear dynamics function - Inverted pendulum system
function dot_state = inverted_pendulum_dynamics(state, u, M, m, l, g)
    x = state(1); dx = state(2); theta = state(3); dtheta = state(4);
    sin_theta = sin(theta);
    cos_theta = cos(theta);
    denom_cart = M + m * sin_theta^2;
    ddx = (u + m * l * dtheta^2 * sin_theta - m * g * sin_theta * cos_theta) / denom_cart;
    denom_pendulum = (M + m) * l - m * l * cos_theta^2;
    ddtheta = (u * cos_theta - (M + m) * g * sin_theta + m * l * (sin_theta * cos_theta ) * dtheta^2) / denom_pendulum;
    dot_state = [dx; ddx; dtheta; ddtheta];
end

function best_gains = tune_pid_controller()
    % Parameters
    M = 1.0; m = 0.1; l = 0.5; g = 9.81;
    dt = 0.02; sim_time = 10.0; % Shorter time for faster tuning
    setpoint = 1.0;
    
    % Parameter ranges to try
    kp_values = 20:10:60;
    ki_values = [0.1, 0.5, 1.0, 2.0, 5.0];
    kd_values = 5:5:25;
    
    best_performance = Inf;
    best_gains = [15, 0.15, 5]; % Default values
    
    % Progress tracking
    total_combinations = length(kp_values) * length(ki_values) * length(kd_values);
    count = 0;
    
    for kp = kp_values
        for ki = ki_values
            for kd = kd_values
                count = count + 1;
                fprintf('Testing combination %d/%d: kp=%.1f, ki=%.1f, kd=%.1f\n', ...
                       count, total_combinations, kp, ki, kd);
                
                % Run simulation with these gains
                [performance, metrics] = simulate_pendulum(kp, ki, kd, M, m, l, g, dt, sim_time, setpoint);
                
                % If better than previous best, update
                if performance < best_performance
                    best_performance = performance;
                    best_gains = [kp, ki, kd];
                    fprintf('NEW BEST! kp=%.1f, ki=%.1f, kd=%.1f, score=%.2f\n', ...
                           kp, ki, kd, performance);
                    fprintf('  Settling time: %.2fs, Overshoot: %.2f, SS error: %.4f\n', ...
                           metrics.settling_time, metrics.overshoot, metrics.steady_state_error);
                end
            end
        end
    end
end

function [score, metrics] = simulate_pendulum(kp, ki, kd, M, m, l, g, dt, sim_time, setpoint)
    % Simulation setup
    num_steps = round(sim_time / dt);
    state = [0.0; 0.0; 0.0; 0.0]; 
    states = zeros(4, num_steps);
    states(:,1) = state;
    
    % PID controller variables
    integral = 0.0;
    prev_error = 0.0;
    
    % Run simulation
    for i = 2:num_steps
        theta = state(3);
        error = setpoint - theta;
        integral = integral + error * dt;
        derivative = (error - prev_error) / dt;
        u = kp * error + ki * integral + kd * derivative;
        prev_error = error;
        
        % System dynamics
        dot_state = inverted_pendulum_dynamics(state, u, M, m, l, g);
        state = state + dot_state * dt;
        states(:,i) = state;
        
        % Early termination if system becomes unstable
        if abs(state(1)) > 10 || abs(state(2)) > 50 || abs(state(3)) > 3*pi
            score = Inf;
            metrics = struct('settling_time', Inf, 'overshoot', Inf, 'steady_state_error', Inf);
            return;
        end
    end
    
    % Calculate performance metrics
    theta_data = states(3,:);
    error_data = setpoint - theta_data;
    
    % Calculate settling time (time to reach and stay within 2% of setpoint)
    tolerance = 0.02 * setpoint;
    settled = false;
    settling_time = sim_time;
    
    for i = 100:num_steps  % Skip initial transient
        if all(abs(error_data(i:end)) < tolerance)
            settling_time = (i-1) * dt;
            settled = true;
            break;
        end
    end
    
    % Calculate overshoot
    if setpoint > 0
        overshoot = max(0, max(theta_data) - setpoint) / setpoint;
    else
        overshoot = max(0, setpoint - min(theta_data)) / abs(setpoint);
    end
    
    % Calculate steady-state error
    steady_state_error = abs(mean(error_data(max(1, end-50):end)));
    
    % Metrics structure
    metrics = struct('settling_time', settling_time, ...
                    'overshoot', overshoot, ...
                    'steady_state_error', steady_state_error);
    
    % Score calculation (weighted sum)
    if ~settled
        score = 1000;  % Heavy penalty for not settling
    else
        score = settling_time + 10*overshoot + 50*steady_state_error;
    end
end