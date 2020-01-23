function [T, XT, YT, LT, CLG, GRD] = run_tutorial_example_from_matlab(XPoint, sim_duration_s, steptime, InpSignal)
%run_tutorial_example_from_matlab Run Webots simulation with the parameters in XPoint.
% XPoint contains: [ego_init_speed_m_s, ego_x_pos, pedestrian_speed]

% Run the simulation and receive the trajectory:
traj = py.tutorial_example_1.run_test(XPoint(1), XPoint(2), XPoint(3), int32(sim_duration_s*1000.0), true);

% Convert trajectory to matlab array
mattraj = Core_py2matlab(traj);  % Core_py2matlab is from Matlab fileexchange, developed by Kyle Wayne Karhohs
YT = [];
LT = [];
CLG = [];
GRD = [];
if isempty(mattraj)
    T = [];
    XT = [];
else
    % Separate time from the simulation trace:
    T = mattraj(:,1)/1000.0;  % Also, convert time to s from ms
    XT = mattraj(:, 2:end);  % Rest of the trace
end
end

