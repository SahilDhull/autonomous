function [T, XT, YT, LT, CLG, GRD] = get_saved_webots_traj(traj_file_name)
%GET_SAVED_WEBOTS_TRAJ Loads Python pickled trajectory file and converts
%into Matlab data format.

% Import Python modules:
insert(py.sys.path,int32(0),'../../../Sim_ATAV/simulation_configurator');
traj_tools = py.importlib.import_module('trajectory_tools');
py.importlib.reload(traj_tools);

% Read Trajectory:
traj = py.trajectory_tools.load_selected_trajectory_for_matlab(traj_file_name);

% Convert Trajectory to matlab array:
mattraj = Core_py2matlab(traj);
YT = [];
LT = [];
CLG = [];
GRD = [];
T = mattraj(:,1)/1000.0;
XT = mattraj(:, 2:end);
end

