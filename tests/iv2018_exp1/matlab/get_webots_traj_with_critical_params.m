function [T, XT, YT, LT, CLG, GRD] = get_webots_traj_with_critical_params(XPoint, staliro_SimulationTime, steptime, InpSignal)
%GET_WEBOTS_TRAJ_WITH_CRITICAL_PARAMS Set the values of the critical
%params, run simulation and receive the trajectory.

global currentExpIndex
global criticalParamList
global FALSIF_EXP_RESULTS_FILE_NAME
global FALSIF_Run_Number
global staliro_run_count
global Falsif_Test_Type
global FALSIF_EXP_FILE_NAME

% Import Python modules:
insert(py.sys.path,int32(0),'../');
test = py.importlib.import_module('iv2018_exp1');
py.importlib.reload(test);

% Run the simulation and receive the trajectory:
traj = py.iv2018_exp1.run_selected_test_critical_params(currentExpIndex, FALSIF_Run_Number, staliro_run_count, py.list(uint32(criticalParamList)), py.list(XPoint'), int32(staliro_SimulationTime*1000.0), FALSIF_EXP_FILE_NAME, FALSIF_EXP_RESULTS_FILE_NAME, Falsif_Test_Type);

% Convert trajectory to matlab array
mattraj = Core_py2matlab(traj);
YT = [];
LT = [];
CLG = [];
GRD = [];
if isempty(mattraj)
    T = [];
    XT = [];
else
    T = mattraj(:,1)/1000.0;
    XT = mattraj(:, 2:end);
    XT(:, 44:55) = 1000.0*XT(:, 44:55); % We multiply values by 1000 for normalization in robustness computation.
end
end

