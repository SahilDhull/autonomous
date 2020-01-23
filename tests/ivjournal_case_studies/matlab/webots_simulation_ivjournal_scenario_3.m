function [T, XT, YT, LT, CLG, GRD] = webots_simulation_ivjournal_scenario_3(XPoint, staliro_SimulationTime, steptime, InpSignal)
%webots_simulation_ivjournal_test_3 Set the values of the critical
%params, run simulation and receive the trajectory.

global currentExpIndex
global FALSIF_Run_Number
global staliro_run_count
global Falsif_Test_Type

% Import Python modules:
insert(py.sys.path,int32(0),'../');
insert(py.sys.path,int32(0),'../../../');
test = py.importlib.import_module('ivjournal_scenario_3');
py.importlib.reload(test);

disp(XPoint')

% Run the simulation and receive the trajectory:
traj = py.ivjournal_scenario_3.run_falsification(currentExpIndex, py.list(XPoint'), FALSIF_Run_Number, staliro_run_count, int32(staliro_SimulationTime*1000.0), Falsif_Test_Type);

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
end
end

