% Import Python modules:
insert(py.sys.path,int32(0),'../');
insert(py.sys.path,int32(0),'../../../');
test = py.importlib.import_module('ivjournal_test_1');
py.importlib.reload(test);

% Run the simulation and receive the trajectory:
traj = py.ivjournal_test_1.run_ca_tests(py.None, py.list([2, 3]));
