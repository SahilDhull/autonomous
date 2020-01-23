function [init_sample] = get_initial_sample_from_python_scenario_1(exp_ind)
%get_initial_sample_from_python_scenario_1 Reads the initial sample used in
%the given experiment file, with the range of search.

% Import Python modules:
insert(py.sys.path,int32(0),'../');
test = py.importlib.import_module('ivjournal_test_1');
py.importlib.reload(test);

% Read initial sample and range:
ret_val = py.ivjournal_test_1.get_initial_sample(exp_ind);
init_sample = zeros(1, length(ret_val));
for i = 1:length(ret_val)
    init_sample(i) = double(py.array.array('d',py.numpy.nditer(ret_val(i))));
end
end
