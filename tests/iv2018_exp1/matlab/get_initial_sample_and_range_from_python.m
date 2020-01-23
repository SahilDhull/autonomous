function [init_sample, init_cond] = get_initial_sample_and_range_from_python(exp_ind, exp_file_name, critical_param_list)
%get_initial_sample_and_range_from_python Reads the initial sample used in
%the given experiment file, with the range of search.

% Import Python modules:
insert(py.sys.path,int32(0),'../');
test = py.importlib.import_module('iv2018_exp1');
py.importlib.reload(test);

% Read initial sample and range:
ret_val = py.iv2018_exp1.get_initial_sample_and_range(exp_ind, exp_file_name, py.list(uint32(critical_param_list)));
init_sample = double(py.array.array('d',py.numpy.nditer(ret_val(1))));
init_cond = Core_py2matlab(ret_val(2)); % Converts array from python to matlab
end
