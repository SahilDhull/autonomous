global staliro_run_count
global FALSIF_Run_Number
global falsification_n_tests
global currentExpIndex
global Falsif_Test_Type

FALSIF_Run_Number = 1;
staliro_run_count = 1;
falsification_n_tests = 700;
currentExpIndex = 0;
Falsif_Test_Type = 'SA';

addpath(genpath('../../../s-taliro'))

%load_requirements_1;
load_requirements_2;
init_cond = get_parameter_ranges();
model = @webots_simulation_ivjournal_test_1;
input_range = [];
cp_array = [];
opt = staliro_options();
opt.runs = 1;
opt.black_box = 1;
opt.SampTime = 0.010;
opt.spec_space = 'X';
opt.optimization_solver = 'SA_Taliro';
opt.taliro = 'dp_taliro';
opt.search_space_constrained.constrained = true;
opt.map2line = 0;
opt.falsification = 1;
%opt.seed = 100;
opt.optim_params.n_tests = falsification_n_tests;

disp(' ')
disp(['Running S-TaLiRo ... '])
opt

sim_duration = 7.0

tic
[results, history] = staliro(model, init_cond, input_range, cp_array, phi, preds, sim_duration, opt);
runtime = toc;
runtime

res_filename = ['results_', datestr(datetime("now"), 'yyyy_mm_dd__HH_MM'), '.mat'];
save(res_filename)
disp(["Results are saved to: ", res_filename])



