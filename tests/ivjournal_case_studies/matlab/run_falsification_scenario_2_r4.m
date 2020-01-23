clear all
close all
clc

global staliro_run_count
global FALSIF_Run_Number
global falsification_n_tests
global currentExpIndex
global Falsif_Test_Type

FALSIF_Run_Number = 1;
staliro_run_count = 1;
falsification_n_tests = 1000;
currentExpIndex = 0;
Falsif_Test_Type = 'SA';

addpath(genpath('../../../s-taliro'))

load_requirement_R4_scenario_2;
init_cond = get_parameter_ranges_scenario_2();
model = @webots_simulation_ivjournal_scenario_2;
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

sim_duration = 15.0

tic
[results, history] = staliro(model, init_cond, input_range, cp_array, phi, preds, sim_duration, opt);
runtime = toc;
runtime

res_filename = ['results_scenario_2_r4_', datestr(datetime("now"), 'yyyy_mm_dd__HH_MM'), '.mat'];
save(res_filename)
disp(["Results are saved to: ", res_filename])



