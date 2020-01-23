clear all
close all
clc
addpath(genpath('../../../s-taliro'))
load_environment_config_scenario_1;
evaluated_set = 'CA';
staliro_webots_trajectory_robustness_computation_scenario_1_r4;
load_requirement_R4_scenario_1;
disp('=============================================')
disp('=========== Scenario 1 R4 ===================')
disp('=============================================')
run_falsification_after_ca_scenario_1;
save 'results_after_ca_scenario_1_r4.mat'
copyfile '../RESULTS_1/*' '../RESULTS_1_R4'
