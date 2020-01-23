clear all
close all
clc
addpath(genpath('../../../s-taliro'))
load_environment_config_scenario_1;
run_ca_tests_scenario_1

clear all
close all
clc
addpath(genpath('../../../s-taliro'))
load_environment_config_scenario_1;
evaluated_set = 'CA';
staliro_webots_trajectory_robustness_computation_scenario_1_r1;
run_falsification_after_ca_scenario_1_r1
save 'results_after_ca_scenario_1_r1.mat'
copyfile '../RESULTS_1/*' '../RESULTS_1_R1'

clear all
close all
clc
addpath(genpath('../../../s-taliro'))
load_environment_config_scenario_1;
evaluated_set = 'CA';
staliro_webots_trajectory_robustness_computation_scenario_1_r2;
run_falsification_after_ca_scenario_1_r2
save 'results_after_ca_scenario_1_r2.mat'
copyfile '../RESULTS_1/*' '../RESULTS_1_R2'

clear all
close all
clc
addpath(genpath('../../../s-taliro'))
load_environment_config_scenario_1;
evaluated_set = 'CA';
staliro_webots_trajectory_robustness_computation_scenario_1_r4;
run_falsification_after_ca_scenario_1_r4
save 'results_after_ca_scenario_1_r4.mat'
copyfile '../RESULTS_1/*' '../RESULTS_1_R4'
