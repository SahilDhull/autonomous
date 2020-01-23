%clear

% evaluated_set = 'CA';
% evaluated_set = 'Global_UR';
% evaluated_set = 'Worst_Falsification_SA';
% evaluated_set = 'Worst_Falsification_UR';
% evaluated_set = 'Zero_Falsification_SA';
% evaluated_set = 'Zero_Falsification_UR';

if ~exist('evaluated_set')
    error('evaluated_set should be set outside this script.\n Options: CA / Global_UR / Worst_Falsification_SA / Worst_Falsification_UR / Zero_Falsification_SA / Zero_Falsification_UR\n')
end

disp(' ')
disp('Computing robustness value on saved Webots simulation trajectories...')

time = 7;
disp(' ')
disp(['Total Simulation time:', num2str(time)]);

load_requirement_R4_scenario_1;

% Note that not all of the options below are used for robustness
% computation.
init_cond = get_parameter_ranges_scenario_1();
model = @webots_simulation_ivjournal_scenario_1;
model_type = determine_model_type(model);
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
opt.optim_params.n_tests = 700;


evaluated_experiment_index = [1];

for eval_exp_ind = evaluated_experiment_index
    if strcmpi(evaluated_set, 'CA')
        TotalNumberOfTests = 195;
        trajectory_file_prefix = CA_TRAJ_FILES_PREFIX;
    elseif strcmpi(evaluated_set, 'Global_UR')
        TotalNumberOfTests = 200;
        trajectory_file_prefix = [GLOBAL_UR_TRAJ_FILES_PREFIX, num2str(eval_exp_ind), '_trajectory_'];
    elseif strcmpi(evaluated_set, 'Worst_Falsification_SA')
        trajectory_file_prefix = [WORST_FALSIFICATION_SA_TRAJ_FILES_PREFIX, num2str(eval_exp_ind), '_trajectory_'];
        TotalNumberOfTests = 153;
    elseif strcmpi(evaluated_set, 'Worst_Falsification_UR')
        trajectory_file_prefix = [WORST_FALSIFICATION_UR_TRAJ_FILES_PREFIX, num2str(eval_exp_ind), '_trajectory_'];
        TotalNumberOfTests = 153;
    elseif strcmpi(evaluated_set, 'Zero_Falsification_SA')
        trajectory_file_prefix = [ZERO_FALSIFICATION_SA_TRAJ_FILES_PREFIX, num2str(eval_exp_ind), '_trajectory_'];
        TotalNumberOfTests = 153;
    elseif strcmpi(evaluated_set, 'Zero_Falsification_UR')
        trajectory_file_prefix = [ZERO_FALSIFICATION_UR_TRAJ_FILES_PREFIX, num2str(eval_exp_ind), '_trajectory_'];
        TotalNumberOfTests = 153;
    end

    
    robustness_list = zeros(1, TotalNumberOfTests);
    aux_cell_array = cell(0, 0);
    traj_cell_array = cell(0, 0);
    disp(' ')
    disp(['Working on files: ', trajectory_file_prefix, '*'])
    start_time = datetime('now');
    disp(start_time)
    disp(' ')
    for temp_exp_ind = 1:TotalNumberOfTests
        disp(['Computing robustness ... ', num2str(temp_exp_ind), '/', num2str(TotalNumberOfTests)])
        temp_traj_file_name = [trajectory_file_prefix, num2str(temp_exp_ind - 1), '.pkl'];
        [temp_T, temp_XT, temp_YT, temp_LT, temp_CLG, temp_GRD] = get_saved_webots_traj(temp_traj_file_name);
        temp_traj.T = temp_T;
        temp_traj.XT = temp_XT;
        % Compute_Robustness_on_trajectory is a modified S-Taliro file.
        [~, temp_rob, temp_aux] = Compute_Robustness_on_trajectory(model, model_type, phi, preds, opt, temp_T, temp_XT, temp_YT, temp_LT, temp_CLG, temp_GRD);
        robustness_list(temp_exp_ind) = temp_rob;
        aux_cell_array{temp_exp_ind} = temp_aux;
        traj_cell_array{temp_exp_ind} = temp_traj;
    end
    falsified_examples = find(robustness_list < 0.0);
    number_of_falsified = length(falsified_examples);
    [min_rob, min_rob_ind] = min(robustness_list);
    [close_zero_rob, close_zero_rob_ind] = min(abs(robustness_list));
    actual_close_zero_rob = robustness_list(close_zero_rob_ind);
    matlab_filename = ['Test1_R4_', evaluated_set, '_', num2str(eval_exp_ind), '.mat'];
    save(matlab_filename);

    disp('Finished!')
    disp(['Saved to: ', matlab_filename])
end
%disp('Please save the workspace variables if you need them later on!')
