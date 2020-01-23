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

time = 10;
disp(' ')
disp(['Total Simulation time:', num2str(time)]);

% Note that not all of the options below are used for robustness
% computation.
opt = staliro_options();
opt.black_box = 1;
opt.SampTime = 0.010;
opt.spec_space = 'X';
opt.taliro = 'dp_taliro';
opt.map2line = 0;
opt.falsification = 0;
disp(' ')
disp('S-TaLiRo options:')
opt

disp(' ')
disp('The specification:')
phi = '[] !(vhc_moving /\ ((ped1_hit /\ ped1_in_corridor) \/ (vhc1_hit /\ vhc1_in_corridor) \/ (vhc2_hit /\ vhc2_in_corridor) \/ (vhc3_hit /\ vhc3_in_corridor) \/ (vhc4_hit /\ vhc4_in_corridor) \/ (vhc5_hit /\ vhc5_in_corridor) \/ ped1_zero_dist \/ vhc1_zero_dist \/ vhc2_zero_dist \/ vhc3_zero_dist \/ vhc4_zero_dist \/ vhc5_zero_dist))'

NUM_ITEMS_IN_TRAJ = 55;
ii = 1;
preds(ii).str='vhc_moving';
preds(ii).A = zeros(1, NUM_ITEMS_IN_TRAJ);
preds(ii).A(4) = -1;
preds(ii).b = -0.1;

ii = ii+1;
preds(ii).str='ped1_hit';
preds(ii).A = zeros(1, NUM_ITEMS_IN_TRAJ);
preds(ii).A(45) = 1;
preds(ii).b = 1000.0*0.05;

ii = ii+1;
preds(ii).str='ped1_in_corridor';
preds(ii).A = zeros(1, NUM_ITEMS_IN_TRAJ);
preds(ii).A(44) = -1;
preds(ii).b = -1000.0*0.5;

ii = ii+1;
preds(ii).str='vhc1_hit';
preds(ii).A = zeros(1, NUM_ITEMS_IN_TRAJ);
preds(ii).A(47) = 1;
preds(ii).b = 1000.0*0.05;

ii = ii+1;
preds(ii).str='vhc1_in_corridor';
preds(ii).A = zeros(1, NUM_ITEMS_IN_TRAJ);
preds(ii).A(46) = -1;
preds(ii).b = -1000.0*0.5;

ii = ii+1;
preds(ii).str='vhc2_hit';
preds(ii).A = zeros(1, NUM_ITEMS_IN_TRAJ);
preds(ii).A(49) = 1;
preds(ii).b = 1000.0*0.05;

ii = ii+1;
preds(ii).str='vhc2_in_corridor';
preds(ii).A = zeros(1, NUM_ITEMS_IN_TRAJ);
preds(ii).A(48) = -1;
preds(ii).b = -1000.0*0.5;

ii = ii+1;
preds(ii).str='vhc3_hit';
preds(ii).A = zeros(1, NUM_ITEMS_IN_TRAJ);
preds(ii).A(51) = 1;
preds(ii).b = 1000.0*0.05;

ii = ii+1;
preds(ii).str='vhc3_in_corridor';
preds(ii).A = zeros(1, NUM_ITEMS_IN_TRAJ);
preds(ii).A(50) = -1;
preds(ii).b = -1000.0*0.5;

ii = ii+1;
preds(ii).str='vhc4_hit';
preds(ii).A = zeros(1, NUM_ITEMS_IN_TRAJ);
preds(ii).A(53) = 1;
preds(ii).b = 1000.0*0.05;

ii = ii+1;
preds(ii).str='vhc4_in_corridor';
preds(ii).A = zeros(1, NUM_ITEMS_IN_TRAJ);
preds(ii).A(52) = -1;
preds(ii).b = -1000.0*0.5;

ii = ii+1;
preds(ii).str='vhc5_hit';
preds(ii).A = zeros(1, NUM_ITEMS_IN_TRAJ);
preds(ii).A(55) = 1;
preds(ii).b = 1000.0*0.05;

ii = ii+1;
preds(ii).str='vhc5_in_corridor';
preds(ii).A = zeros(1, NUM_ITEMS_IN_TRAJ);
preds(ii).A(54) = -1;
preds(ii).b = -1000.0*0.5;

ii = ii+1;
preds(ii).str='ped1_zero_dist';
preds(ii).A = zeros(1, NUM_ITEMS_IN_TRAJ);
preds(ii).A(45) = 1;
preds(ii).b = 0.0;

ii = ii+1;
preds(ii).str='vhc1_zero_dist';
preds(ii).A = zeros(1, NUM_ITEMS_IN_TRAJ);
preds(ii).A(47) = 1;
preds(ii).b = 0.0;

ii = ii+1;
preds(ii).str='vhc2_zero_dist';
preds(ii).A = zeros(1, NUM_ITEMS_IN_TRAJ);
preds(ii).A(49) = 1;
preds(ii).b = 0.0;

ii = ii+1;
preds(ii).str='vhc3_zero_dist';
preds(ii).A = zeros(1, NUM_ITEMS_IN_TRAJ);
preds(ii).A(51) = 1;
preds(ii).b = 0.0;

ii = ii+1;
preds(ii).str='vhc4_zero_dist';
preds(ii).A = zeros(1, NUM_ITEMS_IN_TRAJ);
preds(ii).A(53) = 1;
preds(ii).b = 0.0;

ii = ii+1;
preds(ii).str='vhc5_zero_dist';
preds(ii).A = zeros(1, NUM_ITEMS_IN_TRAJ);
preds(ii).A(55) = 1;
preds(ii).b = 0.0;

evaluated_experiment_index = [1:20];

for eval_exp_ind = evaluated_experiment_index
    if strcmpi(evaluated_set, 'CA')
        TotalNumberOfTests = 47;
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

    model = staliro_blackbox();
    model_type = determine_model_type(model);
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
        temp_XT(:, 44:55) = 1000.0*temp_XT(:, 44:55); % Multiplying values by 1000 for normalization for robustness computation.
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
    matlab_filename = [evaluated_set, '_', num2str(eval_exp_ind), '.mat'];
    save(matlab_filename);

    disp('Finished!')
    disp(['Saved to: ', matlab_filename])
end
%disp('Please save the workspace variables if you need them later on!')
