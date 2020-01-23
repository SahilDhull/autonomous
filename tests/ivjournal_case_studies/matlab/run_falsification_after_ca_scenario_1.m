global currentExpIndex
global staliro_run_count
global FALSIF_Run_Number

FALSIF_Run_Number = 1;
FALSIFICATION_SOLVER = 'SA_Taliro';
TOTAL_SA_BUDGET = 300;
SINGLE_SA_BUDGET = 100;
FALSIF_EXP_RESULTS_FILE_NAME = [FALSIF_EXP_RESULTS_FILE_PREFIX, FALSIFICATION_SOLVER, '_', num2str(FALSIF_Run_Number), '.csv'];
init_cond = get_parameter_ranges_scenario_1();
model = @webots_simulation_ivjournal_scenario_1;
input_range = [];
cp_array = [];
opt = staliro_options();
opt.runs = 1;
opt.black_box = 1;
opt.SampTime = 0.010;
opt.spec_space = 'X';
opt.optimization_solver = FALSIFICATION_SOLVER;
opt.taliro = 'dp_taliro';
opt.search_space_constrained.constrained = true;
opt.map2line = 0;
opt.falsification = 1;

disp(' ')
disp(['Running S-TaLiRo ... '])
opt


executed_test_pool = [];
experiments_to_run_pool = find(robustness_list>0);
results = cell(0);
history = cell(0);
remaining_budget = TOTAL_SA_BUDGET;
run_count = 0;


sim_duration = 7.0


while remaining_budget > 0
    run_count = run_count + 1;
    staliro_run_count = run_count;
    if strcmpi(FALSIFICATION_SOLVER, 'SA_Taliro')
        opt.optim_params.n_tests = min(remaining_budget, SINGLE_SA_BUDGET) + 1;
    else
        opt.optim_params.n_tests = min(remaining_budget, SINGLE_SA_BUDGET);
    end
    remaining_tests_pool = setdiff(experiments_to_run_pool, executed_test_pool);
    if isempty(remaining_tests_pool)
        remaining_budget = 0;
    else
        [val, temp_exp_ind] = min(robustness_list(remaining_tests_pool));
        exp_matlab_ind = remaining_tests_pool(temp_exp_ind);
        executed_test_pool = [executed_test_pool, exp_matlab_ind];
        disp(' ')
        disp(['Running S-TaLiRo ... ', num2str(exp_matlab_ind), '/', num2str(TotalNumberOfTests)])
        tic
        critical_pred = aux_cell_array{exp_matlab_ind}.pred;
        currentExpIndex = exp_matlab_ind - 1;
        init_sample = get_initial_sample_from_python_scenario_1(currentExpIndex);
        if strcmpi(FALSIFICATION_SOLVER, 'SA_Taliro')
            opt.optim_params.init_sample = init_sample';
            % opt.optim_params.init_rob = robustness_list(exp_matlab_ind);  % We don't need to recompute initial robustness
        end
        if strcmpi(FALSIFICATION_SOLVER, 'CE_Taliro')
            opt.optim_params.num_iteration = 10;
        end
        [results{run_count}, history{run_count}] = staliro(model,init_cond,input_range,cp_array,phi,preds,sim_duration,opt);
        results{run_count}.run(results{run_count}.optRobIndex).bestRob
        if strcmpi(FALSIFICATION_SOLVER, 'SA_Taliro')
            remaining_budget = remaining_budget - results{run_count}.run.nTests + 1;
        else
            remaining_budget = remaining_budget - results{run_count}.run.nTests;
        end
        runtime = toc;
        runtime
        res_filename = ['results_scenario_1_r1_run_', num2str(run_count), '_', datestr(datetime("now"), 'yyyy_mm_dd__HH_MM'), '.mat'];
        save(res_filename)
        disp(["Results are saved to: ", res_filename])
    end
end


