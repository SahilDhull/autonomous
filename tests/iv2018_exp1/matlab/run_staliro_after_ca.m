global criticalParamList
global currentExpIndex
global staliro_run_count
global FALSIF_Run_Number
global FALSIF_EXP_FILE_NAME

model = @get_webots_traj_with_critical_params;
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
opt.falsification = 0;
opt.seed = 100 + FALSIF_Run_Number;
opt

executed_test_pool = [];
experiments_to_run_pool = [1:length(robustness_list)];
results = cell(0);
remaining_budget = TOTAL_SA_BUDGET;
run_count = 0;
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
        criticalParamList = [13, 14, 15];
        currentExpIndex = exp_matlab_ind - 1;
        [init_sample, init_cond] = get_initial_sample_and_range_from_python(currentExpIndex, FALSIF_EXP_FILE_NAME, criticalParamList);
        if strcmpi(FALSIFICATION_SOLVER, 'SA_Taliro')
            opt.optim_params.init_sample = init_sample';
            opt.optim_params.init_rob = robustness_list(exp_matlab_ind);  % We don't need to recompute initial robustness
        end
        if strcmpi(FALSIFICATION_SOLVER, 'CE_Taliro')
            opt.optim_params.num_iteration = 10;
        end
        results{run_count} = staliro(model,init_cond,input_range,cp_array,phi,preds,time,opt);
        results{run_count}.run(results{run_count}.optRobIndex).bestRob
        if strcmpi(FALSIFICATION_SOLVER, 'SA_Taliro')
            remaining_budget = remaining_budget - results{run_count}.run.nTests + 1;
        else
            remaining_budget = remaining_budget - results{run_count}.run.nTests;
        end
        runtime = toc;
        runtime
    end
end
