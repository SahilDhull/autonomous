% Indices of states in simulation trace:
cur_traj_ind = 1;
EGO_X = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
EGO_Y = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
EGO_THETA = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
EGO_V = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
AGENT_X = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
AGENT_Y = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
AGENT_THETA = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
AGENT_V = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
PED_X = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
PED_Y = cur_traj_ind;
NUM_ITEMS_IN_TRAJ = cur_traj_ind;

% Predicates for MTL requirement:
ii = 1;
preds(ii).str='y_check1';
preds(ii).A = zeros(1, NUM_ITEMS_IN_TRAJ);
preds(ii).A(AGENT_Y) = 1;
preds(ii).A(EGO_Y) = -1;
preds(ii).b = 1.5;

ii = ii+1;
preds(ii).str='y_check2';
preds(ii).A = zeros(1, NUM_ITEMS_IN_TRAJ);
preds(ii).A(AGENT_Y) = -1;
preds(ii).A(EGO_Y) = 1;
preds(ii).b = 1.5;

ii = ii+1;
preds(ii).str='x_check1';
preds(ii).A = zeros(1, NUM_ITEMS_IN_TRAJ);
preds(ii).A(AGENT_X) = 1;
preds(ii).A(EGO_X) = -1;
preds(ii).b = 8;

ii = ii+1;
preds(ii).str='x_check2';
preds(ii).A = zeros(1, NUM_ITEMS_IN_TRAJ);
preds(ii).A(AGENT_X) = -1;
preds(ii).A(EGO_X) = 1;
preds(ii).b = 0;

% Metric Temporal Logic Requirement:
phi = '[](!(y_check1 /\ y_check2 /\ x_check1 /\ x_check2))';

% Ranges for test parameters (ego_init_speed_m_s, ego_x_pos, pedestrian_speed):
init_cond = [0.0, 15.0;
             15.0, 25.0;
             2.0, 5.0];
         
% Provide our Matlab wrapper function for running the tests as the model.
model = @run_tutorial_example_from_matlab;
opt = staliro_options();
opt.runs = 1;  % Do falsification only once.
opt.black_box = 1;  % Because we use a custom Matlab function as the model.
opt.SampTime = 0.010;  % Sample time. Same as Webots world time step.
opt.spec_space = 'X';  % Requirements are defined on state space.
opt.optimization_solver = 'SA_Taliro';  % Use Simulated Annealing
opt.taliro = 'dp_taliro';  % Use dp_taliro to compute robustness
%opt.search_space_constrained.constrained = true; % search from a polyhedron
opt.map2line = 0;
opt.falsification = 1;  % Stop when falsified
opt.optim_params.n_tests = 100;  % maximum number of tries

sim_duration = 15.0;

disp(['Running S-TaLiRo ... '])

[results, history] = staliro(model, init_cond, [], [], phi, preds, sim_duration, opt);

res_filename = ['results_', datestr(datetime("now"), 'yyyy_mm_dd__HH_MM'), '.mat'];
save(res_filename)
disp(["Results are saved to: ", res_filename])
