disp(' ')

disp('The specification:')
phi = '[]((![]_[0, 0.6](brake_active /\ !collision_estimated)) /\ !( (brake_active /\ X(!brake_active)) /\ <>_(0, 0.5]( (brake_active /\ X(!brake_active)) /\ <>_(0, 0.5](brake_active /\ X(!brake_active)))))'

cur_traj_ind = 1;
EGO_X = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
EGO_Y = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
EGO_THETA = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
EGO_V = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
EGO_A = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
EGO_YAW_RATE = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
AGENT_1_X = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
AGENT_1_Y = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
AGENT_1_THETA = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
AGENT_1_V = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
AGENT_1_A = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
AGENT_1_YAW_RATE = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
EGO_THROTTLE = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
AGENT_1_OVERALL_LOCAL_ERROR = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
AGENT_1_RADAR_LOCAL_ERROR = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
AGENT_1_CAMERA_LOCAL_ERROR = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
AGENT_1_VISIBILITY = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
AGENT_1_DISTANCE = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
AGENT_1_FUTURE_MIN_DISTANCE = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
AGENT_1_FUTURE_MIN_DIST_TIME = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
NUM_ITEMS_IN_TRAJ = cur_traj_ind - 1;

ii = 1;
preds(ii).str='collision_estimated';
preds(ii).A = zeros(1, NUM_ITEMS_IN_TRAJ);
preds(ii).A(AGENT_1_FUTURE_MIN_DISTANCE) = 1;
preds(ii).b = 0.5;

ii = ii+1;
preds(ii).str='brake_active';
preds(ii).A = zeros(1, NUM_ITEMS_IN_TRAJ);
preds(ii).A(EGO_THROTTLE) = 1;
preds(ii).b = -0.5;

