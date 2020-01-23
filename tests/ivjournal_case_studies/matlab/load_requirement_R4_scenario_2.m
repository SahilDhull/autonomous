disp(' ')
disp('The specification:')
phi = '[]!( ([]_[0, 0.4]( (!agent1_hit) /\ agent1_visible /\ (!(agent1_detected /\ not_agent1_local_error)))) /\ (<>_(0.4, 3.4]agent1_hit))'

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
AGENT_2_X = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
AGENT_2_Y = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
AGENT_2_THETA = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
AGENT_2_V = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
AGENT_2_A = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
AGENT_2_YAW_RATE = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
AGENT_1_OVERALL_LOCAL_ERROR = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
AGENT_2_OVERALL_LOCAL_ERROR = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
AGENT_1_LIDAR_LOCAL_ERROR = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
AGENT_2_LIDAR_LOCAL_ERROR = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
AGENT_1_CAMERA_LOCAL_ERROR = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
AGENT_2_CAMERA_LOCAL_ERROR = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
AGENT_1_RADAR_LOCAL_ERROR = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
AGENT_2_RADAR_LOCAL_ERROR = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
AGENT_1_VISIBILITY = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
AGENT_2_VISIBILITY = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
AGENT_1_DISTANCE = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
AGENT_2_DISTANCE = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
NUM_ITEMS_IN_TRAJ = cur_traj_ind - 1;

ii = 1;
preds(ii).str='agent1_visible';
preds(ii).A = zeros(1, NUM_ITEMS_IN_TRAJ);
preds(ii).A(AGENT_1_VISIBILITY) = -1;
preds(ii).b = -0.25;

ii = ii+1;
preds(ii).str='agent1_detected';
preds(ii).A = zeros(1, NUM_ITEMS_IN_TRAJ);
preds(ii).A(AGENT_1_OVERALL_LOCAL_ERROR) = -1;
preds(ii).b = 0.1;

ii = ii+1;
preds(ii).str='not_agent1_local_error';
preds(ii).A = zeros(1, NUM_ITEMS_IN_TRAJ);
preds(ii).A(AGENT_1_OVERALL_LOCAL_ERROR) = 1;
preds(ii).b = 1.0;

ii = ii+1;
preds(ii).str='agent1_hit';
preds(ii).A = zeros(1, NUM_ITEMS_IN_TRAJ);
preds(ii).A(AGENT_1_DISTANCE) = 1;
preds(ii).b = 0.05;
