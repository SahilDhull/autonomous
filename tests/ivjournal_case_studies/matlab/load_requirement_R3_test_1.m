disp(' ')
disp('The specification:')
phi = '[]((!ped1_hit U ([]_[0, 0.4](ped1_visible /\ !ped1_detected /\ !ped1_hit)))->([]_[0.4, inf]!ped1_hit))'

cur_traj_ind = 1;
EGO_X = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
EGO_Y = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
EGO_THETA = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
EGO_V = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
PED_X = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
PED_Y = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
OVERALL_PED_LOCAL_ERROR = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
LIDAR_PED_LOCAL_ERROR = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
CAMERA_PED_LOCAL_ERROR = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
PED_VISIBILITY = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
NUM_ITEMS_IN_TRAJ = cur_traj_ind - 1;

ii = 1;
preds(ii).str='vhc_moving';
preds(ii).A = zeros(1, NUM_ITEMS_IN_TRAJ);
preds(ii).A(EGO_V) = -1;
preds(ii).b = -0.01;

ii = ii+1;
preds(ii).str='ped1_visible';
preds(ii).A = zeros(1, NUM_ITEMS_IN_TRAJ);
preds(ii).A(PED_VISIBILITY) = -1;
preds(ii).b = -0.25;

ii = ii+1;
preds(ii).str='ped1_detected';
preds(ii).A = zeros(1, NUM_ITEMS_IN_TRAJ);
preds(ii).A(OVERALL_PED_LOCAL_ERROR) = -1;
preds(ii).b = 0.1;

ii = ii+1;
preds(ii).str='ped1_hit';
preds(ii).A = zeros(1, NUM_ITEMS_IN_TRAJ);
preds(ii).A(PED_Y) = 1;
preds(ii).A(EGO_Y) = -1;
preds(ii).b = 3.85;
