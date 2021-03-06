disp(' ')
disp('The specification:')
phi = '[]((ped1_visible /\ (!ped1_detected /\ ped1_local_error)) -> (<>_(0, 0.8]((ped1_detected /\ !ped1_local_error) \/ !ped1_visible)))'

cur_traj_ind = 1;
EGO_X = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
EGO_Y = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
EGO_THETA = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
EGO_V = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
PED_X = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
PED_Y = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
OVERALL_PED_LOCAL_ERROR = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
% LIDAR_PED_LOCAL_ERROR = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
CAMERA_PED_LOCAL_ERROR = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
PED_VISIBILITY = cur_traj_ind; cur_traj_ind = cur_traj_ind + 1;
NUM_ITEMS_IN_TRAJ = cur_traj_ind - 1;

ii = 1;
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
preds(ii).str='ped1_local_error';
preds(ii).A = zeros(1, NUM_ITEMS_IN_TRAJ);
preds(ii).A(OVERALL_PED_LOCAL_ERROR) = -1;
preds(ii).b = -1.0;

