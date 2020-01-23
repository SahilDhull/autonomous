import sys
import os
import scipy.io as spio
#sys.stdout.flush()
FILE_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(FILE_PATH + "/../../../")
from controller import Supervisor
from controller import Robot
from Sim_ATAV.simulation_control.webots_vehicle import WebotsVehicle
from Sim_ATAV.simulation_control.webots_pedestrian import WebotsPedestrian
from Sim_ATAV.simulation_control.sim_object_generator import SimObjectGenerator
from Sim_ATAV.simulation_control.supervisor_controls import SupervisorControls


def loadmat(filename):
    """this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects"""
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(mat_obj_dict):
    """checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries"""
    for key in mat_obj_dict:
        if isinstance(mat_obj_dict[key], spio.matlab.mio5_params.mat_struct):
            mat_obj_dict[key] = _todict(mat_obj_dict[key])
    return mat_obj_dict


def _todict(matobj):
    """A recursive function which constructs from matobjects nested dictionaries"""
    mat_obj_dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            mat_obj_dict[strg] = _todict(elem)
        else:
            mat_obj_dict[strg] = elem
    return mat_obj_dict


VHC_HEIGHT_FROM_GROUND = 0.35

# Read Matlab trajectory:
if len(sys.argv) > 1:
    matlab_traj_file = sys.argv[1]
else:
    matlab_traj_file = 'matlab_trajectory.mat'

mat_info = loadmat(matlab_traj_file)
print('Matlab trajectory file: {}'.format(matlab_traj_file))

try:
    num_of_vehicles = mat_info['num_of_vehicles']
except:
    num_of_vehicles = 0
try:
    num_of_pedestrians = mat_info['num_of_pedestrians']
    num_states_per_ped = mat_info['num_states_per_ped']
except:
    num_of_pedestrians = 0
    num_states_per_ped = 3

try:
    time_step = mat_info['time_step']
    time_step = int(1000*time_step)
except:
    time_step = 10

try:
    x_pos_ind = mat_info['x_pos_ind']
    y_pos_ind = mat_info['y_pos_ind']
    theta_ind = mat_info['theta_ind']
    num_states_per_vhc = mat_info['num_states_per_vhc']
except:
    x_pos_ind = 0
    y_pos_ind = 1
    theta_ind = 2
    num_states_per_vhc = 3

try:
    follow_vhc_ind = mat_info['follow_vhc_ind']
except:
    follow_vhc_ind = 0

try:
    follow_height = mat_info['follow_height']
except:
    follow_height = 130

try:
    cost_hist = mat_info['cost_hist']
except:
    cost_hist = None

traj = mat_info['traj']


#print(traj)
colors = [[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]
models = ['ToyotaPrius', 'BmwX5', 'CitroenCZero', 'LincolnMKZ', 'RangeRoverSportSVR', 'TeslaModel3']

models_dict = {}
for vhc_i in range(num_of_vehicles):
    try:
        modelname_ids = mat_info['vehicle_model_ids']
        models_dict[vhc_i] = models[modelname_ids[vhc_i]]
    except:
        models_dict[vhc_i] = models[vhc_i % len(models)]

supervisor_control = SupervisorControls()
supervisor_control.init(None)
sim_obj_generator = SimObjectGenerator()

display = supervisor_control.getDisplay('display')

print("Time Step set to: {}".format(supervisor_control.set_default_time_step(time_step=time_step)))

vhc_objects = []
for vhc_i in range(num_of_vehicles):
    x_pos = traj[0, num_states_per_vhc * vhc_i + x_pos_ind]
    y_pos = traj[0, num_states_per_vhc * vhc_i + y_pos_ind]
    theta = traj[0, num_states_per_vhc * vhc_i + theta_ind]
    color = colors[vhc_i % len(colors)]
    vhc_objects.append(WebotsVehicle())
    vhc_objects[-1].def_name = 'VHC_'+str(vhc_i)
    vhc_objects[-1].vehicle_model = models_dict[vhc_i]
    vhc_objects[-1].current_position = [x_pos, VHC_HEIGHT_FROM_GROUND, y_pos]
    vhc_objects[-1].current_orientation = theta
    vhc_objects[-1].color = color[:]
    vhc_objects[-1].controller = "void"
    vhc_objects[-1].is_controller_name_absolute = True
    vhc_str = sim_obj_generator.generate_vehicle_string(vhc_objects[-1])
    supervisor_control.add_obj_to_sim_from_string(vhc_str)
    vhc_objects[-1].node = supervisor_control.get_obj_node(vhc_objects[-1])
    vhc_objects[-1].translation = supervisor_control.get_obj_field(vhc_objects[-1], "translation")
    vhc_objects[-1].rotation = supervisor_control.get_obj_field(vhc_objects[-1], "rotation")
    vhc_objects[-1].name = supervisor_control.get_obj_field(vhc_objects[-1], "name")
    print('Vehicle {} initialized at x={}, y={}, theta={} Color (rgb)={}'.format(vhc_i, x_pos, y_pos, theta, color))

viewpoint = supervisor_control.getFromDef('VIEWPOINT')
if viewpoint is not None:
    follow_point = viewpoint.getField('follow')
    viewpoint_position = viewpoint.getField('position')
    if follow_point is not None:
        follow_point.setSFString(vhc_objects[follow_vhc_ind].name.getSFString())
        print('Viewpoint set to follow the vehicle {}'.format(follow_vhc_ind))

# # No pedestrian support for playback yet.
# for ped_i in range(num_of_pedestrians):
#     x_pos = traj[num_states_per_ped * ped_i + x_pos_ind + num_states_per_vhc * num_of_vehicles]
#     y_pos = traj[num_states_per_ped * ped_i + y_pos_ind + num_states_per_vhc * num_of_vehicles]
#     theta = traj[num_states_per_ped * ped_i + theta_ind + num_states_per_vhc * num_of_vehicles]
#     color = colors[vhc_i % len(colors)]
#     ped = WebotsPedestrian()
#     ped.def_name = 'PED_'+str(ped_i)
#     ped.current_position = [x_pos, 0.35, y_pos]
#     ped.current_rotation = [0, 1, 0, theta]
#     ped_str = sim_obj_generator.generate_pedestrian_string(ped)
#     supervisor_control.add_obj_to_sim_from_string(ped_str)
#     print('Pedestrian {} initialized at x={}, y={}, theta={} Color (rgb)={}'.format(vhc_i, x_pos, y_pos, theta, color))
display.setFont('Lucida Console', 20, True)
for time_ind in range(len(traj)):
    if cost_hist is not None and time_ind % 10 == 0 and time_ind > 0:
        display.setColor(0x000000)
        display.fillRectangle(0, 0, display.getWidth(), display.getHeight())
        display.setColor(0xFFFFFF)
        cur_cost = cost_hist[:, time_ind-1]
        display.drawText('1->A1: {:.2f}'.format(cur_cost[0]), 5, 5)
        display.drawText('1->A2: {:.2f}'.format(cur_cost[1]), 5, 30)
        display.drawText('min.A1: {:.2f}'.format(min(cost_hist[0,0:time_ind-1])), 5, 55)
        display.drawText('min.A2: {:.2f}'.format(min(cost_hist[1,0:time_ind-1])), 5, 80)
        #print(cur_cost)
    for vhc_i in range(num_of_vehicles):
        x_pos = traj[time_ind, num_states_per_vhc * vhc_i + x_pos_ind]
        y_pos = traj[time_ind, num_states_per_vhc * vhc_i + y_pos_ind]
        theta = traj[time_ind, num_states_per_vhc * vhc_i + theta_ind]
        #print('Vhc:{} {} {} {}'.format(vhc_i, x_pos, y_pos, theta))
        #sys.stdout.flush()
        supervisor_control.set_obj_position_3D(vhc_objects[vhc_i], [x_pos, VHC_HEIGHT_FROM_GROUND, y_pos])
        supervisor_control.set_obj_rotation(vhc_objects[vhc_i], [0, 1, 0, theta])
        supervisor_control.reset_obj_physics(vhc_objects[vhc_i])
        if vhc_i == follow_vhc_ind and viewpoint_position is not None:
            viewpoint_position.setSFVec3f([x_pos, follow_height, y_pos])
    supervisor_control.step_simulation(int(time_step))
supervisor_control.pause_simulation()
