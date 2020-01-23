"""Saves and load trajectory to / from file
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""

import pickle


def load_selected_trajectory(traj_file_name):
    """Loads a pickled and saved trajectory from a file."""
    with open(traj_file_name, 'rb') as traj_file:
        trajectory = pickle.load(traj_file)
        traj_file.close()
    return trajectory


def load_selected_trajectory_for_matlab(traj_file_name):
    trajectory = load_selected_trajectory(traj_file_name)
    return trajectory.tolist()


def save_trajectory_to_file(trajectory, traj_file_name):
    """Pickles and saves the given trajectory into file. Creates the file if it doesn't exist."""
    with open(traj_file_name, 'wb+') as traj_file:
        pickle.dump(trajectory, traj_file)
        traj_file.close()
