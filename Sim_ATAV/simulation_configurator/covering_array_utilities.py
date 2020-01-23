"""Provides methods to handle covering array experiment files.
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""

import pandas as pd
import numpy as np


def load_experiment_data(csv_file_name, header_line_count=6, index_col=None):
    """Loads the experiment data from csv file."""
    # In the ACTS generated experiments, first 6 lines are the header for the file. We skip those.
    if header_line_count > 0:
        exp_data_frame = pd.read_csv(csv_file_name, skiprows=range(header_line_count), index_col=index_col)
    else:
        exp_data_frame = pd.read_csv(csv_file_name, index_col=index_col)
    return exp_data_frame


def load_experiment_results_data(csv_file_name):
    """Loads the experiment results data from csv file."""
    # In results file, we don't have header lines but we have the first column as index column.
    # We get rid of that index columns by setting index_col = 0
    exp_data_frame = pd.read_csv(csv_file_name, index_col=0)
    return exp_data_frame


def get_experiment_field_value(exp_data_frame, exp_ind, field_name):
    """Returns a specific field of a specific experiment."""
    return exp_data_frame.iloc[[exp_ind]][field_name].iloc[0]


def set_experiment_field_value(exp_data_frame, exp_ind, field_name, new_value):
    """Sets a specific field of a specific experiment."""
    exp_data_frame.at[exp_ind, field_name] = new_value


def get_experiment_all_fields(exp_data_frame, exp_ind):
    """Returns the whole row for an experiment."""
    return exp_data_frame.iloc[[exp_ind]]


def get_field_value_for_current_experiment(cur_experiment, field_name):
    """Returns a specific field of the given experiment."""
    return cur_experiment[field_name].iloc[0]


def save_experiment_results(csv_file_name, data_frame):
    data_frame.to_csv(csv_file_name)


def add_column_to_data_frame(original_data_frame, new_column_name, new_column_value):
    new_data_frame = original_data_frame
    new_data_frame.loc[:, new_column_name] = \
        pd.Series(new_column_value*np.ones(len(original_data_frame.index)), index=original_data_frame.index)
    return new_data_frame


def get_ca_strength_string(ca_strength):
    """In case of mixed strength, ca_strength will be an array."""
    if isinstance(ca_strength, list):
        ca_str = ''
        for (strength_ind, strength) in enumerate(ca_strength):
            ca_str = ca_str + (str(int(strength)) if strength_ind == len(ca_strength) - 1 else str(int(strength)) + '_')
    else:
        ca_str = str(int(ca_strength))
    return ca_str


def load_parameters_from_covering_array(environment_config_dict, parameter_name_type_dict,
                                        parameter_conversion_dict=None, ca_strength=2, exp_type='CA', exp_to_run=None):
    dict_of_parameters_dict = {}

    try:
        ca_exp_file_name = environment_config_dict['exp_config_folder'] + environment_config_dict['exp_short_name'] + \
            '_ca_' + get_ca_strength_string(ca_strength) + '_way.csv'

        if exp_type == 'CA':
            exp_data_frame = load_experiment_data(ca_exp_file_name, header_line_count=6)
        else:
            exp_data_frame = load_experiment_data(ca_exp_file_name, header_line_count=0, index_col=0)
        num_of_experiments = len(exp_data_frame.index)
        if exp_to_run is None:
            experiment_set = range(num_of_experiments)
        else:
            experiment_set = list(exp_to_run)

        for exp_ind in experiment_set:
            experiment_data = get_experiment_all_fields(exp_data_frame, exp_ind)
            for (param_name, param_type) in parameter_name_type_dict.items():
                if exp_ind not in dict_of_parameters_dict:
                    dict_of_parameters_dict[exp_ind] = {}
                dict_of_parameters_dict[exp_ind][param_name] = \
                    param_type(get_field_value_for_current_experiment(experiment_data, param_name))
                if parameter_conversion_dict is not None and param_name in parameter_conversion_dict:
                    dict_of_parameters_dict[exp_ind][param_name] = \
                        parameter_conversion_dict[param_name](dict_of_parameters_dict[exp_ind][param_name])
    except Exception as ex:
        print('Parameters could not be loaded from Covering Array. ERROR: ' + repr(ex))
    return dict_of_parameters_dict
