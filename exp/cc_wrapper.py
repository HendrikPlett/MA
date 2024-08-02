# Standard
import os
from typing import Tuple, Iterable
import pandas as pd

# Third party
from cc_ground_truth import graph, latex_name


#------------------------------------------------------
# Chamber configuration constants

CHAMBER_CONFIGURATIONS = {
    'lt_camera_walks_v1': {'chamber': 'lt', 'configuration': 'camera'},
    'lt_color_regression_v1': {'chamber': 'lt', 'configuration': 'camera'},
    'lt_interventions_standard_v1': {'chamber': 'lt', 'configuration': 'standard'},
    'lt_walks_v1': {'chamber': 'lt', 'configuration': 'standard'},
    'wt_walks_v1': {'chamber': 'wt', 'configuration': 'standard'},
    'lt_malus_v1': {'chamber': 'lt', 'configuration': 'standard'},
    'wt_bernoulli_v1': {'chamber': 'wt','configuration': 'standard'},
    'wt_changepoints_v1': {'chamber': 'wt', 'configuration': 'standard'},
    'wt_intake_impulse_v1': {'chamber': 'wt', 'configuration': 'standard'},
    'wt_pressure_control_v1': {'chamber': 'wt', 'configuration': 'pressure-control'},
    'lt_test_v1': {'chamber': 'lt', 'configuration': 'standard'},
    'wt_test_v1': {'chamber': 'wt', 'configuration': 'standard'},
    'lt_camera_test_v1': {'chamber': 'lt', 'configuration': 'camera'},
    'wt_validate_v1': {'chamber': 'wt','configuration': 'standard'},
    'wt_pc_validate_v1': {'chamber': 'wt','configuration': 'pressure-control'},
    'lt_validate_v1': {'chamber': 'lt','configuration': 'standard'},
    'lt_camera_validate_v1': {'chamber': 'lt', 'configuration': 'standard'},
}

#------------------------------------------------------
# Variable groups for chamber: lt, configuration: camera

SMALL_VAR = ["red", "green", "blue", "current", 
        "ir_1", "vis_1", "ir_2", "vis_2", "ir_3", "vis_3"] # 10 Variables

MID_VAR = ["red", "green", "blue", "current", "pol_1", "pol_2",
        "angle_1", "angle_2",
        "ir_1", "vis_1", "ir_2", "vis_2", "ir_3", "vis_3", "l_11", "l_12", "l_21", "l_22",
        "l_31", "l_32"] # 20 Variables

ALL_VAR = ["red", "green", "blue", "osr_c", "v_c", "current", "pol_1", "pol_2",
        "osr_angle_1", "osr_angle_2", "v_angle_1", "v_angle_2", "angle_1", "angle_2",
        "ir_1", "vis_1", "ir_2", "vis_2", "ir_3", "vis_3", "l_11", "l_12", "l_21", "l_22",
        "l_31", "l_32", "diode_ir_1", "diode_vis_1", "diode_ir_2", "diode_vis_2", "diode_ir_3",
        "diode_vis_3", "t_ir_1", "t_vis_1", "t_ir_2", "t_vis_2", "t_ir_3","t_vis_3"] # 38 Variables

#------------------------------------------------------
# Variable positions for visualization for light tunnel

POSITIONS_LT = {
    "red": (-1, 0),
    "green": (0, 0),
    "blue": (1, 0),
    "osr_c": (-1, 2.5),
    "v_c": (1, 2.5),
    "current": (0, 1.5),
    "pol_1": (2, 0.5),
    "pol_2": (3, 0.5),
    "osr_angle_1": (4, 2.5),
    "osr_angle_2": (5.5, 1),
    "v_angle_1": (3, 2.5),
    "v_angle_2": (4.5, 1.5),
    "angle_1": (3, 1.5),
    "angle_2": (4, 0.5),
    "ir_1": (-3, -0.5),
    "vis_1": (-2, -2),
    "ir_2": (-1, -3),
    "vis_2": (1, -3),
    "ir_3": (1.75, -2.25),
    "vis_3": (3.5, -1),
    "l_11": (-3.5, -1.5),
    "l_12": (-2, -1),
    "l_21": (0, -2),
    "l_22": (0, -4),
    "l_31": (2.75, -1.75),
    "l_32": (4, -2),
    "diode_ir_1": (-4.5, -1),
    "diode_vis_1": (-3, -3.5),
    "diode_ir_2": (-1, -5),
    "diode_vis_2": (2, -4),
    "diode_ir_3": (3, -3.5),
    "diode_vis_3": (5, -0.5),
    "t_ir_1": (-4.5, 0),
    "t_vis_1": (-3.5, -2.5),
    "t_ir_2": (-2.5, -4.5),
    "t_vis_2": (1.5, -5),
    "t_ir_3": (4.5, -3),
    "t_vis_3": (5, -1.5)
}

#------------------------------------------------------
# Function to transform variable name to Latex

LATEX_NAME = latex_name


#------------------------------------------------------
# Helper class to fetch data

class CCWrapper:

    def __init__(self, data_path = "./cc_data/"):
        self._exp_family = None
        self._configuration = None
        self._variables = None
        self._cc_data_path = data_path

    def fetch_true_dag(self):
        true_dag = graph(
            chamber=self._configuration['chamber'],
            configuration=self._configuration['configuration']
            ).loc[self._variables, self._variables]
        return true_dag

    def fetch_experiments(self, experiments: list[str], sizes = None):

        # Check validity of input
        if (sizes is not None) and (len(experiments) != len(sizes)):
            raise ValueError("As many sizes as experiments necessary.")

        # Ensure the dataset is available
        if not os.path.exists(os.path.join(self._cc_data_path, self._exp_family)):
            msg = os.path.join(os.getcwd(), self._cc_data_path, self._exp_family)
            raise ValueError(f"Experiment family data not available under {msg}")

        # Put all experiments with desired number of rows and variables into list
        experiments_data = []

        for experiment in experiments:
            path = os.path.join(self._cc_data_path, self._exp_family, f"{experiment}.csv")
            if sizes is not None:
                size = sizes[experiments.index(experiment)]
                experiments_data.append(
                    pd.read_csv(path).iloc[0:size, :].loc[:, self._variables]
                )
            else:
                experiments_data.append(
                    pd.read_csv(path).loc[:, self._variables]
                )

        return list(experiments_data)

    def set_variables(self, variables: list[str]):
        self._variables = variables

    def set_exp_family(self, exp_family: str):
        if not isinstance(exp_family, str):
            raise TypeError("Must pass a string as exp_family.")
        if exp_family not in tuple(CHAMBER_CONFIGURATIONS.keys()):
            raise ValueError("Not a valid experimental family")
        self._exp_family = exp_family
        self._configuration = CHAMBER_CONFIGURATIONS[self._exp_family]

    def get_exp_family(self):
        return self._exp_family

    def get_variables(self):
        return self._variables