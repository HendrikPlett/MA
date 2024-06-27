# Standard
import os
from typing import Tuple, Iterable
import pandas as pd

# Third party
from causalchamber.datasets import Dataset
from causalchamber.ground_truth import variables, graph


class CausalChamberWrapper:

    #------------------------------------------------------
    # Chamber configuration constants
    chamber_configurations = {
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
    # Variables for chamber: lt, configuration: camera
    # TODO: Later make this dynamic for all experimantal families.

    lt_s_core_var =  ['red', 'green', 'blue', 'current', 'ir_1', 'ir_2', 
                    'ir_3', 'vis_1', 'vis_2', 'vis_3'] # 10 Variables

    lt_s_all_var_wo_sensor_parameters = ['red', 'green', 'blue', 'current', 'ir_1', 
                                        'ir_2', 'ir_3', 'vis_1', 'vis_2', 'vis_3', 
                                        'pol_1', 'pol_2', 'angle_1', 'angle_2', 
                                        'l_11', 'l_12', 'l_21', 'l_22', 'l_31', 'l_32'] # 20 Variables

    lt_s_all_var = variables(chamber="lt", configuration="standard")

    
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
        if not sizes is None:
            if len(experiments) != len(sizes):
                raise ValueError("As many sizes as experiments necessary")
        
        # Download dataset if not already available
        if not os.path.exists(concat_to_path([self._cc_data_path, self._exp_family])):
            Dataset(name=self._exp_family, root=self._cc_data_path, download=True)
        # --- Now the dataset is loaded in either case

        # Put all experiments with desired number of rows and variables into list
        experiments_data = []

        if not sizes is None:
            for experiment, size in zip(experiments, sizes):
                path = concat_to_path([self._cc_data_path, self._exp_family, experiment])+".csv"
                experiments_data.append(
                    pd.read_csv(path).iloc[0:size, :].loc[:, self._variables]
                )
        else:
            for experiment in experiments:
                path = concat_to_path([self._cc_data_path, self._exp_family, experiment])+".csv"
                experiments_data.append(
                    pd.read_csv(path).loc[:, self._variables]
                )

        return list(experiments_data)
    
    def set_variables(self, variables: list[str]):
        self._variables = variables

    def set_exp_family(self, exp_family: str):
        if not isinstance(exp_family, str):
            raise TypeError("Must pass a string as exp_family.")
        if exp_family not in tuple(self.chamber_configurations.keys()):
            raise ValueError("Not a valid experimental family")
        self._exp_family = exp_family
        self._configuration = self.chamber_configurations[self._exp_family]

    def get_exp_family(self):
        return self._exp_family
    
    def get_variables(self):
        return self._variables


def concat_to_path(dirs: Iterable[str]):
    path = ""
    for dir in dirs:
        path += (dir+"/")
    path = path[:-1]
    return path
