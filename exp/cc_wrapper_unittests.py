import unittest 
import pandas as pd
import os
import glob


# Wrapper
from cc_wrapper import CCWrapper
from cc_wrapper import SMALL_VAR, MID_VAR, ALL_VAR
from cc_wrapper import CHAMBER_CONFIGURATIONS
VAR = [SMALL_VAR, MID_VAR, ALL_VAR]
EXP_FAMILY = "lt_interventions_standard_v1"

# CausalChamber
from causalchamber.ground_truth import graph
from causalchamber.datasets import Dataset
DATA_DIR = "cc_data"

class CCTests(unittest.TestCase):

    def test_false_inputs(self):
        ccw = CCWrapper()
        with self.assertRaises(TypeError):
            ccw.set_exp_family(5)
        with self.assertRaises(ValueError):
            ccw.set_exp_family("Does_not_exist")

    def test_true_dag(self):
        for var in VAR:
            # Desired DAG
            desired_dag = graph(chamber=CHAMBER_CONFIGURATIONS[EXP_FAMILY]['chamber'], 
                                configuration=CHAMBER_CONFIGURATIONS[EXP_FAMILY]['configuration']).loc[var, var]
            # CCWrapper DAG
            ccw = CCWrapper()
            ccw.set_exp_family(EXP_FAMILY)
            ccw.set_variables(var)
            returned_dag = ccw.fetch_true_dag()
            # Assert 
            pd.testing.assert_frame_equal(desired_dag, returned_dag)

    def test_single_dataframe(self):
        csv_files = glob.glob(os.path.join(DATA_DIR,EXP_FAMILY, '*.csv'))
        ints = [30, 100, 200]
        for var in VAR:
            for file in csv_files:
                for nr in ints:
                    # Desired Df
                    desired_df = pd.read_csv(file).loc[:, var].iloc[0:nr, :]
                    # Returned Df
                    ccw = CCWrapper()
                    ccw.set_exp_family(EXP_FAMILY)
                    ccw.set_variables(var)
                    returned_df = ccw.fetch_experiments([os.path.splitext(os.path.basename(file))[0]], sizes=[nr])[0]
                    # Assert 
                    pd.testing.assert_frame_equal(desired_df, returned_df)
                    self.assertEqual(returned_df.shape, (nr, len(var)))
            
    def test_multiple_dataframes(self):
        df1 = pd.read_csv('cc_data/lt_interventions_standard_v1/uniform_reference.csv').loc[:, MID_VAR]
        df2 = pd.read_csv('cc_data/lt_interventions_standard_v1/uniform_green_mid.csv').loc[:, MID_VAR]
        df3 = pd.read_csv('cc_data/lt_interventions_standard_v1/uniform_blue_strong.csv').loc[:, MID_VAR]
        dfs_desired = [df1, df2, df3]

        df1_subset = pd.read_csv('cc_data/lt_interventions_standard_v1/uniform_reference.csv').loc[:, MID_VAR].iloc[0:1000, :]
        df2_subset = pd.read_csv('cc_data/lt_interventions_standard_v1/uniform_green_mid.csv').loc[:, MID_VAR].iloc[0:300, :]
        df3_subset = pd.read_csv('cc_data/lt_interventions_standard_v1/uniform_blue_strong.csv').loc[:, MID_VAR].iloc[0:500, :]
        dfs_subset_desired = [df1_subset, df2_subset, df3_subset]

        # Full dataset test
        ccw = CCWrapper()
        ccw.set_exp_family(EXP_FAMILY)
        ccw.set_variables(MID_VAR)
        dfs_wrapper = ccw.fetch_experiments(experiments=['uniform_reference', 'uniform_green_mid', 'uniform_blue_strong'])
        for df_wrapper, df_desired in zip(dfs_wrapper, dfs_desired):
            pd.testing.assert_frame_equal(df_wrapper, df_desired)

        # Subset dataset test
        dfs_wrapper_subset = ccw.fetch_experiments(experiments=['uniform_reference', 'uniform_green_mid', 'uniform_blue_strong'],
                              sizes=[1000, 300, 500])
        for df_subset_wrapper, df_subset_desired in zip(dfs_wrapper_subset, dfs_subset_desired):
            pd.testing.assert_frame_equal(df_subset_wrapper, df_subset_desired)


if __name__ == '__main__': 
    unittest.main()
