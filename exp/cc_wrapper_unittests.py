import unittest 
import pandas as pd
import os
import glob


# Wrapper
from cc_wrapper import CCWrapper
from cc_wrapper import SMALL_VAR, MID_VAR, ALL_VAR
from cc_wrapper import CHAMBER_CONFIGURATIONS
VAR = [SMALL_VAR, MID_VAR, ALL_VAR]
DATASET = "lt_interventions_standard_v1"

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
            desired_dag = graph(chamber=CHAMBER_CONFIGURATIONS[DATASET]['chamber'], 
                                configuration=CHAMBER_CONFIGURATIONS[DATASET]['configuration']).loc[var, var]
            # CCWrapper DAG
            ccw = CCWrapper()
            ccw.set_exp_family(DATASET)
            ccw.set_variables(var)
            returned_dag = ccw.fetch_true_dag()
            # Assert 
            pd.testing.assert_frame_equal(desired_dag, returned_dag)

    def test_dataframes(self):
        #Dataset(name=DATASET, root=DATA_DIR, download=False)
        csv_files = glob.glob(os.path.join(DATA_DIR,DATASET, '*.csv'))
        ints = [30, 100, 200]
        for var in VAR:
            for file in csv_files:
                for nr in ints:
                    # Desired Df
                    desired_df = pd.read_csv(file).loc[:, var].iloc[0:nr, :]
                    # Returned Df
                    ccw = CCWrapper()
                    ccw.set_exp_family(DATASET)
                    ccw.set_variables(var)
                    returned_df = ccw.fetch_experiments([os.path.splitext(os.path.basename(file))[0]], sizes=[nr])[0]
                    # Assert 
                    pd.testing.assert_frame_equal(desired_df, returned_df)
                    self.assertEqual(returned_df.shape, (nr, len(var)))
            


if __name__ == '__main__':
    unittest.main()
