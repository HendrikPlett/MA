import sys
sys.path.append("../src/") # Path to causalbenchmark package
import unittest
from unittest.mock import patch, Mock
from functools import wraps
import numpy as np
import pandas as pd

from causalbenchmark.compute.algorithms import Algorithm, PC, UT_IGSP, GES, GIES, GNIES, NoTears, Golem, VarSortRegress, R2SortRegress, ICP


class TestPDAGTransform(unittest.TestCase):
    """
    Test whether the outputs of the third party algorithm implemenation are
        postprocessed correctly.
    """
    def setUp(self):
        df1 = pd.DataFrame({'A': [1, 2, 5], 
                            'B': [3, 7, 15], 
                            'C': [7, 8, 9]
                            })
        df2 = pd.DataFrame({'A': [10, 11, 17], 
                            'B': [8, 14, 4], 
                            'C': [17, 17, 26]})
        self.data = [df1, df2]
        self.variables = df1.columns

        # Output requires no transformation
        self.INTERNAL_OUTPUT1 = np.array([[0, 0, 1], [0, 0, 1], [1, 0, 0]])
        self.DESIRED_OUTPUT1 = self.INTERNAL_OUTPUT1.copy()

        # Output must be transformed into binary (and potentially a CPDAG)
        self.INTERNAL_OUTPUT2 = np.array([[0, 0.3, 0], [0, 0, -13], [0, 0, 0]])
        self.DESIRED_OUTPUT2 = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        self.DESIRED_OUTPUT2_CPDAG = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

        # Output must be transformed into binary and into a valid DAG via postprocessing
        self.INTERNAL_OUTPUT3 = np.array([[0, 0.5, 0.1], [0, 0, -13], [0, 0, 0]])
        self.DESIRED_OUTPUT3 = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        self.DESIRED_OUTPUT3_CPDAG = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])


    def check_output_format(self, result, runtime):
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIsInstance(runtime, float)
        self.assertEqual(result.shape, (3, 3))
        self.assertEqual(list(result.columns), ['A', 'B','C'])
        self.assertEqual(list(result.index), ['A', 'B','C'])

    def run_test(self, alg: Algorithm, desired_result: np.ndarray):
        result, runtime = alg.fit(self.data)
        self.check_output_format(result, runtime)
        pd.testing.assert_frame_equal(result, 
                                      pd.DataFrame(desired_result, columns=self.variables, index=self.variables),
                                      check_dtype=False
                                      )


    @patch('causalbenchmark.compute.algorithms.pc')
    def test_PC_algorithm(self, mock_pc):
        # Set up mock
        pc_return = Mock()
        pc_return.G.graph = np.array([[0, -1, 1], 
                                      [-1, 0, 1], 
                                      [1, -1, 0]])
        mock_pc.return_value = pc_return
        # Run mocked algorithm 
        self.run_test(alg=PC(alpha=0.05),
                      desired_result=np.array([[0, 1, 1], 
                                               [1, 0, 0], 
                                               [1, 1, 0]])
                    )
    
    @patch('causalbenchmark.compute.algorithms.ut_igsp.fit')
    def test_UT_IGSP_algorithm(self, mock_fit):
        # Set up mock       
        mock_fit.return_value = (self.INTERNAL_OUTPUT1, None, None)
        # Run
        self.run_test(alg=UT_IGSP(alpha_ci=0.05, alpha_inv=0.05),
                      desired_result=self.DESIRED_OUTPUT1)

    @patch('causalbenchmark.compute.algorithms.ges.fit_bic')
    def test_GES_algorithm(self, mock_fit_bic):
        # Set up mock
        mock_fit_bic.return_value = (self.INTERNAL_OUTPUT1, None)
        # Run
        self.run_test(alg=GES(),
                      desired_result=self.DESIRED_OUTPUT1)

    @patch('causalbenchmark.compute.algorithms.gies.fit_bic')
    def test_GIES_algorithm(self, mock_fit_bic):
        mock_fit_bic.return_value = (self.INTERNAL_OUTPUT1, None)
        self.run_test(alg=GIES(interventions=[['A'], ['B']]),
                       desired_result=self.DESIRED_OUTPUT1)

    @patch('causalbenchmark.compute.algorithms.gnies.fit')
    def test_GNIES_algorithm(self, mock_fit):
        mock_fit.return_value = (None, self.INTERNAL_OUTPUT1, None)
        self.run_test(alg=GNIES(),
                      desired_result=self.DESIRED_OUTPUT1)

    @patch('causalbenchmark.compute.algorithms.notears_linear')
    def test_NoTEARS_algorithm_dag(self, mock_notears_linear):
        mock_notears_linear.return_value = self.INTERNAL_OUTPUT2
        self.run_test(alg=NoTears(return_cpdag=False),
                      desired_result=self.DESIRED_OUTPUT2)
    
    @patch('causalbenchmark.compute.algorithms.notears_linear')
    def test_NoTEARS_algorithm_cpdag(self, mock_notears_linear):
        mock_notears_linear.return_value = self.INTERNAL_OUTPUT2
        self.run_test(alg=NoTears(return_cpdag=True),
                      desired_result=self.DESIRED_OUTPUT2_CPDAG)

    @patch('causalbenchmark.compute.algorithms.fit_golem')
    def test_GOLEM_algorithm_EV_dag(self, mock_fit_golem):
        mock_fit_golem.return_value = self.INTERNAL_OUTPUT3
        self.run_test(alg=Golem(equal_variances=True, return_cpdag=False), 
                      desired_result=self.DESIRED_OUTPUT3)
        
    @patch('causalbenchmark.compute.algorithms.fit_golem')
    def test_GOLEM_algorithm_NV_dag(self, mock_fit_golem):
        mock_fit_golem.return_value = self.INTERNAL_OUTPUT3
        self.run_test(alg=Golem(equal_variances=False, return_cpdag=False), 
                      desired_result=self.DESIRED_OUTPUT3)

    @patch('causalbenchmark.compute.algorithms.fit_golem')
    def test_GOLEM_algorithm_EV_cpdag(self, mock_fit_golem):
        mock_fit_golem.return_value = self.INTERNAL_OUTPUT3
        self.run_test(alg=Golem(equal_variances=True, return_cpdag=True), 
                      desired_result=self.DESIRED_OUTPUT3_CPDAG)
        
    @patch('causalbenchmark.compute.algorithms.fit_golem')
    def test_GOLEM_algorithm_NV_cpdag(self, mock_fit_golem):
        mock_fit_golem.return_value = self.INTERNAL_OUTPUT3
        self.run_test(alg=Golem(equal_variances=False, return_cpdag=True), 
                      desired_result=self.DESIRED_OUTPUT3_CPDAG)
    
    @patch('causalbenchmark.compute.algorithms.var_sort_regress')
    def test_VarSortRegress_algorithm(self, mock_var_sort_regress):
        mock_var_sort_regress.return_value = self.INTERNAL_OUTPUT2
        self.run_test(alg=VarSortRegress(),
                      desired_result=self.DESIRED_OUTPUT2)

    @patch('causalbenchmark.compute.algorithms.r2_sort_regress')
    def test_R2SortRegress_algorithm(self, mock_r2_sort_regress):
        mock_r2_sort_regress.return_value = self.INTERNAL_OUTPUT2
        self.run_test(alg=R2SortRegress(),
                      desired_result=self.DESIRED_OUTPUT2)

    @patch('causalbenchmark.compute.algorithms.causalicp.fit')
    def test_ICP_algorithm(self, mock_fit):
        mock_fit_return = Mock()
        mock_fit_return.estimate = {0, 1} # Representing variable with index 0 and 1 as estimated parents
        mock_fit.return_value = mock_fit_return
        dr = np.array([[0,0,1],
                       [0,0,1],
                       [0,0,0]])
        self.run_test(alg=ICP(target='C'),
                      desired_result=dr)

    @patch('causalbenchmark.compute.algorithms.causalicp.fit')
    def test_ICP_algorithm(self, mock_fit):
        mock_fit_return = Mock()
        mock_fit_return.estimate = set() # Mocks the case when ICP finds no parents of variable 'C'
        mock_fit.return_value = mock_fit_return
        dr = np.array([[0,0,0],
                       [0,0,0],
                       [0,0,0]])
        self.run_test(alg=ICP(target='C'),
                      desired_result=dr)

"""
class TestFullRun(unittest.TestCase):
    
    
    def setUp():
        # Create TrueDAG and data
        ... 

    def check_output_format():
        ...    
        
    def test_PC_algorithm():
        ...

    def test_UT_IGSP_algorithm():
        ...

    def test_GES_algorithm():
        ...

    def test_GIES_algorithm():
        ...

    def test_GNIES_algorithm():
        ...

    def test_NoTEARS_algorithm_dag():
        ...

    def test_NoTEARS_algorithm_cpdag():
        ...

    def test_GOLEM_algorithm_dag():
        ...

    def test_GOLEM_algorithm_cpdag():
        ...

    def test_VarSortRegress_algorithm():
        ...

    def test_R2SortRegress_algorithm():
        ...
"""


if __name__ == '__main__':
    unittest.main()