import sys
sys.path.append("../src/") # Path to causalbenchmark package
import unittest
import pandas as pd

from causalbenchmark.util import *

class TestListOperations(unittest.TestCase):

    def test_has_duplicates(self):
        self.assertTrue(has_duplicates([1, 2, 3, 2]))
        self.assertFalse(has_duplicates([1, 2, 3, 4]))
        self.assertTrue(has_duplicates(['a', 'b', 'c', 'a']))
        self.assertFalse(has_duplicates([]))

    def test_enforce_no_duplicates(self):
        enforce_no_duplicates([[1, 2, 3], [4, 5, 6]]) # No exception
        with self.assertRaises(ValueError):
            enforce_no_duplicates([[1, 2, 3], [4, 5, 5]])

    def test_give_superlist(self):
        self.assertEqual(give_superlist([1, 2, 3, 4], [2, 3]), [1, 2, 3, 4])
        self.assertEqual(give_superlist([2, 3], [1, 2, 3, 4]), [1, 2, 3, 4])
        self.assertEqual(give_superlist([], [1, 2, 3, 4]), [1, 2, 3, 4])
        self.assertEqual(give_superlist([1, 2, 3], [1, 2, 3]), [1, 2, 3])
        with self.assertRaises(ValueError):
            give_superlist([1, 2, 3], [3, 4, 5])

    def test_give_sublist(self):
        self.assertEqual(give_sublist([1, 2, 3, 4], [2, 3]), [2, 3])
        self.assertEqual(give_sublist([2, 3], [1, 2, 3, 4]), [2, 3])
        self.assertEqual(give_sublist([], [1, 2, 3, 4]), [])
        self.assertEqual(give_superlist([1, 2, 3], [1, 2, 3]), [1, 2, 3])
        with self.assertRaises(ValueError):
            give_sublist([1, 2, 3], [3, 4, 5])

    def test_variables_increase(self):
        self.assertTrue(variables_increase([1, 2], [1, 2, 3, 4]))
        self.assertFalse(variables_increase([1, 2, 3], [1, 2]))
        self.assertTrue(variables_increase([], [1, 2, 3]))
        self.assertTrue(variables_increase([1, 2, 3], [1, 2, 3]))
        with self.assertRaises(ValueError):
            variables_increase([1, 2, 3], [4, 5, 6])

    def test_same_order(self):
        self.assertTrue(same_order([1, 2, 3], [1, 2, 3, 4, 5]))
        self.assertFalse(same_order([1, 2, 3], [3, 2, 1, 4, 5]))
        self.assertTrue(same_order([], [1, 2, 3]))
        with self.assertRaises(ValueError):
            same_order([1, 2, 3], [4, 5, 6])


class TestDataFrameOperations(unittest.TestCase):

    def setUp(self):
        self.df1 = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        self.df2 = pd.DataFrame({
            'A': [7, 8, 9],
            'B': [10, 11, 12]
        })
        self.df3 = pd.DataFrame({
            'A': [13, 14, 15],
            'B': [16, 17, 18]
        })
        self.df_diff_columns = pd.DataFrame({
            'A': [1, 2, 3],
            'C': [4, 5, 6]
        })

    def test_same_columns(self):
        self.assertTrue(same_columns([self.df1, self.df2, self.df3]))
        self.assertFalse(same_columns([self.df1, self.df_diff_columns]))
        with self.assertRaises(ValueError):
            same_columns([])

    def test_pool_dfs(self):
        result1 = pool_dfs([self.df1, self.df2, self.df3])
        expected_result1 = pd.DataFrame({
            'A': [1, 2, 3, 7, 8, 9, 13, 14, 15],
            'B': [4, 5, 6, 10, 11, 12, 16, 17, 18]
        })
        pd.testing.assert_frame_equal(result1, expected_result1)

        result2 = pool_dfs([self.df1])
        expected_result2 = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        pd.testing.assert_frame_equal(result2, expected_result2)

        with self.assertRaises(ValueError):
            pool_dfs([self.df1, self.df_diff_columns])

    def test_bootstrap_sample(self):
        sample_size_fraction = [1/3, 2/3, 1]
        dfs = [self.df1, self.df2, self.df3]
        result = bootstrap_sample(dfs, sample_size_fraction, seed=1)
        for df_sample, sample_size, df_orig in zip(result, sample_size_fraction, dfs):
            self.assertEqual(len(df_sample), int(len(df_orig) * sample_size))

        sample_size_integer = [2, 2, 2]
        result = bootstrap_sample([self.df1, self.df2, self.df3], sample_size_integer, seed=1)
        for df, sample_size in zip(result, sample_size_integer):
            self.assertEqual(len(df), sample_size)

        invalid_sample_size = [-1, 0.5, 2]
        with self.assertRaises(ValueError):
            bootstrap_sample([self.df1, self.df2, self.df3], invalid_sample_size, seed=1)

        with self.assertRaises(AssertionError):
            bootstrap_sample([self.df1, self.df2], [0.5, 0.5, 0.5], seed=1)


class TestAdjacencyMatrixDataFrameOperations(unittest.TestCase):
    
    def setUp(self):
        self.df1 = pd.DataFrame(
            [[1, 0, 0], [0, 1, 1], [0, 1, 1]],
            index=['a', 'b', 'c'],
            columns=['a', 'b', 'c']
        )
        self.df2 = pd.DataFrame(
            [[1, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 1]],
            index=['a', 'b', 'c', 'd'],
            columns=['a', 'b', 'c', 'd']
        )
        self.df_invalid1 = pd.DataFrame(
            [[1, 0], [0, 1]],
            index=['a', 'b'],
            columns=['a', 'c']  # Different columns
        )
        self.df_invalid2 = pd.DataFrame(
            [[1, 0], [0, 1]],
            index=['a', 'a'], # Same names
            columns=['a', 'b']  
        )
        self.binary_df = pd.DataFrame(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            index=['a', 'b', 'c'],
            columns=['a', 'b', 'c']
        )
        self.outsidezeroone_df1 = pd.DataFrame(
            [[0.2, 1.5, 0], [0, 0.1, 0], [0, 0, 1]], # 1.5 not in [0.1]
            index=['a', 'b', 'c'],
            columns=['a', 'b', 'c']
        )
        self.outsidezeroone_df2 = pd.DataFrame(
            [[1, 0.2, 0], [0, 1, 0], [0, 0.2, -0.1]], # -0.1 not in [0.1]
            index=['a', 'b', 'c'],
            columns=['a', 'b', 'c']
        )
        

    def test_is_sub_adj_mat(self):
        self.assertTrue(is_sub_adj_mat(self.df1, self.df2))
        self.assertFalse(is_sub_adj_mat(self.df2, self.df1))
        with self.assertRaises(ValueError):
            is_sub_adj_mat(self.df_invalid1, self.df2)
        with self.assertRaises(ValueError):
            is_sub_adj_mat(self.df_invalid2, self.df2)

    def test_enforce_sub_adj_mat(self):
        enforce_sub_adj_mat(self.df1, self.df2)
        with self.assertRaises(ValueError):
            enforce_sub_adj_mat(self.df2, self.df1)
        with self.assertRaises(ValueError):
            enforce_sub_adj_mat(self.df_invalid1, self.df2)
        with self.assertRaises(ValueError):
            enforce_sub_adj_mat(self.df_invalid2, self.df2)

    def test_reduce_to_size(self):
        reduced = reduce_to_size(self.df2, self.df1)
        expected_reduced = pd.DataFrame(
            [[1, 0, 0], [0, 1, 1], [0, 1, 1]],
            index=['a', 'b', 'c'],
            columns=['a', 'b', 'c']
        )
        pd.testing.assert_frame_equal(reduced, expected_reduced)
        with self.assertRaises(ValueError):
            reduce_to_size(self.df1, self.df2)

    def test_pad_zeros_to_size(self):
        padded = pad_zeros_to_size(self.df1, self.df2)
        expected_padded = pd.DataFrame(
            [[1, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
            index=['a', 'b', 'c', 'd'],
            columns=['a', 'b', 'c', 'd']
        )
        pd.testing.assert_frame_equal(padded, expected_padded)
        with self.assertRaises(ValueError):
            pad_zeros_to_size(self.df2, self.df1)

    def test_enforce_valid_bstr_adj_mat(self):
        enforce_valid_bstr_adj_mat(self.df1)
        with self.assertRaises(ValueError):
            enforce_valid_bstr_adj_mat(self.outsidezeroone_df1)
        with self.assertRaises(ValueError):
            enforce_binary_adj_mat(self.df_invalid1)
        with self.assertRaises(ValueError):
            enforce_binary_adj_mat(self.df_invalid2)

    def test_enforce_binary_adj_mat(self):
        enforce_binary_adj_mat(self.binary_df)
        with self.assertRaises(ValueError):
            enforce_binary_adj_mat(self.outsidezeroone_df1)
        with self.assertRaises(ValueError):
            enforce_binary_adj_mat(self.outsidezeroone_df2)

    def test_enforce_valid_adj_mat(self):
        enforce_valid_adj_mat(self.df1)
        enforce_valid_adj_mat(self.df2)
        with self.assertRaises(ValueError):
            enforce_valid_adj_mat(self.df_invalid1)
        with self.assertRaises(ValueError):
            enforce_valid_adj_mat(self.df_invalid2)


if __name__ == '__main__':
    unittest.main()
