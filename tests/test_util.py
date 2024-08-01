import sys
sys.path.append("../src/") # Path to causalbenchmark package
import unittest
import time
import pandas as pd
from unittest.mock import patch

from causalbenchmark.util import *

class TestDecorators(unittest.TestCase):

    def test_measure_time(self):
        @measure_time
        def sleep_function(): # Some simple test function
            time.sleep(0.1)
            return 10
        result, runtime = sleep_function()
        self.assertEqual(result, 10)
        self.assertTrue(runtime >= 0.1)
    

class TestListOperations(unittest.TestCase):

    def test_has_duplicates(self):
        # Test on simple made up lists
        self.assertTrue(has_duplicates([1, 2, 3, 2]))
        self.assertTrue(has_duplicates([1, 1, 1, 1]))
        self.assertFalse(has_duplicates([1, 2, 3, 4]))
        self.assertFalse(has_duplicates(['a', 'b', 3, 4]))
        self.assertTrue(has_duplicates(['a', 'b', 'c', 'a', 'c']))
        self.assertTrue(has_duplicates(['a', 'b', 'c', 'd', 'd']))
        self.assertFalse(has_duplicates([]))

    def test_enforce_no_duplicates(self):
        enforce_no_duplicates([[1, 2, 3, 4, 5, 6], [4, 5, 6]]) # Should throw no exception
        enforce_no_duplicates([['a', 'b'], [4, 5, 6]]) # Should throw no exception
        with self.assertRaises(ValueError):
            enforce_no_duplicates([[1, 2, 3], [4, 5, 5]]) # Dublicates -> Should throw ValueError
        with self.assertRaises(ValueError):
            enforce_no_duplicates([[1, 2, 3, 4, 5, 1, 2], [4, 5, 6]])

    def test_give_superlist(self):
        self.assertEqual(give_superlist([1, 2, 3, 4], [1, 2, 3]), [1, 2, 3, 4])
        self.assertEqual(give_superlist(['a', 'b'], ['a', 'b', 'c']), ['a', 'b', 'c'])
        self.assertEqual(give_superlist([2, 3], [1, 2, 3, 4]), [1, 2, 3, 4])
        self.assertEqual(give_superlist([], [1, 2, 3, 4]), [1, 2, 3, 4]) # Empty list case
        self.assertEqual(give_superlist([1, 2, 3], [1, 2, 3]), [1, 2, 3])
        with self.assertRaises(ValueError):
            give_superlist([1, 2, 2], [1, 2, 2, 3]) # Dublicates -> ValueError
        with self.assertRaises(ValueError):
            give_superlist([1, 2, 3], [3, 4, 5]) # No superlist exists -> ValueError

    def test_give_sublist(self):
        # Same lists as in test_give_superlist, only that the other list is asserted to be returned
        self.assertEqual(give_superlist(['a', 'b'], ['a', 'b', 'c']), ['a', 'b', 'c'])
        self.assertEqual(give_sublist([1, 2, 3, 4], [2, 3]), [2, 3])
        self.assertEqual(give_sublist([2, 3], [1, 2, 3, 4]), [2, 3])
        self.assertEqual(give_sublist([], [1, 2, 3, 4]), [])
        self.assertEqual(give_superlist([1, 2, 3], [1, 2, 3]), [1, 2, 3])
        with self.assertRaises(ValueError):
            give_superlist([1, 2, 2], [1, 2, 2, 3]) # Dublicates -> ValueError
        with self.assertRaises(ValueError):
            give_superlist([1, 2, 3], [3, 4, 5]) # No sublist exists -> ValueError

    def test_variables_increase(self):
        self.assertTrue(variables_increase([1, 2], [1, 2, 3, 4]))
        self.assertTrue(variables_increase(['a', 'b'], ['a', 'b', 'c']))
        self.assertFalse(variables_increase([1, 2, 3], [1, 2]))
        self.assertTrue(variables_increase([], [1]))
        self.assertTrue(variables_increase([1, 2], [1, 2]))
        with self.assertRaises(ValueError):
            variables_increase([1, 2, 3], [4, 5, 6])
        with self.assertRaises(ValueError):
            variables_increase([1, 2, 2], [1, 2, 2, 3])

    def test_same_order(self):
        self.assertTrue(same_order([1, 2, 3], [1, 2, 3, 4, 5]))
        self.assertTrue(same_order([1, 2, 5], [1, 2, 4, 5]))
        self.assertFalse(same_order([1, 2, 3], [3, 2, 1, 4, 5]))
        self.assertTrue(same_order([], [1, 2, 3]))
        with self.assertRaises(ValueError):
            same_order([1, 2, 3], [4, 5, 6])


class TestDataFrameOperations(unittest.TestCase):

    def setUp(self):
        self.df1 = pd.DataFrame({
            'X': [1, 2, 3, 4],
            'Y': [101, 102, 103, 104]
        })
        self.df2 = pd.DataFrame({
            'X': [5, 6, 7],
            'Y': [105, 106, 107]
        })
        self.df3 = pd.DataFrame({
            'X': [8, 9],
            'Y': [108, 109]
        })
        self.df4 = pd.DataFrame({
            'X': [1.3, 1.9, 14],
            'Y': [0.7, 0.4, 4.1]
        })
        self.df_diff_columns = pd.DataFrame({
            'X': [1, 2, 3],
            'Z': [101, 102, 103] # Different column name
        })

    def test_same_columns(self):
        self.assertTrue(same_columns([self.df1, self.df2, self.df3]))
        self.assertFalse(same_columns([self.df1, self.df_diff_columns]))
        with self.assertRaises(ValueError):
            same_columns([])

    def test_pool_dfs(self):
        pooled_dfs = pool_dfs([self.df1, self.df2, self.df3, self.df4])
        expected_pooled_dfs = pd.DataFrame({
            'X': [1,2,3,4,5,6,7,8,9,1.3,1.9,14],
            'Y': [101,102,103,104,105,106,107,108,109,0.7,0.4,4.1]
        })
        pd.testing.assert_frame_equal(pooled_dfs, expected_pooled_dfs)

        pooled_dfs2 = pool_dfs([self.df1])
        pd.testing.assert_frame_equal(pooled_dfs2, self.df1)

        with self.assertRaises(ValueError):
            pool_dfs([self.df1, self.df_diff_columns])

    def test_bootstrap_sample(self):
        # Test fraction bootstrap
        sample_size_fraction = [0.25, 1/3, 1]
        dfs = [self.df1, self.df2, self.df3]
        bstr_sample1 = bootstrap_sample(dfs, sample_size_fraction, seed=1)
        self.assertTrue(len(dfs)==len(bstr_sample1))
        for df_sample, sample_size, df_orig in zip(bstr_sample1, sample_size_fraction, dfs):
            self.assertEqual(len(df_sample), int(len(df_orig) * sample_size))
            self.assertTrue(same_columns([df_sample, df_orig]))

        # Test absolute number bootstrap
        sample_size_integer = [3, 2, 2] 
        bstr_sample2 = bootstrap_sample(dfs, sample_size_integer, seed=1)
        self.assertTrue(len(dfs)==len(bstr_sample2))
        for df_sample, sample_size, df_original in zip(bstr_sample2, sample_size_integer, dfs):
            self.assertEqual(len(df_sample), sample_size)
            self.assertTrue(same_columns([df_sample, df_original]))

        false_sample_size = [-1, 0.5, 2]
        with self.assertRaises(ValueError):
            bootstrap_sample([self.df1, self.df2, self.df3], false_sample_size, seed=1)

        with self.assertRaises(AssertionError):
            bootstrap_sample([self.df1, self.df2], [0.5, 0.5, 0.5], seed=1)

    def test_standardize_dfs(self):
        original_dfs = [self.df1, self.df2, self.df3, self.df4]
        stand_dfs = standardize_dfs(original_dfs)
        
        # Check output format
        self.assertIsInstance(stand_dfs, list)
        self.assertEqual(len(stand_dfs), 4)
        
        # Test for correct shape, mean, sd, and correlation
        for orig_df, stand_df in zip(original_dfs, stand_dfs):
            # Shape
            self.assertIsInstance(stand_df, pd.DataFrame)
            self.assertEqual(stand_df.shape, orig_df.shape)
            self.assertEqual(list(stand_df.columns), list(orig_df.columns))
            # Mean, sd
            for col in stand_df.columns:
                self.assertAlmostEqual(stand_df[col].mean(), 0, places=5)
                self.assertAlmostEqual(stand_df[col].std(), 1, places=5)
            
            # Correlation must have remained the same
            pd.testing.assert_frame_equal(orig_df.corr(), stand_df.corr())
        
        # Ensure columns remain the same if different columns are passed in dfs
        dfs_stand_diff = standardize_dfs([self.df4, self.df_diff_columns])
        self.assertEqual(list(dfs_stand_diff[0].columns), list(self.df4.columns))
        self.assertEqual(list(dfs_stand_diff[1].columns), list(self.df_diff_columns.columns))
        
        # Empty input should just do nothing
        self.assertEqual(standardize_dfs([]), [])
        
        # Case of a single dataframe
        df_stand_single = standardize_dfs([self.df4])
        self.assertEqual(len(df_stand_single), 1)
        self.assertAlmostEqual(df_stand_single[0]['X'].mean(), 0, places=5)
        self.assertAlmostEqual(df_stand_single[0]['X'].std(), 1, places=5)
        pd.testing.assert_frame_equal(self.df4.corr(), df_stand_single[0].corr())
        
        # Ensure that original data is unchanged and new data is created
        original_df1 = self.df1.copy()
        standardize_dfs([self.df1])
        pd.testing.assert_frame_equal(self.df1, original_df1)


class TestAdjacencyMatrixDataFrameOperations(unittest.TestCase):
    
    def setUp(self):
        self.valid_df1 = pd.DataFrame(
            [[1, 0, 0], [0, 1, 1], [0, 1, 1]],
            index=['a', 'b', 'c'],
            columns=['a', 'b', 'c']
        )
        self.valid_df2 = pd.DataFrame(
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
            index=['a', 'a'], # Same index names
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
        self.assertTrue(is_sub_adj_mat(self.valid_df1, self.valid_df2))
        self.assertFalse(is_sub_adj_mat(self.valid_df2, self.valid_df1))
        with self.assertRaises(ValueError):
            is_sub_adj_mat(self.df_invalid1, self.valid_df2)
        with self.assertRaises(ValueError):
            is_sub_adj_mat(self.df_invalid2, self.valid_df2)

    def test_enforce_sub_adj_mat(self):
        enforce_sub_adj_mat(self.valid_df1, self.valid_df2)
        with self.assertRaises(ValueError):
            enforce_sub_adj_mat(self.valid_df2, self.valid_df1)
        with self.assertRaises(ValueError):
            enforce_sub_adj_mat(self.df_invalid1, self.valid_df2)
        with self.assertRaises(ValueError):
            enforce_sub_adj_mat(self.df_invalid2, self.valid_df2)

    def test_reduce_to_size(self):
        reduced = reduce_to_size(self.valid_df2, self.valid_df1)
        expected_reduced = pd.DataFrame(
            [[1, 0, 0], [0, 1, 1], [0, 1, 1]],
            index=['a', 'b', 'c'],
            columns=['a', 'b', 'c']
        )
        pd.testing.assert_frame_equal(reduced, expected_reduced)
        with self.assertRaises(ValueError):
            reduce_to_size(self.valid_df1, self.valid_df2)

    def test_pad_zeros_to_size(self):
        padded = pad_zeros_to_size(self.valid_df1, self.valid_df2)
        expected_padded = pd.DataFrame(
            [[1, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
            index=['a', 'b', 'c', 'd'],
            columns=['a', 'b', 'c', 'd']
        )
        pd.testing.assert_frame_equal(padded, expected_padded)
        with self.assertRaises(ValueError):
            pad_zeros_to_size(self.valid_df2, self.valid_df1)

    def test_enforce_valid_bstr_adj_mat(self):
        enforce_valid_bstr_adj_mat(self.valid_df1)
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
        enforce_valid_adj_mat(self.valid_df1)
        enforce_valid_adj_mat(self.valid_df2)
        with self.assertRaises(ValueError):
            enforce_valid_adj_mat(self.df_invalid1)
        with self.assertRaises(ValueError):
            enforce_valid_adj_mat(self.df_invalid2)


if __name__ == '__main__':
    unittest.main()
