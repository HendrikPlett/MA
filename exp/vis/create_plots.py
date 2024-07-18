import os
import sys
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src'))
dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_path)
sys.path.append(dir_path)
from causalbenchmark.compute import BootstrapComparison
from causalbenchmark.visualize import VisBootstrapComparison
from cc_wrapper import POSITIONS_LT, LATEX_NAME 

RESULT_DIR = "results"
THIS_FILE_DIR = os.path.dirname(__file__)


def main():
    # Retrieve all pickle files
    pickle_dir = os.path.join(os.path.dirname(THIS_FILE_DIR), RESULT_DIR)
    pickle_files = [os.path.join(pickle_dir, file) for file in os.listdir(pickle_dir) if file.endswith('.pkl')]

    # Create plot for each pickle file
    for file in pickle_files:
        bstrpcomp = BootstrapComparison.unpickle(file)
        bstrpcompvis = VisBootstrapComparison(
            bstrp_comp=bstrpcomp, 
            pos=POSITIONS_LT, 
            latex_transf = LATEX_NAME
        )
        bstrpcompvis.pair_comp_plot()
        # Get name of pkl file without ending
        name_with_ending = os.path.basename(file)
        name_without_ending, _ = os.path.splitext(name_with_ending)
        SAVE_PATH = os.path.join(THIS_FILE_DIR, f"{name_without_ending}.pdf")
        if os.path.exists(SAVE_PATH):
            print(f"Warning: Plot {SAVE_PATH} exists; will be overwritten!")
        bstrpcompvis.save_fig(SAVE_PATH)

if __name__=="__main__":
    main()
