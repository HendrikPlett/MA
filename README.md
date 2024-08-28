# Master's Thesis Hendrik Plett

This repository contains the code for my Master's Thesis at ETH Zurich in the summer of 2024. 

## Package `causalbenchmark`

Under `src/`, we provide the `causalbenchmark` package. It allows the estimation and visualization of edge retrieval probabilities of various causal discovery algorithms. It contains the two subpackages `compute` and `visualize` and provides the module `utils.py` which defines several helper functions. 

### Subpackage `causalbenchmark.compute`

The `compute` subpackage implements the bootstrap estimator for the Edge Probability Matrix (EPM). It contains the following modules:

- `algorithms.py` Contains wrapper classes for various third-party algorithm implementations and unifies their API for downstream computations. Below, you will find references to the third-party implementations we use:
    - PC: [`causal-learn package`](https://github.com/py-why/causal-learn)
    - UT-IGSP: [`causaldag package`](https://github.com/uhlerlab/causaldag)
    - GES: [`ges package`](https://github.com/juangamella/ges)
    - GIES: [`gies package`](https://github.com/juangamella/gies)
    - GnIES: [`gnies package`](https://github.com/juangamella/gnies)
    - NoTEARS: [`notears package`](https://github.com/xunzheng/notears)
    - Golem: [`golem repository`](https://github.com/ignavierng/golem)
    - ICP: [`icp package`](https://github.com/juangamella/icp)
    - Var/R2-SortnRegress: [`CausalDisco package`](https://github.com/CausalDisco/CausalDisco)
- `causal_inference_task.py` Implements the CausalInferenceTask class which takes an algorithm, data and a true DAG as input. Then, it computes sortability metrics for the data, runs the algorithm and computes the average consistent extension for the returned PDAG. 
- `bootstrap.py` Implements the Bootstrap class which takes an algorithm, data and a true DAG and creates the desired number of CausalInferenceTask instances by bootstrapping the passed data. Also, the module implements the BootstrapComparison class which keeps track of multiple Bootstrap instances. 
- `savable.py` Implement the Pickable class which provides functionality for pickling and unpickling. 
- `ut_igsp.py` Wrapper for the UT-IGSP algorithm, copied from [`here`](https://github.com/juangamella/gnies-paper/blob/master/src/ut_igsp.py). 

### Subpackage `causalbenchmark.visualize`

The `visualize` subpackage allows to visualize instances of the Bootstrap and BootstrapComparison class. It contains the following modules: 

- `helper.py` Defines the class AdjGraphs which stores and verifies the two EPMs we want to compare plus the true DAG.
- `edgelogic.py` Defines the class EdgeLogic which provides basic logic to retrieve different edge categories (True Positives, False Positives, ...).
- `edges.py` Defines the class Edges which takes instances of EdgeLogic and AdjGraphs and provides functionality to compute the desired edges. Also, it can draw the edges onto a passed axis.
- `nodes.py` Defines the class Nodes that computes desired nodes based on an instance of AdjGraphs. Also, it can draw the nodes onto a passed axis. 
- `visbootstrap.py` Defines the VisBootstrap and VisBootstrapComparison classes that visualize Bootstrap and BootstrapComparison instances by repeatedly using the Edges and Nodes classes. 

## Tests

Under `tests/`, we provide unittests for the two modules `util.py` and `algorithms.py`. 

## Experiments

Under `exp/`, we apply the `causalbenchmark` package to the [Light Tunnel dataset](https://github.com/juangamella/causal-chamber/tree/main/datasets/lt_interventions_standard_v1). The directory itself contains modules designed to handle the Light Tunnel data:

- `cc_download.py`: Provides functionality to load a desired family of datasets to the local machine.
- `cc_ground_truth.py`: Contains ground truth DAGs for the Light Tunnel dataset. Copied from [here](https://github.com/juangamella/causal-chamber/blob/main/src/causalchamber/ground_truth.py). We need such a local copy because ETH EULER compute nodes do not have internet access. 
- `cc_vis.ipynb`: Creates several figures used in the Master's Thesis to describe the Light Tunnel dataset. 
- `cc_wrapper.py`: Provides a wrapper class that allows to retrieve a desired dataset with a specified shape. 
- `cc_wrapper_unittests.py`: Unittests for the module above.
- `euler_setup.sh`, `submit_jobs.sh`: Bash scripts to run the experiments defined under `exp/scripts/` on ETH's EULER cluster. 

In the subdirectory `exp/scripts/`, we define the actual benchmarking experiments. The module `exp_assistant.py` sets up datasets and global variables we will frequently need in the other modules. Typically, one module then deals with a single algorithm (i.e. `exp_ges.py`) or the comparison between a group of algorithms (i.e. `exp_score_based_comparison.py`). Besides, each module usually creates and runs multiple BootstrapComparison objects and saves the results as separate `.pkl` files under `exp/results/`. 

Under `exp/vis/`, the module `create_plots.py` will load all `.pkl` files from `exp/results/`. Then, it will use `compute.visualize` to visualize the respective Bootstrap and BootstrapComparison instances. 

Finally, the two bash scripts `run_exps_on_euler.sh` and `retrieve_results_and_plot.sh` in the home directory streamline the entire workflow on the ETH Euler Cluster. If you want to use them, you will have to adapt certain directory names to your local machine and your EULER account credentials. 