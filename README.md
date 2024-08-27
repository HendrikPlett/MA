# Master Thesis Hendrik Plett

This repository contains the code written for my Master Thesis at ETH Zurich in summer 2024. 

## Package `causalbenchmark`

Under `src/`, we provide the 'causalbenchmark' package. It allows to estimate and visualize edge retrieval probabilites of various causal discovery algorithms. It contains the two subpackages 'compute' and 'visualize' and provides the module `utils.py` which defines several helper functions. 

## Subpackage `causalbenchmark.compute`

The 'compute' subpackage implements the bootstrap estimator for the Edge Probability Matrix (EPM). It contains the following modules:

- `algorithms.py` Contains wrapper classes for various third party algorithm implementations and unifies the API for downstream analysis. Below, you find references to the third party implementations we use:
    - PC: [`causal-learn package`](https://github.com/py-why/causal-learn)
    - UT-IGSP: [`causaldag package`](https://github.com/uhlerlab/causaldag)
    - GES: [`ges package`](https://github.com/juangamella/ges)
    - GIES: [`gies package`](https://github.com/juangamella/gies)
    - GnIES: [`gnies package`](https://github.com/juangamella/gnies)
    - NoTEARS: [`notears package`](https://github.com/xunzheng/notears)
    - Golem: [`golem repository`](https://github.com/ignavierng/golem)
    - ICP: [`icp package`](https://github.com/juangamella/icp)
    - Var/R2-SortnRegress: [`CausalDisco package`](https://github.com/CausalDisco/CausalDisco)
- `causal_inference_task.py` Implements CausalInferenceTask class which takes an algorithm, an data and a true DAG. Then, it computes sortability metrics for the data, runs the algorithm and computes the average consistent extension for the returned PDAG. 
- `bootstrap.py` Implements the Bootstrap class which takes an algorithm, data and a true DAG and creates a desired number of CausalInferenceTask instances by bootstrapping the passed data. Also, the module implements the BootstrapComparison class which keeps track of multiple Bootstrap instances. 
- `savable.py` Implement the Savable class which provides functionality for pickling and unpickling. 
- `ut_igsp.py` Wrapper for the UT-IGSP algorithm, copied from [`here`](https://github.com/juangamella/gnies-paper/blob/master/src/ut_igsp.py). 

### Subpackage `causalbenchmark.visualize`

The 'visualize' subpackage allows to visualize instances of the Bootstrap and BootstrapComparison class. It contains the following modules: 

- `helper.py` Defines util class AdjGraphs which organizes and verifies the two EPMs we want to compare plus the true DAG.
- `edgelogic.py` Defines class EdgeLogic which provides basic logic to retrieve different edge categories (True Positives, False Positives, ...).
- `edges.py` Defines class Edges which takes instances of EdgeLogic and AdjGraphs and provides functionality to compute the desired edges. Also, it can draw the edges onto a passed axes.
- `nodes.py` Defines class Nodes which computes desired nodes based on instance of AdjGraphs. Also, it can draw the nodes onto a passed axes. 
- `visbootstrap.py` Defines VisBootstrap and VisBootstrapComparison classes that visualize Bootstrap and BootstrapComparison instances by repeatedly using the Edges and Nodes classes. 

## Tests

Under `tests/`, we provide unittests for the two modules `util.py` and `algorithms.py`. 

## Experiments

Under `exp/`, we apply the `causalbenchmark` package to the [Light Tunnel dataset](https://github.com/juangamella/causal-chamber/tree/main/datasets/lt_interventions_standard_v1)