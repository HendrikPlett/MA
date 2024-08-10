from exp_assistant import (
    # Precreated datasets/true DAGs
    MID_VAR_TRUE_DAG,
    MID_VAR_UNIFORM_REFERENCE,
    # Other useful constants
    NR_BOOTSTRAPS,
    DEFAULT_DATA_SIZE,
    # Predefined benchmarking functions
    increase_obs_data_small_var,
    increase_obs_data_mid_var,
    increase_variables,
    increase_colors,
    standardized_data_comparison,
    # Benchmarking classes
    BootstrapComparison,
    Bootstrap,
    # Algorithm 
    GES, NoTears, Golem, VarSortRegress, R2SortRegress, Algorithm
)

PROCESSES = 20

def compare_all(algorithms: list[Algorithm], names: list[str],
                standardize_data: bool = False, processes: int = PROCESSES):
    postfix = "StandScale" if standardize_data else "OrigScale"
    nm = "-".join([alg.__class__.__name__ for alg in algorithms])
    bstrpcomp = BootstrapComparison(name=f"{nm}-Comparison-{postfix}")
    for alg, name in zip(algorithms, names):
        bstrpcomp.add_bootstrap(
            Bootstrap(
                name=name,
                true_dag=MID_VAR_TRUE_DAG,
                algorithm=alg,
                data_to_bootstrap_from=MID_VAR_UNIFORM_REFERENCE,
                sample_sizes=DEFAULT_DATA_SIZE,
                standardize_data=standardize_data,
                nr_bootstraps=NR_BOOTSTRAPS,
                PROCESSES=processes
            )
        )
    bstrpcomp.run_comparison()
    bstrpcomp.pickle()



if __name__ == "__main__":
    algorithms=[
        GES(),
        NoTears(return_cpdag=False, lambda1=81),
        Golem(equal_variances=False),
        Golem(equal_variances=True),
        VarSortRegress(),
        R2SortRegress()
    ]
    names=[
        "GES",
        "NoTears",
        "Golem-EV",
        "Golem-NV",
        "VarSortRegress",
        "R2SortRegress"
    ]
    compare_all(
        algorithms=algorithms,
        names=names,
        standardize_data=False
    )
    compare_all(
        algorithms=algorithms,
        names=names,
        standardize_data=True
    )
