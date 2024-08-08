from exp_assistant import (
    # Precreated datasets/true DAGs
    MID_VAR_TRUE_DAG,
    MID_VAR_UNIFORM_REFERENCE,
    # Other useful constants
    CLUSTER_CPUS,
    NR_BOOTSTRAPS,
    DEFAULT_DATA_SIZE,
    # Predefined benchmarking functions
    increase_obs_data_small_var,
    increase_obs_data_mid_var,
    increase_variables,
    increase_colors,
    increase_hyperparameter,
    standardized_data_comparison,
    # Benchmarking classes
    BootstrapComparison,
    Bootstrap,
    # Algorithm 
    NoTears
)


def dag_versus_cpdag_comparison():
    bstrcomp = BootstrapComparison("NoTears-DAGversusCPDAG")
    bstrcomp.add_bootstrap(
        Bootstrap(
                name=f"Return DAG",
                true_dag=MID_VAR_TRUE_DAG,
                algorithm=NoTears(return_cpdag=False),
                data_to_bootstrap_from=MID_VAR_UNIFORM_REFERENCE,
                sample_sizes=DEFAULT_DATA_SIZE,
                nr_bootstraps=NR_BOOTSTRAPS, 
                CLUSTER_CPUS=CLUSTER_CPUS
        )
    )
    bstrcomp.add_bootstrap(
        Bootstrap(
                name=f"Return CPDAG",
                true_dag=MID_VAR_TRUE_DAG,
                algorithm=NoTears(return_cpdag=True),
                data_to_bootstrap_from=MID_VAR_UNIFORM_REFERENCE,
                sample_sizes=DEFAULT_DATA_SIZE,
                nr_bootstraps=NR_BOOTSTRAPS,
                CLUSTER_CPUS=CLUSTER_CPUS
        )
    )
    bstrcomp.run_comparison()
    bstrcomp.pickle()




if __name__ == "__main__":
    #increase_obs_data_small_var(alg=NoTears())
    #increase_obs_data_mid_var(alg=NoTears())
    #increase_variables(alg=NoTears())
    #increase_colors(alg=NoTears())
    increase_hyperparameter(cls=NoTears, change_param_dict={'lambda1': [0.01, 0.1, 0.5, 1, 2, 4, 8]})
    #standardized_data_comparison(alg=NoTears())
    #dag_versus_cpdag_comparison()

