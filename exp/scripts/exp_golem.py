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
    increase_hyperparameter,
    standardized_data_comparison,
    # Benchmarking classes
    BootstrapComparison,
    Bootstrap,
    # Algorithm 
    Golem
)


def ev_nv_dag_cpdag_comparison(processes: int):
    bstrcomp = BootstrapComparison("Golem-EVversusNV")
    bstrcomp.add_bootstrap(
        Bootstrap(
                name=f"EV",
                true_dag=MID_VAR_TRUE_DAG,
                algorithm=Golem(equal_variances=True, return_cpdag=False),
                data_to_bootstrap_from=MID_VAR_UNIFORM_REFERENCE,
                sample_sizes=DEFAULT_DATA_SIZE,
                nr_bootstraps=NR_BOOTSTRAPS, 
                PROCESSES=processes
        )
    )
    bstrcomp.add_bootstrap(
        Bootstrap(
                name=f"NV",
                true_dag=MID_VAR_TRUE_DAG,
                algorithm=Golem(equal_variances=False, return_cpdag=False),
                data_to_bootstrap_from=MID_VAR_UNIFORM_REFERENCE,
                sample_sizes=DEFAULT_DATA_SIZE,
                nr_bootstraps=NR_BOOTSTRAPS, 
                PROCESSES=processes
        )
    )
    """
    bstrcomp.add_bootstrap(
        Bootstrap(
                name=f"EV-Return CPDAG",
                true_dag=MID_VAR_TRUE_DAG,
                algorithm=Golem(equal_variances=True, return_cpdag=True),
                data_to_bootstrap_from=MID_VAR_UNIFORM_REFERENCE,
                sample_sizes=DEFAULT_DATA_SIZE,
                nr_bootstraps=NR_BOOTSTRAPS, 
                PROCESSES=processes
        )
    )
    bstrcomp.add_bootstrap(
        Bootstrap(
                name=f"NV-Return CPDAG",
                true_dag=MID_VAR_TRUE_DAG,
                algorithm=Golem(equal_variances=False, return_cpdag=True),
                data_to_bootstrap_from=MID_VAR_UNIFORM_REFERENCE,
                sample_sizes=DEFAULT_DATA_SIZE,
                nr_bootstraps=NR_BOOTSTRAPS, 
                PROCESSES=processes
        )
    )
    """
    bstrcomp.run_comparison()
    bstrcomp.pickle()



# GOLEM specific: 
PROCESSES = 20

if __name__ == "__main__":
    # Trick: Run one process per core and not too many processes, around 10 and at most 200 Golem fits.

    # increase_obs_data_mid_var(alg=Golem(equal_variances=True), processes=10)
    # increase_obs_data_mid_var(alg=Golem(equal_variances=False), processes=10)
    # increase_variables(alg=Golem(equal_variances=False), processes=PROCESSES)
    # increase_colors(alg=Golem(equal_variances=False), processes=PROCESSES)
    # TODO: increase_hyperparameter(cls=Golem, change_param_dict={'lambda1': [0.01, 0.1, 0.5, 1, 2, 4, 8]}, fix_param_dict={'equal_variances': False})
    # standardized_data_comparison(alg=Golem(equal_variances=True), processes=PROCESSES)
    # standardized_data_comparison(alg=Golem(equal_variances=False), processes=PROCESSES)
    ev_nv_dag_cpdag_comparison(processes=PROCESSES)
