from exp_assistant import (
    # Precreated datasets/true DAGs
    MID_VAR_TRUE_DAG,
    MID_VAR_UNIFORM_REFERENCE_3000_SAMPLE,
    MID_INTERVENTIONS_COLORS_VARIABLES,
    MID_INTERVENTIONS_COLORS_DATASETS_MID_VAR,
    STRONG_INTERVENTIONS_COLORS_VARIABLES,
    STRONG_INTERVENTIONS_COLORS_DATASETS_MID_VAR,
    # Other useful constants
    NR_BOOTSTRAPS,
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
    GES, GNIES, GIES
)

PROCESSES = 50
SAMPLE_SIZE = [300]


def compare_all(obs_ds, interventions, intervention_ds, name, processes = PROCESSES):
    bstrpcomp = BootstrapComparison(name=f"GES-GNIES-GIES-Comparison-{name}")
    bstrpcomp.add_bootstrap(
        Bootstrap(
            name=GES.__name__,
            true_dag=MID_VAR_TRUE_DAG,
            algorithm=GES(),
            data_to_bootstrap_from=obs_ds,
            sample_sizes=SAMPLE_SIZE,
            nr_bootstraps=NR_BOOTSTRAPS,
            PROCESSES=processes
        )
    )
    bstrpcomp.add_bootstrap(
        Bootstrap(
            name=GNIES.__name__,
            true_dag=MID_VAR_TRUE_DAG,
            algorithm=GNIES(),
            data_to_bootstrap_from=intervention_ds,
            sample_sizes=SAMPLE_SIZE*len(intervention_ds),
            nr_bootstraps=NR_BOOTSTRAPS,
            PROCESSES=processes
        )
    )
    bstrpcomp.add_bootstrap(
        Bootstrap(
            name=GIES.__name__,
            true_dag=MID_VAR_TRUE_DAG,
            algorithm=GIES(interventions=interventions),
            data_to_bootstrap_from=intervention_ds*len(intervention_ds),
            sample_sizes=SAMPLE_SIZE,
            nr_bootstraps=NR_BOOTSTRAPS,
            PROCESSES=processes
        )
    )
    
    bstrpcomp.run_comparison()
    bstrpcomp.pickle()



if __name__ == "__main__":
    compare_all(
        obs_ds=MID_VAR_UNIFORM_REFERENCE_3000_SAMPLE,
        interventions=MID_INTERVENTIONS_COLORS_VARIABLES,
        intervention_ds=MID_INTERVENTIONS_COLORS_DATASETS_MID_VAR,
        name="MidRGBinterventions"
    )
    compare_all(
        obs_ds=MID_VAR_UNIFORM_REFERENCE_3000_SAMPLE,
        interventions=STRONG_INTERVENTIONS_COLORS_VARIABLES,
        intervention_ds=STRONG_INTERVENTIONS_COLORS_DATASETS_MID_VAR,
        name="StrongRGBinterventions"
    )