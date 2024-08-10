from exp_assistant import (
    # Precreated datasets/true DAGs
    MID_VAR_TRUE_DAG,
    MID_VAR_UNIFORM_REFERENCE_2000_SAMPLE,
    MID_VAR_UNIFORM_REFERENCE_3000_SAMPLE,
    MID_INTERVENTIONS_COLORS_VARIABLES,
    MID_INTERVENTIONS_COLORS_DATASETS_MID_VAR,
    STRONG_INTERVENTIONS_COLORS_VARIABLES,
    STRONG_INTERVENTIONS_COLORS_DATASETS_MID_VAR,
    STRONG_INTERVENTIONS_THETA_VARIABLES,
    STRONG_INTERVENTIONS_THETA_DATASETS_MID_VAR,
    # Other useful constants
    NR_BOOTSTRAPS,
    # Benchmarking classes
    BootstrapComparison,
    Bootstrap,
    # Algorithm 
    GES, GNIES, GIES
)

PROCESSES = 100
SAMPLE_SIZE = [1000]


def compare_all(obs_ds, interventions, intervention_ds, name, processes = PROCESSES):
    bstrpcomp = BootstrapComparison(name=f"GES-GNIES-GIES-Comparison-{name}")
    bstrpcomp.add_bootstrap(
        Bootstrap(
            name=GES.__name__,
            true_dag=MID_VAR_TRUE_DAG,
            algorithm=GES(),
            data_to_bootstrap_from=obs_ds,
            sample_sizes=[SAMPLE_SIZE[0]*len(intervention_ds)], # Ensures each algorithm gets same number of samples
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
            data_to_bootstrap_from=intervention_ds,
            sample_sizes=SAMPLE_SIZE*len(intervention_ds),
            nr_bootstraps=NR_BOOTSTRAPS,
            PROCESSES=processes
        )
    )
    bstrpcomp.run_comparison()
    bstrpcomp.pickle()



if __name__ == "__main__":
    compare_all(
        obs_ds=MID_VAR_UNIFORM_REFERENCE_3000_SAMPLE, # List with one df 
        interventions=MID_INTERVENTIONS_COLORS_VARIABLES,
        intervention_ds=MID_INTERVENTIONS_COLORS_DATASETS_MID_VAR, # List with three dfs
        name="MidRGBinterventions"
    )
    compare_all(
        obs_ds=MID_VAR_UNIFORM_REFERENCE_3000_SAMPLE,
        interventions=STRONG_INTERVENTIONS_COLORS_VARIABLES,
        intervention_ds=STRONG_INTERVENTIONS_COLORS_DATASETS_MID_VAR,
        name="StrongRGBinterventions"
    )
    compare_all(
        obs_ds=MID_VAR_UNIFORM_REFERENCE_2000_SAMPLE,
        interventions=STRONG_INTERVENTIONS_THETA_VARIABLES,
        intervention_ds=STRONG_INTERVENTIONS_THETA_DATASETS_MID_VAR,
        name="StrongThetaInterventions"
    )
    
