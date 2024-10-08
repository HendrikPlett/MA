from exp_assistant import (
    # Precreated datasets/true DAGs
    MID_VAR_TRUE_DAG,
    MID_INTERVENTIONS_COLORS_VARIABLES,
    MID_INTERVENTIONS_COLORS_DATASETS_MID_VAR,
    STRONG_INTERVENTIONS_COLORS_VARIABLES,
    STRONG_INTERVENTIONS_COLORS_DATASETS_MID_VAR,
    STRONG_INTERVENTIONS_THETA_VARIABLES,
    STRONG_INTERVENTIONS_THETA_DATASETS_MID_VAR,
    # Other useful constants
    DEFAULT_DATA_SIZE,
    NR_BOOTSTRAPS,
    # Benchmarking classes
    BootstrapComparison,
    Bootstrap,
    # Algorithm 
    GIES
)

PROCESSES = 50
SAMPLE_SIZE = [1000]


def compare_interventions(interventions, int_datasets, names, processes: int = PROCESSES):
    assert len(interventions) == len(int_datasets) == len(names)
    bstrpcomp = BootstrapComparison(name="GIES-DifferentInterventions")
    for intvnt, dataset, name in zip(interventions, int_datasets, names):
        bstrpcomp.add_bootstrap(
            Bootstrap(
                name=name,
                true_dag=MID_VAR_TRUE_DAG,
                algorithm=GIES(interventions=intvnt),
                data_to_bootstrap_from=dataset,
                sample_sizes=DEFAULT_DATA_SIZE*len(dataset),
                nr_bootstraps=NR_BOOTSTRAPS,
                PROCESSES=processes
            )
        )
    bstrpcomp.run_comparison()
    bstrpcomp.pickle()


def increase_data(datasizes: list[int], processes: int = PROCESSES):
    bstrpcomp = BootstrapComparison(name="GIES-IncreaseSampleSize-midRGB-Interventions")
    for size in datasizes:
        bstrpcomp.add_bootstrap(
            Bootstrap(
                name=f"Samples: {size}",
                true_dag=MID_VAR_TRUE_DAG,
                algorithm=GIES(interventions=MID_INTERVENTIONS_COLORS_VARIABLES),
                data_to_bootstrap_from=MID_INTERVENTIONS_COLORS_DATASETS_MID_VAR,
                sample_sizes=[size]*len(MID_INTERVENTIONS_COLORS_DATASETS_MID_VAR),
                nr_bootstraps=NR_BOOTSTRAPS,
                PROCESSES=processes
            )
        )
    bstrpcomp.run_comparison()
    bstrpcomp.pickle()


if __name__ == "__main__":
    compare_interventions(
        interventions=[
            MID_INTERVENTIONS_COLORS_VARIABLES[:2],
            [MID_INTERVENTIONS_COLORS_VARIABLES[0], MID_INTERVENTIONS_COLORS_VARIABLES[2]],
            MID_INTERVENTIONS_COLORS_VARIABLES[1:],
            STRONG_INTERVENTIONS_THETA_VARIABLES
        ],
        int_datasets=[
            MID_INTERVENTIONS_COLORS_DATASETS_MID_VAR[:2],
            [MID_INTERVENTIONS_COLORS_DATASETS_MID_VAR[0], MID_INTERVENTIONS_COLORS_DATASETS_MID_VAR[2]],
            MID_INTERVENTIONS_COLORS_DATASETS_MID_VAR[1:],
            STRONG_INTERVENTIONS_THETA_DATASETS_MID_VAR
        ],
        names=[
            "IntRG", "IntRB", "IntGB", "IntP1P2"
        ]
    )

    increase_data(datasizes=[60,125,250,500,750,1000])