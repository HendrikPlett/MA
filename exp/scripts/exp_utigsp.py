from exp_assistant import (
    # Precreated datasets/true DAGs
    MID_VAR_TRUE_DAG,
    MID_VAR_UNIFORM_REFERENCE,
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
    UT_IGSP
)

PROCESSES = 100
SAMPLE_SIZE = [1000]
ALPHA_CI = 0.05
ALPHA_INV = 0.05


def compare_interventions(int_datasets, names, processes: int = PROCESSES):
    assert len(int_datasets) == len(names)
    bstrpcomp = BootstrapComparison(name="UTIGSP-ObservationalPlusDifferentInterventions")
    for dataset, name in zip(int_datasets, names):
        bstrpcomp.add_bootstrap(
            Bootstrap(
                name=name,
                true_dag=MID_VAR_TRUE_DAG,
                algorithm=UT_IGSP(alpha_ci=ALPHA_CI, alpha_inv=ALPHA_INV),
                data_to_bootstrap_from=dataset,
                sample_sizes=DEFAULT_DATA_SIZE*len(dataset),
                nr_bootstraps=NR_BOOTSTRAPS,
                PROCESSES=processes
            )
        )
    bstrpcomp.run_comparison()
    bstrpcomp.pickle()


def increase_data(datasizes: list[int], processes: int = PROCESSES):
    bstrpcomp = BootstrapComparison(name="UTIGSP-IncreaseSampleSize-OmidRGB-Interventions")
    ds = [*MID_VAR_UNIFORM_REFERENCE, *MID_INTERVENTIONS_COLORS_DATASETS_MID_VAR]
    for size in datasizes:
        bstrpcomp.add_bootstrap(
            Bootstrap(
                name=f"Samples: {size}",
                true_dag=MID_VAR_TRUE_DAG,
                algorithm=UT_IGSP(alpha_ci=ALPHA_CI, alpha_inv=ALPHA_INV),
                data_to_bootstrap_from=ds,
                sample_sizes=[size]*len(ds),
                nr_bootstraps=NR_BOOTSTRAPS,
                PROCESSES=processes
            )
        )
    bstrpcomp.run_comparison()
    bstrpcomp.pickle()

def increase_significance_level(increase: str, levels: list[float], processes: int = PROCESSES):
    assert increase in ["AlphaCI", "AlphaINV"]
    bstrpcomp = BootstrapComparison(name=f"UTIGSP-Increase{increase}-OmidRGB-Interventions")
    ds = [*MID_VAR_UNIFORM_REFERENCE, *MID_INTERVENTIONS_COLORS_DATASETS_MID_VAR]
    for lev in levels:
        if increase == "AlphaCI":
            alg = UT_IGSP(alpha_ci=lev, alpha_inv=ALPHA_INV)
        else:
            alg = UT_IGSP(alpha_ci=ALPHA_CI, alpha_inv=lev)
        bstrpcomp.add_bootstrap(
            Bootstrap(
                name=f"{increase}: {lev}",
                true_dag=MID_VAR_TRUE_DAG,
                algorithm=alg,
                data_to_bootstrap_from=ds,
                sample_sizes=DEFAULT_DATA_SIZE*len(ds),
                nr_bootstraps=NR_BOOTSTRAPS,
                PROCESSES=processes
            )
        )
    bstrpcomp.run_comparison()
    bstrpcomp.pickle()



if __name__ == "__main__":
    compare_interventions(
        int_datasets=[
            [*MID_VAR_UNIFORM_REFERENCE ,*MID_INTERVENTIONS_COLORS_DATASETS_MID_VAR[:2]],
            [*MID_VAR_UNIFORM_REFERENCE, MID_INTERVENTIONS_COLORS_DATASETS_MID_VAR[0], MID_INTERVENTIONS_COLORS_DATASETS_MID_VAR[2]],
            [*MID_VAR_UNIFORM_REFERENCE, *MID_INTERVENTIONS_COLORS_DATASETS_MID_VAR[1:]],
            [*MID_VAR_UNIFORM_REFERENCE, *STRONG_INTERVENTIONS_THETA_DATASETS_MID_VAR]
        ],
        names=[
            "O+IntRG", "O+IntRB", "O+IntGB", "O+IntP1P2"
        ]
    )

    increase_data(datasizes=[60,125,250,500,750,1000])
    increase_significance_level(increase="AlphaCI", levels=[0.001, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6])
    increase_significance_level(increase="AlphaINV", levels=[0.001, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6])