from exp_assistant import (
    # Precreated datasets/true DAGs
    MID_VAR_TRUE_DAG,
    MID_VAR_UNIFORM_REFERENCE_1000_SAMPLE,
    MID_VAR_UNIFORM_REFERENCE_2000_SAMPLE,
    MID_VAR_UNIFORM_REFERENCE_3000_SAMPLE,
    MID_VAR_UNIFORM_REFERENCE,
    # Other useful constants
    NR_BOOTSTRAPS,
    DEFAULT_DATA_SIZE,
    # Benchmarking classes
    BootstrapComparison,
    Bootstrap,
    # Algorithm 
    GES
)

PROCESSES = 50


def test_empirical_distribution_size(processes: int = PROCESSES):
    bstrpcomp = BootstrapComparison("Influence-EmpiricalDistributionSize-on-GES")
    datasets = [MID_VAR_UNIFORM_REFERENCE_1000_SAMPLE, 
                 MID_VAR_UNIFORM_REFERENCE_2000_SAMPLE,
                 MID_VAR_UNIFORM_REFERENCE_3000_SAMPLE,
                 MID_VAR_UNIFORM_REFERENCE] # All lists with one df in them each
    names = ["N=1000", "N=2000", "N=3000", "N=10000"]
    for data, name in zip(datasets, names):
        bstrpcomp.add_bootstrap(
            Bootstrap(
                name=name,
                true_dag=MID_VAR_TRUE_DAG,
                algorithm=GES(),
                data_to_bootstrap_from=data,
                sample_sizes=DEFAULT_DATA_SIZE*len(data),
                nr_bootstraps=NR_BOOTSTRAPS,
                PROCESSES = processes
            )
        )
    bstrpcomp.run_comparison()
    bstrpcomp.pickle()


if __name__ == "__main__":
    test_empirical_distribution_size()