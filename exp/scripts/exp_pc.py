from config import (
    # Precreated datasets/true DAGs
    SMALL_VAR_TRUE_DAG,
    MID_VAR_TRUE_DAG,
    SMALL_VAR_UNIFORM_REFERENCE,
    MID_VAR_UNIFORM_REFERENCE,
    RED_TRUE_DAG,
    RED_GREEN_TRUE_DAG,
    RED_UNIFORM_REFERENCE,
    RED_GREEN_UNIFORM_REFERENCE,
    # Other useful constants
    OBS_DATA_SIZES,
    DEFAULT_DATA_SIZE,
    # Benchmarking classes
    BootstrapComparison,
    Bootstrap,
    # Algorithm 
    PC
)

### ------------------------------------------------------
# Set default configurations for PC algorithm
ALPHA = 0.1

### ------------------------------------------------------
# Actual jobs

def increase_obs_data_small_var(sizes: list[int]):
    bstrpcomp = BootstrapComparison("PC-SmallVar-IncreaseObservationalData")
    for size in sizes:
        bstrpcomp.add_bootstrap(
            Bootstrap(
                name=f"n = {size}",
                true_dag=SMALL_VAR_TRUE_DAG,
                algorithm=PC(alpha=ALPHA),
                data_to_bootstrap_from=SMALL_VAR_UNIFORM_REFERENCE,
                sample_sizes=[size],
            )
        )
    bstrpcomp.run_comparison()
    bstrpcomp.pickle()


def increase_obs_data_mid_var(sizes: list[int]):
    bstrpcomp = BootstrapComparison("PC-MidVar-IncreaseObservationalData")
    for size in sizes:
        bstrpcomp.add_bootstrap(
            Bootstrap(
                name=f"n = {size}",
                true_dag=MID_VAR_TRUE_DAG,
                algorithm=PC(alpha=ALPHA),
                data_to_bootstrap_from=MID_VAR_UNIFORM_REFERENCE,
                sample_sizes=[size],
            )
        )
    bstrpcomp.run_comparison()
    bstrpcomp.pickle()

def increase_variables():
    names = ['Small Var', 'Med Var']
    true_dags = [SMALL_VAR_TRUE_DAG, MID_VAR_TRUE_DAG]
    datas = [SMALL_VAR_UNIFORM_REFERENCE, MID_VAR_UNIFORM_REFERENCE]
    bstrpcomp = BootstrapComparison("PC-IncreaseVariableCount")
    for name, true_dag, data in zip(names, true_dags, datas):
        bstrpcomp.add_bootstrap(
            Bootstrap(
                name=name,
                true_dag=true_dag,
                algorithm=PC(alpha=ALPHA),
                data_to_bootstrap_from=data,
                sample_sizes=DEFAULT_DATA_SIZE,
            )
        )
    bstrpcomp.run_comparison()
    bstrpcomp.pickle()

def increase_colors():
    names = ['Red', 'Red, Green', 'Red, Green, Blue']
    true_dags = [RED_TRUE_DAG, RED_GREEN_TRUE_DAG, MID_VAR_TRUE_DAG]
    datas = [RED_UNIFORM_REFERENCE, RED_GREEN_UNIFORM_REFERENCE, MID_VAR_UNIFORM_REFERENCE]
    bstrpcomp = BootstrapComparison("PC-IncreaseColorCount")
    for name, true_dag, data in zip(names, true_dags, datas):
        bstrpcomp.add_bootstrap(
            Bootstrap(
                name=name,
                true_dag=true_dag,
                algorithm=PC(alpha=ALPHA),
                data_to_bootstrap_from=data,
                sample_sizes=DEFAULT_DATA_SIZE,
            )
        )
    bstrpcomp.run_comparison()
    bstrpcomp.pickle()

def increase_alpha(alphas: list):
    bstrpcomp = BootstrapComparison("PC-IncreaseAlpha")
    for alpha in alphas:
        bstrpcomp.add_bootstrap(
            Bootstrap(
                name=f"Alpha: {alpha}",
                true_dag=MID_VAR_TRUE_DAG,
                algorithm=PC(alpha=alpha),
                data_to_bootstrap_from=MID_VAR_UNIFORM_REFERENCE,
                sample_sizes=DEFAULT_DATA_SIZE,
            )
        )
    bstrpcomp.run_comparison()
    bstrpcomp.pickle()


if __name__ == "__main__":
    increase_obs_data_small_var(sizes=OBS_DATA_SIZES)
    increase_obs_data_mid_var(sizes=OBS_DATA_SIZES)
    increase_variables()
    increase_colors()
    increase_alpha(alphas=[0.001, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6])

