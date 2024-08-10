from exp_assistant import (
    # Precreated datasets/true DAGs
    SMALL_VAR,
    SMALL_VAR_TRUE_DAG,
    # Observational
    SMALL_VAR_UNIFORM_REFERENCE,
    SMALL_VAR_UNIFORM_REFERENCE_1000_SAMPLE,
    SMALL_VAR_UNIFORM_REFERENCE_2000_SAMPLE,
    SMALL_VAR_UNIFORM_REFERENCE_3000_SAMPLE,

    MID_INTERVENTIONS_COLORS_VARIABLES,
    MID_INTERVENTIONS_COLORS_DATASETS_SMALL_VAR,

    STRONG_INTERVENTIONS_COLORS_VARIABLES,
    STRONG_INTERVENTIONS_COLORS_DATASETS_SMALL_VAR,

    STRONG_INTERVENTIONS_THETA_VARIABLES,
    STRONG_INTERVENTIONS_THETA_DATASETS_SMALL_VAR,
    # Other useful constants
    NR_BOOTSTRAPS,
    DEFAULT_DATA_SIZE,
    LATEX_NAME,
    # Benchmarking classes
    BootstrapComparison,
    Bootstrap,
    # Algorithm 
    ICP
)

### ------------------------------------------------------
# Set default configurations for ICP algorithm
ALPHA = 0.1
PROCESSES = 50

def icp_predict_different_targets(targets: list[str], data: list, processes: int = PROCESSES, name: str = ""):
    bstrpcomp = BootstrapComparison(name=f"ICP-DifferentTargets-SameData-{name}")
    for target in targets:
        bstrpcomp.add_bootstrap(
            Bootstrap(
                name=f"Target: {LATEX_NAME(target)}",
                true_dag=SMALL_VAR_TRUE_DAG,
                algorithm=ICP(target=target, alpha=ALPHA),
                data_to_bootstrap_from=data,
                sample_sizes=DEFAULT_DATA_SIZE*len(data),
                nr_bootstraps=NR_BOOTSTRAPS,
                PROCESSES=processes
            )
        )
    bstrpcomp.run_comparison()
    bstrpcomp.pickle()

def icp_predict_same_target_using_different_data(target: str, datasets: list[list], 
                                                 datasets_desc: list[str], processes: int = PROCESSES, name: str = ""):
    bstrpcomp = BootstrapComparison(name=f"ICP-DifferentData-SameTarget-{name}")
    for dataset, describtion in zip(datasets, datasets_desc):
        bstrpcomp.add_bootstrap(
            Bootstrap(
                name=describtion,
                true_dag=SMALL_VAR_TRUE_DAG,
                algorithm=ICP(target=target, alpha=ALPHA),
                data_to_bootstrap_from=dataset,
                sample_sizes=DEFAULT_DATA_SIZE*len(dataset),
                nr_bootstraps=NR_BOOTSTRAPS,
                PROCESSES=processes
            )
        )
    bstrpcomp.run_comparison()
    bstrpcomp.pickle()

def icp_increase_alpha(target: str, dataset: list, alphas: list[float], processes: int = PROCESSES, name: str = ""):
    bstrpcomp = BootstrapComparison(name=f"ICP-IncreaseAlpha-{name}")
    for alpha in alphas:
        bstrpcomp.add_bootstrap(
            Bootstrap(
                name=f"Alpha: {alpha}",
                true_dag=SMALL_VAR_TRUE_DAG,
                algorithm=ICP(target=target, alpha=alpha),
                data_to_bootstrap_from=dataset,
                sample_sizes=DEFAULT_DATA_SIZE*len(dataset),
                nr_bootstraps=NR_BOOTSTRAPS,
                PROCESSES=processes
            )
        )
    bstrpcomp.run_comparison()
    bstrpcomp.pickle()


def increase_useless_observational_data_environments(target: str, dataset: list,
                                                     nr_observational_data: int, 
                                                     processes: int = PROCESSES,
                                                     name: str = ""):
    assert nr_observational_data <= 10
    bstrpcomp = BootstrapComparison(name=f"ICP-IncreaseObsEnvironments-{name}")
    obs_data = [SMALL_VAR_UNIFORM_REFERENCE[0].iloc[0:1000,:]]
    for nr in range(1, nr_observational_data):
        bstrpcomp.add_bootstrap(
            Bootstrap(
                name=f"ObsDataCount: {nr}",
                true_dag=SMALL_VAR_TRUE_DAG,
                algorithm=ICP(target=target, alpha=ALPHA),
                data_to_bootstrap_from=[*dataset, *obs_data],
                sample_sizes=DEFAULT_DATA_SIZE*(len(dataset)+len(obs_data)),
                nr_bootstraps=NR_BOOTSTRAPS,
                PROCESSES=processes
            )
        )
        obs_data.append(SMALL_VAR_UNIFORM_REFERENCE[0].iloc[nr*1000:(nr+1)*1000,:])
    bstrpcomp.run_comparison()
    bstrpcomp.pickle()
   

if __name__ == "__main__":
    """ SMALL_VAR = ["red", "green", "blue", "current", "ir_1", "vis_1", "ir_2", "vis_2", "ir_3", "vis_3"] """
    
    
    # --- 1) experiment ---
    icp_predict_different_targets(targets=["current", "ir_1", "vis_1"], 
                                  data=[*SMALL_VAR_UNIFORM_REFERENCE_1000_SAMPLE, *MID_INTERVENTIONS_COLORS_DATASETS_SMALL_VAR[:2]],
                                  name="Targets_c_ir1_vis1-Int_OmRmG"
                                  )
    

    # --- 2) experiment ---
    list_of_datasets2 = [
        [SMALL_VAR_UNIFORM_REFERENCE[0].iloc[0:1000, :], SMALL_VAR_UNIFORM_REFERENCE[0].iloc[1000:2000, :]],
        [*SMALL_VAR_UNIFORM_REFERENCE, *MID_INTERVENTIONS_COLORS_DATASETS_SMALL_VAR[:1]],
        [*SMALL_VAR_UNIFORM_REFERENCE, *MID_INTERVENTIONS_COLORS_DATASETS_SMALL_VAR[:2]],
        [*SMALL_VAR_UNIFORM_REFERENCE, *MID_INTERVENTIONS_COLORS_DATASETS_SMALL_VAR[:3]]
    ]
    describtion_of_datasets2 = [
        "Obs+Obs",
        "Obs+MidR",
        "Obs+MidRG",
        "Obs+MidRGB"
    ]
    icp_predict_same_target_using_different_data(target='ir_1',
                                                 datasets=list_of_datasets2,
                                                 datasets_desc=describtion_of_datasets2,
                                                 name="Target_ir1-Int_OO-OmR-OmRmG-OmRmGmB")
    
    # --- 3) experiment ---    
    icp_predict_same_target_using_different_data(target='vis_1',
                                                datasets=list_of_datasets2,
                                                datasets_desc=describtion_of_datasets2,
                                                name="Target_vis1-Int_OO-OmR-OmRmG-OmRmGmB")


    # --- 4) experiment ---    
    list_of_datasets3 = [
        [SMALL_VAR_UNIFORM_REFERENCE[0].iloc[0:1000, :], SMALL_VAR_UNIFORM_REFERENCE[0].iloc[1000:2000, :]],
        [*SMALL_VAR_UNIFORM_REFERENCE, MID_INTERVENTIONS_COLORS_DATASETS_SMALL_VAR[0]],
        [*SMALL_VAR_UNIFORM_REFERENCE, MID_INTERVENTIONS_COLORS_DATASETS_SMALL_VAR[0], STRONG_INTERVENTIONS_COLORS_DATASETS_SMALL_VAR[1]],
        [*SMALL_VAR_UNIFORM_REFERENCE, MID_INTERVENTIONS_COLORS_DATASETS_SMALL_VAR[0], STRONG_INTERVENTIONS_COLORS_DATASETS_SMALL_VAR[1], STRONG_INTERVENTIONS_COLORS_DATASETS_SMALL_VAR[-1]]
    ]
    describtion_of_datasets3 = [
        "Obs+Obs",
        "Obs+MidR",
        "Obs+MidR+StrongG",
        "Obs+MidR+StrongG+StrongB"
    ]
    icp_predict_same_target_using_different_data(target='ir_1',
                                                 datasets=list_of_datasets3,
                                                 datasets_desc=describtion_of_datasets3,
                                                 name="Target_ir1-Int_OO-OmR-OmRsG-OmRsGsB")


    
    # --- 5) experiment ---    
    icp_predict_same_target_using_different_data(target='vis_1',
                                                 datasets=list_of_datasets3,
                                                 datasets_desc=describtion_of_datasets3,
                                                 name="Target_vis1-Int_OO-OmR-OmRsG-OmRsGsB")


    # --- 6) experiment ---
    icp_increase_alpha(target="vis_1",
                       dataset=[*SMALL_VAR_UNIFORM_REFERENCE_1000_SAMPLE, *MID_INTERVENTIONS_COLORS_DATASETS_SMALL_VAR[:2]],
                       alphas=[0.001, 0.01, 0.05, 0.10, 0.2, 0.4, 0.8],
                       name="Target_vis1-Int_OmRmG"
                       )
    
    # --- 7) experiment ---
    increase_useless_observational_data_environments(target="vis_1",
                                                     dataset=MID_INTERVENTIONS_COLORS_DATASETS_SMALL_VAR[:2],
                                                     nr_observational_data=5,
                                                     name="Target_vis1-Int_5OmRmG")