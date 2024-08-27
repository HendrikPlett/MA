#!bin/bash
# Creates independent sbatch jobs for each passed python script
submit_job() {
    local script="$1"
    local cpus="$2"
    local mem_per_cpu="$3"
    local time="$4"
    
    sbatch \
        --job-name="$(basename "$script" .py)" \
        --ntasks="$cpus" \
        --ntasks-per-node="$cpus" \
        --cpus-per-task=1 \
        --mem-per-cpu="$mem_per_cpu" \
        --time="$time" \
        --mail-type=BEGIN,END \
        --mail-user="hplett@student.ethz.ch" \
        --wrap="python3 $script"
    
    echo "Submitted job $script with $cpus CPUs, ${mem_per_cpu}MB per CPU, and $time time limit"
}

# Set up environment
module load stack/2024-04 gcc/8.5.0 python/3.9.18
source $HOME/maplett/bin/activate

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""

# Submit the wanted scripts
submit_job "scripts/exp_pc.py" 52 2048 "00:20:00"
submit_job "scripts/exp_pc_kci.py" 102 4096 "02:00:00"
submit_job "scripts/exp_ges.py" 52 2048 "00:20:00"
submit_job "scripts/exp_notears.py" 52 4096 "00:40:00"
submit_job "scripts/exp_golem.py" 22 8192 "04:00:00"
submit_job "scripts/exp_icp_small_var.py" 52 2048 "00:20:00"
submit_job "scripts/exp_score_based_comparison.py" 22 8192 "06:00:00"
submit_job "scripts/exp_GniES_comparison.py" 102 2048 "02:00:00"
submit_job "scripts/exp_bootstrap_effect.py" 52 2028 "00:20:00"
submit_job "scripts/exp_gies.py" 52 2048 "01:00:00"
submit_job "scripts/exp_gnies.py" 102 2048 "12:00:00"
submit_job "scripts/exp_utigsp.py" 102 2048 "12:00:00"