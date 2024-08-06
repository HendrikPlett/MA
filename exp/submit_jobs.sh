#!bin/bash
# Creates independent sbatch jobs for each passed python script
submit_job() {
    local script="$1"
    local cpus="$2"
    local mem_per_cpu="$3"
    local time="$4"
    
    sbatch \
        --job-name="$(basename "$script" .py)" \
        --ntasks=1 \
        --cpus-per-task="$cpus" \
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

# Submit the wanted scripts
submit_job "scripts/exp_pc.py" 8 2048 "04:00:00"