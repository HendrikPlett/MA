# Run in the current shell using "source run_exps_on_euler.sh"

ORIGINAL_DIR=$(pwd)
LOCAL_HOME_PATH="/Users/hendrikplett/Desktop/3. Uni/3.Master_Statistik/4. Semester/Masterarbeit/3. Code"

# Copy MA directory to EULER Cluster
cd "$LOCAL_HOME_PATH"
rsync -avzh --delete "MA/" "hplett@euler.ethz.ch:/cluster/home/hplett/MA"

# Login to EULER and submit jobs
ssh hplett@euler.ethz.ch << 'EOF'
cd "/cluster/home/hplett/MA/exp"
bash euler_setup.sh
bash submit_jobs.sh
EOF

# Go back to original directory
cd "$ORIGINAL_DIR"
