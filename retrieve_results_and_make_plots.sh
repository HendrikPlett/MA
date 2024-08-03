# Run in the current shell using "source retrieve_results_and_make_plots.sh"

LOCAL_HOME_PATH="/Users/hendrikplett/Desktop/3. Uni/3.Master_Statistik/4. Semester/Masterarbeit/3. Code"

# Retrieve results from EULER
cd "$LOCAL_HOME_PATH"
rsync -avz --include='*.pkl' --exclude='*' hplett@euler.ethz.ch:/cluster/home/hplett/MA/exp/results/ ./MA/exp/results/

# Create plots
cd "$LOCAL_HOME_PATH"
cd MA/exp/vis
conda activate masterthesis
python3 create_plots.py
