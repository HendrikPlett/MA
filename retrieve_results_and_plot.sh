# Run in the current shell using "source retrieve_results_and_make_plots.sh"

LOCAL_HOME_PATH="/Users/hendrikplett/Desktop/3. Uni/3.Master_Statistik/4. Semester/Masterarbeit/3. Code"
STRING_FILTER=$1 # Retrieve only pickle files that start with this passed String


# Retrieve results from EULER
cd "$LOCAL_HOME_PATH"

if [ -z "$STRING_FILTER" ]; then
    rsync -avz --include='*.pkl' --exclude='*' hplett@euler.ethz.ch:/cluster/home/hplett/MA/exp/results/ ./MA/exp/results/
else
    rsync -avz --include="$STRING_FILTER*.pkl" --exclude='*' hplett@euler.ethz.ch:/cluster/home/hplett/MA/exp/results/ ./MA/exp/results/
fi

# Create plots
cd "$LOCAL_HOME_PATH"
cd MA/exp/vis
conda activate masterthesis
python3 create_plots.py

# Delete .pkl files, they are permanently stored only on EULER
cd ../results
rm *.pkl

cd "$LOCAL_HOME_PATH"
cd MA