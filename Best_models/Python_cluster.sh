#!/usr/bin/env python
#SBATCH --partition=day
#SBATCH --job-name=six_month_10_100
#SBATCH --ntasks=4 --nodes=4
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5G
#SBATCH --time=0-23:01:00
#SBATCH --mail-type=ALL

#################################################
# please DO NOT remove the following two commands
#################################################
module load StdEnv
export SLURM_EXPORT_ENV=ALL
#################################################

# Define your Python script and its arguments
SCRIPT="your_script.py"
ARGS=""  # Add any script arguments here

# Create a unique working directory for each job
WORK_DIR="job_${SLURM_JOB_ID}"
mkdir $WORK_DIR
cd $WORK_DIR

# Copy your script and input data to the working directory
cp "$SCRIPT" .
# cp -r /path/to/your/data/* .  # Copy data if needed

# Run your script in parallel
parallel -j $SLURM_NTASKS --delay 1 "$SCRIPT $ARGS" ::: $(seq $SLURM_NTASKS)

# Copy any output files back to the original directory
# cp -r results/* /path/to/your/output

# Clean up the working directory
cd ..
rm -rf $WORK_DIR

# Deactivate your Python environment (if needed)
# source deactivate

echo "Job finished"