#!/bin/bash
#SBATCH --job-name=my_python_job        # Job name
#SBATCH --output=output_%j.log          # Standard output file (%j expands to job ID)
#SBATCH --error=error_%j.log            # Error log file
#SBATCH --gpus=a100-40
#SBATCH --time=00:10:00                  # Time limit (hh:mm:ss)

# Load modules if needed
# module load python/3.11 cuda/11.7  # Adjust based on your cluster setup

# Activate venv
source venv/bin/activate

# Run your Python script
echo "lr=0.01 epochs=60 hidden_size=256"
python lstm_basic.py 0.01 60

# python lstm_advanced.py data/glove.6B.50d.pkl out_file
