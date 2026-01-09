#!/bin/bash
#SBATCH --job-name=cgm_only
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=100000
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH -A shakeri-lab
#SBATCH --array=0-9
#SBATCH --output=logs/train-out_%A_%a.log
#SBATCH --error=logs/train-err_%A_%a.log

# MODEL_NAMES=("TimeMixer" "TimeMixerPP" "ModernTCN" "TSLANet" "TEFN" "TOTEM" "GPT4TS")
MODEL_NAMES=("SAITS" "FreTS" "CSDI" "SCINet" "TimeMixer" "TimeMixerPP" "TSLANet" "TEFN" "TOTEM" "GPT4TS")
# MODEL_NAMES=("SAITS" "FreTS" "SCINet" "TimeMixer" "TSLANet" "TEFN" "TOTEM" "GPT4TS")
# MODEL_NAMES=("CSDI")

PARAM_RANGE_FILE="param_range.json"
CONFIG_FILE="config.yml"
NUM_TRIALS=40

mkdir -p "logs"

model=${MODEL_NAMES[$SLURM_ARRAY_TASK_ID]}

echo "Starting Hyperparameter Optimization for Model: $model"
echo "-------------------------------------------------------"


# Activate environment
source /project/shakeri-lab/Amir/py_env/bin/activate

python hyperparameter_engine.py \
    --model_name "$model" \
    --ParamRangeDir "$PARAM_RANGE_FILE" \
    --config-path "$CONFIG_FILE" \
    --NTrials $NUM_TRIALS \


# --train_best
# --is_evaluate

if [ $? -eq 0 ]; then
    echo "Optimization for $model completed successfully."
else
    echo "Optimization for $model failed."
    exit 1
fi
