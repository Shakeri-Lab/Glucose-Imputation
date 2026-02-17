#!/bin/bash
#SBATCH --job-name=full_ablation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH -A cdt_computing
#SBATCH --array=0-3
#SBATCH --output=new_logs/full_ablation_%A_%a.log
#SBATCH --error=new_logs/full_ablation_err_%A_%a.log

# ================= CONFIGURATION =================
MODEL_NAMES=("SAITS" "FreTS" "SCINet" "Lerp")
# MODEL_NAMES=("Lerp")

PARAM_RANGE="param_range.json"
CONFIG_FILE="config.yml"
SAVING_PATH="./DAY_NIGHT"
NUM_TRIALS=40

# --- DATA PATHS ---
ALIREZA_PATH="/project/shakeri-lab/Alireza_timeseries/benchmark/datasets_processed/PEDAP/splits/pedap_patients_70_10_20_seed42/"
SIM_PATH="/project/shakeri-lab/Amir/Data_Processing/Sim/sim/UVA-T1D-Simulator/TestAID/NewImputationModerateConfig/"
TCR_PATH="/project/shakeri-lab/Amir/CGM_Imputation/tcr_data/"

# --- SCENARIO PARAMS ---
A_RATIOS=(0.1 0.2 0.3 0.5)
A_LENGTHS=(10 30 60)
B_MEALS=(1 2 3)
B_MIN_LENS=(1 3.5 5)
B_MAX_LENS=(2 4 6)
C_HYPO_LENS=(30 60 120)


# A_RATIOS=(0.1)
# A_LENGTHS=(10)
# B_MEALS=(1)
# B_MIN_LENS=(1)
# B_MAX_LENS=(2)
# C_HYPO_LENS=(30)


# =================================================
export PYTHONHASHSEED=7
mkdir -p "new_logs"

model=${MODEL_NAMES[$SLURM_ARRAY_TASK_ID]}
echo "Starting Full Ablation for Model: $model"

source /project/shakeri-lab/Amir/py_env/bin/activate

# =================================================
#  PART 1: RUN SCENARIOS A & B (Alireza + Sim)
# =================================================
# Format: DATA_PATH|IS_PEDAP_FLAG|DATA_CATEGORY
DATASETS_AB=(
  "${SIM_PATH}||simulate"
  "${ALIREZA_PATH}|--is_pedap|pedap"
  "${ALIREZA_PATH}|--is_pedap|none_pedap"
)

for entry in "${DATASETS_AB[@]}"; do
  IFS='|' read -r DATA_PATH IS_PEDAP DATA_CAT <<< "$entry"
  
  echo "=================================================="
  echo "Running A & B on: $DATA_PATH"
  echo "Flag: $IS_PEDAP"
  echo "Data Category: $DATA_CAT"
  echo "=================================================="
  
  # --- Scenario A ---
  for ratio in "${A_RATIOS[@]}"; do
    for length in "${A_LENGTHS[@]}"; do
      python hyperparameter_engine.py \
          --model_name "$model" --ParamRangeDir "$PARAM_RANGE" --config-path "$CONFIG_FILE" \
          --NTrials $NUM_TRIALS --is_evaluate --is_ablation --saving_path "$SAVING_PATH" \
          --type "A" --data_path "$DATA_PATH" $IS_PEDAP --data_category "$DATA_CAT" \
          --protocol_mask_ratio $ratio --window_scenario_A_length $length
    done
  done
  
  # --- Scenario B ---
  for meal in "${B_MEALS[@]}"; do
    for min_l in "${B_MIN_LENS[@]}"; do
      for max_l in "${B_MAX_LENS[@]}"; do
        if (( $(echo "$min_l < $max_l" | bc -l) )); then
          python hyperparameter_engine.py \
              --model_name "$model" --ParamRangeDir "$PARAM_RANGE" --config-path "$CONFIG_FILE" \
              --NTrials $NUM_TRIALS --is_evaluate --is_ablation --saving_path "$SAVING_PATH" \
              --type "B" --data_path "$DATA_PATH" $IS_PEDAP --data_category "$DATA_CAT" \
              --num_meal_hide $meal --min_length_B $min_l --max_length_B $max_l
        fi
      done
    done
  done
done



# =================================================
#  PART 2: RUN SCENARIO C (TCR Data ONLY)
# =================================================
echo "=================================================="
echo "Running Scenario C on: TCR Dataset"
echo "Path: $TCR_PATH"
echo "Data Category: simulate"
echo "=================================================="

for hypo in "${C_HYPO_LENS[@]}"; do
    python hyperparameter_engine.py \
        --model_name "$model" --ParamRangeDir "$PARAM_RANGE" --config-path "$CONFIG_FILE" \
        --NTrials $NUM_TRIALS --is_evaluate --is_ablation --saving_path "$SAVING_PATH" \
        --type "C" --data_path "$TCR_PATH" --data_category "simulate" \
        --hypo_length $hypo
done

echo "All tasks completed for $model."