PRED=$1
API_KEY=$2

# Define the ordered list of JSON files to evaluate
files_to_evaluate=(
    "clean.json"
    "bright.json"
    "lowlight.json"
    "snow.json"
    "fog.json"
    "rain.json"
    "lens.json"
    "water.json"
    "cameracrash.json"
    "framelost.json"
    "saturate.json"
    "motion.json"
    "zoom.json"
    "biterror.json"
    "colorquant.json"
    "h256.json"
)

# Loop over the ordered list and evaluate each file
for filename in "${files_to_evaluate[@]}"; do
    file_path="${PRED}/${filename}"
    # Check if the file exists in the directory
    if [[ -f "$file_path" ]]; then
        # Run the evaluation command on each file
        python evaluate/eval_prediction.py \
            "$file_path" \
            ../data/QA_dataset_nus/drivelm_train_200_final_v4_norm_abc.json \
            --api_key "$API_KEY"
    else
        echo "Warning: $file_path not found, skipping..."
    fi
done