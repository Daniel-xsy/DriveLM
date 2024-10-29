PRED=$1
PREFIX=$2

# Loop over all files in $PRED directory that start with $PREFIX
for file in ${PRED}/${PREFIX}*; do
    # Run the evaluation command on each file
    python evaluate/eval_behavior.py \
        "$file" \
        ../data/QA_dataset_nus/drivelm_val_norm_300.json \
        --thresh 0.05
done